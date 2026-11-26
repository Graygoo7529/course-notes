"""由融合结构约束的颜色平滑、明度分解和软色阶。"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .features import Structure
from .pipeline import FloatImage, Mask, _float_array, _mask, _number, _unit_array


@dataclass(frozen=True)
class ColorConfig:
    levels: int = 6
    amount: float = 0.35
    transition: float = 0.25
    detail: float = 0.35
    luma_smoothness: float = 8.0
    chroma_smoothness: float = 1.5
    tau_lightness: float = 8.0
    tau_chroma: float = 12.0
    tau_gradient: float = 0.12
    solver_tolerance: float = 1e-5
    solver_iterations: int = 250

    def validate(self) -> None:
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or self.levels < 2:
            raise ValueError("levels 必须是至少为 2 的整数。")
        _number(self.amount, "amount", 0, 1)
        _number(self.transition, "transition", float(np.finfo(np.float32).eps))
        if self.transition > 0.5:
            raise ValueError("transition 必须位于 (0, 0.5]。")
        _number(self.detail, "detail", 0, 1)
        _number(self.luma_smoothness, "luma_smoothness", 0)
        _number(self.chroma_smoothness, "chroma_smoothness", 0)
        _number(self.tau_lightness, "tau_lightness", float(np.finfo(np.float32).eps))
        _number(self.tau_chroma, "tau_chroma", float(np.finfo(np.float32).eps))
        _number(self.tau_gradient, "tau_gradient", float(np.finfo(np.float32).eps))
        _number(self.solver_tolerance, "solver_tolerance", float(np.finfo(np.float32).eps))
        if (isinstance(self.solver_iterations, bool) or
                not isinstance(self.solver_iterations, int) or self.solver_iterations < 1):
            raise ValueError("solver_iterations 必须是正整数。")


@dataclass(frozen=True)
class ColorDebug:
    base: FloatImage
    detail: FloatImage
    mapped: FloatImage
    boundary: FloatImage
    weight_right: FloatImage
    weight_down: FloatImage
    solver_residual: float
    solver_iterations: int


def build_boundary_weights(
    guide: FloatImage, structure: Structure, mask: Mask, config: ColorConfig,
) -> tuple[FloatImage, FloatImage, FloatImage]:
    """相似颜色且沿邻接方向变化较小时，允许较强平滑。"""
    config.validate()
    _unit_array(guide, "guide", 3)
    _mask(mask, guide.shape)
    if structure.strength.shape != mask.shape:
        raise ValueError("structure 与遮罩的尺寸必须一致。")
    lab = cv2.cvtColor(guide, cv2.COLOR_RGB2Lab).astype(np.float32)
    scale = np.array(
        [config.tau_lightness, config.tau_chroma, config.tau_chroma], dtype=np.float32,
    )
    difference = (lab[1:, :, :] - lab[:-1, :, :]) / scale
    color_down = np.sum(difference * difference, axis=2)
    difference = (lab[:, 1:, :] - lab[:, :-1, :]) / scale
    color_right = np.sum(difference * difference, axis=2)
    gradient_right = (
        (structure.jxx[:, :-1] + structure.jxx[:, 1:]) * 0.5 /
        (config.tau_gradient**2)
    )
    gradient_down = (
        (structure.jyy[:-1, :] + structure.jyy[1:, :]) * 0.5 /
        (config.tau_gradient**2)
    )
    right = np.exp(-0.5 * (color_right + gradient_right))
    down = np.exp(-0.5 * (color_down + gradient_down))
    right *= mask[:, :-1] & mask[:, 1:]
    down *= mask[:-1, :] & mask[1:, :]
    right, down = right.astype(np.float32), down.astype(np.float32)
    total = np.zeros(mask.shape, dtype=np.float32)
    count = np.zeros(mask.shape, dtype=np.float32)
    total[:, :-1] += right
    total[:, 1:] += right
    count[:, :-1] += mask[:, :-1]
    count[:, 1:] += mask[:, 1:]
    total[:-1, :] += down
    total[1:, :] += down
    count[:-1, :] += mask[:-1, :]
    count[1:, :] += mask[1:, :]
    boundary = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return right, down, boundary


def _apply_system(
    values: NDArray[np.float64], right: FloatImage, down: FloatImage,
    mask: Mask, smoothness: float,
) -> NDArray[np.float64]:
    result = values.copy()
    horizontal = smoothness * right * (values[:, :-1] - values[:, 1:])
    result[:, :-1] += horizontal
    result[:, 1:] -= horizontal
    vertical = smoothness * down * (values[:-1, :] - values[1:, :])
    result[:-1, :] += vertical
    result[1:, :] -= vertical
    result[~mask] = 0
    return result


def edge_aware_smooth(
    source: FloatImage, right: FloatImage, down: FloatImage, mask: Mask,
    smoothness: float, tolerance: float = 1e-5, max_iterations: int = 250,
) -> tuple[FloatImage, float, int]:
    """Jacobi 预条件共轭梯度；返回图像、真实相对残差和迭代数。

    四邻接差分隐式实现稀疏矩阵，背景无边且 RHS 为零，等价于只解前景。
    迭代及残差检查用 float64，最后转回接口约定的 float32。
    """
    _float_array(source, "source")
    _mask(mask, source.shape)
    if source.ndim != 2:
        raise ValueError("edge_aware_smooth 每次处理一个单通道平面。")
    if right.shape != (source.shape[0], max(0, source.shape[1] - 1)):
        raise ValueError("right 权重形状不正确。")
    if down.shape != (max(0, source.shape[0] - 1), source.shape[1]):
        raise ValueError("down 权重形状不正确。")
    for name, weights in (("right", right), ("down", down)):
        if (weights.dtype != np.float32 or not np.isfinite(weights).all() or
                np.any(weights < 0) or np.any(weights > 1)):
            raise ValueError(f"{name} 权重必须位于 [0,1]。")
    if (np.any(right[~(mask[:, :-1] & mask[:, 1:])] != 0) or
            np.any(down[~(mask[:-1] & mask[1:])] != 0)):
        raise ValueError("平滑权重不能跨越人物遮罩。")
    _number(smoothness, "smoothness", 0)
    _number(tolerance, "tolerance", float(np.finfo(np.float32).eps))
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations 必须是正整数。")
    if smoothness == 0:
        return np.where(mask, source, 0).astype(np.float32), 0.0, 0

    target = np.where(mask, source, 0).astype(np.float64)
    estimate = target.copy()
    diagonal = np.ones(source.shape, dtype=np.float64)
    diagonal[:, :-1] += smoothness * right
    diagonal[:, 1:] += smoothness * right
    diagonal[:-1] += smoothness * down
    diagonal[1:] += smoothness * down
    residual = target - _apply_system(estimate, right, down, mask, smoothness)
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    product = float(np.sum(residual * preconditioned))
    target_norm = max(float(np.sqrt(np.sum(target * target))), 1e-12)
    relative = float(np.sqrt(np.sum(residual * residual)) / target_norm)
    if not np.isfinite(relative):
        raise ArithmeticError("保边平滑产生非有限残差，请检查输入和参数尺度。")
    iterations = 0
    while relative > tolerance and iterations < max_iterations:
        applied = _apply_system(direction, right, down, mask, smoothness)
        denominator = float(np.sum(direction * applied, dtype=np.float64))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ArithmeticError("保边平滑线性系统失去正定性。")
        step = product / denominator
        estimate += step * direction
        residual -= step * applied
        relative = float(np.sqrt(np.sum(residual * residual)) / target_norm)
        iterations += 1
        if relative <= tolerance:
            residual = target - _apply_system(estimate, right, down, mask, smoothness)
            relative = float(np.sqrt(np.sum(residual * residual)) / target_norm)
            if relative <= tolerance:
                break
        preconditioned = residual / diagonal
        next_product = float(np.sum(residual * preconditioned))
        direction = preconditioned + (next_product / product) * direction
        product = next_product
    if not np.isfinite(relative) or relative > tolerance:
        raise ArithmeticError(
            f"保边平滑未收敛：相对残差 {relative:.3g}，迭代 {iterations}/{max_iterations}。"
        )
    estimate[~mask] = 0
    return estimate.astype(np.float32), relative, iterations


def map_lightness(
    base: FloatImage, levels: int, amount: float, transition: float,
) -> FloatImage:
    """单调有界的连续色阶映射；amount=0 时精确保留基础明度。"""
    _float_array(base, "base")
    if base.ndim != 2 or np.any(base < 0) or np.any(base > 1):
        raise ValueError("base 必须是 [0,1] 内的单通道明度。")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
        raise ValueError("levels 必须是至少为 2 的整数。")
    _number(amount, "amount", 0, 1)
    _number(transition, "transition", float(np.finfo(np.float32).eps))
    if transition > 0.5:
        raise ValueError("transition 必须位于 (0, 0.5]。")
    spacing = 1.0 / (levels - 1)
    half_width = transition * spacing
    quantized = np.zeros_like(base)
    for index in range(levels - 1):
        threshold = (index + 0.5) * spacing
        position = np.clip((base - threshold + half_width) / (2 * half_width), 0, 1)
        quantized += spacing * position * position * (3 - 2 * position)
    return np.clip((1 - amount) * base + amount * quantized, 0, 1).astype(np.float32)


def rebuild_colors(lightness: FloatImage, chroma: FloatImage, mask: Mask) -> FloatImage:
    """归一化明度与原生 Lab 色度重建 RGB；不修改输入，不乘 alpha。"""
    _unit_array(lightness, "lightness")
    _float_array(chroma, "chroma")
    _mask(mask, lightness.shape)
    if chroma.shape != (*lightness.shape, 2):
        raise ValueError("chroma 必须为 H×W×2 的 Lab a*、b*。")
    lab = np.concatenate((100 * lightness[..., None], chroma), axis=2)
    colors = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    colors[~mask] = 0
    return colors


def make_colors(
    image: FloatImage, mask: Mask, structure: Structure,
    config: ColorConfig = ColorConfig(),
) -> tuple[FloatImage, ColorDebug]:
    """原图取色，由共享结构决定哪里平滑、怎样简化明度。"""
    config.validate()
    _unit_array(image, "image", 3)
    _mask(mask, image.shape)
    source_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    lightness = np.clip(source_lab[..., 0] / 100, 0, 1).astype(np.float32)
    right, down, boundary = build_boundary_weights(
        structure.guide, structure, mask, config,
    )
    base, residual, iterations = edge_aware_smooth(
        lightness, right, down, mask, config.luma_smoothness,
        config.solver_tolerance, config.solver_iterations,
    )
    base = np.clip(base, 0, 1)
    detail = np.where(mask, lightness - base, 0).astype(np.float32)
    mapped = np.clip(
        map_lightness(base, config.levels, config.amount, config.transition) +
        config.detail * detail,
        0, 1,
    ).astype(np.float32)
    # 色度使用同一套边界，但采用更温和的平滑强度。
    chroma = source_lab[..., 1:].astype(np.float32)
    smooth_chroma = np.empty_like(chroma)
    chroma_residuals = [residual]
    chroma_iterations = [iterations]
    for channel in range(2):
        smooth_chroma[..., channel], residual_c, iterations_c = edge_aware_smooth(
            chroma[..., channel], right, down, mask, config.chroma_smoothness,
            config.solver_tolerance, config.solver_iterations,
        )
        chroma_residuals.append(residual_c)
        chroma_iterations.append(iterations_c)
    colors = rebuild_colors(mapped, smooth_chroma, mask)
    debug = ColorDebug(
        base, detail, mapped, boundary, right, down,
        max(chroma_residuals), max(chroma_iterations),
    )
    return colors, debug
