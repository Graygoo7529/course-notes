"""共享图像接口与第一版对照：RGB/RGBA 为 float32 [0,1]，遮罩为 bool。

内部颜色约定为 RGB。所有处理均保留 H、W，不修改调用者传入的数组。
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

FloatImage = NDArray[np.float32]
Mask = NDArray[np.bool_]
Labels = NDArray[np.uint8]


@dataclass(frozen=True)
class Selection:
    """rect 为 (x,y,width,height)；labels 为 GrabCut 的 0/1/2/3 标签。

    两种初始化方式二选一。笔划标记应由外部交互层写入 labels，
    其中 0/1 为确定背景/前景，2/3 为可能背景/前景。
    """

    rect: tuple[int, int, int, int] | None = None
    labels: Labels | None = None
    iterations: int = 5
    seed: int = 0


@dataclass(frozen=True)
class Config:
    sigma: float = 1.5
    levels: int = 6
    amount: float = 0.65
    threshold: float = 0.035
    softness: float = 0.08
    strength: float = 0.8

    def validate(self) -> None:
        _number(self.sigma, "sigma", 0)
        _integer(self.levels, "levels", 2)
        _number(self.amount, "amount", 0, 1)
        _number(self.threshold, "threshold", 0)
        _number(self.softness, "softness", 0)
        if self.softness == 0:
            raise ValueError("softness 必须大于 0。")
        _number(self.strength, "strength", 0, 1)


def _number(value: float, name: str, low: float, high: float | None = None) -> None:
    if not np.isfinite(value) or value < low or (high is not None and value > high):
        limit = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ValueError(f"{name} 必须为有限数且满足 {limit}。")


def _integer(value: int, name: str, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError(f"{name} 必须是 >= {low} 的整数。")


def _unit_array(array: FloatImage, name: str, channels: int | None = None) -> None:
    ndim = 2 if channels is None else 3
    if array.ndim != ndim or (channels is not None and array.shape[2] != channels):
        raise ValueError(f"{name} 的形状应为 H×W" + (f"×{channels}。" if channels else "。"))
    if array.dtype != np.float32 or array.size == 0:
        raise ValueError(f"{name} 必须是非空 float32 数组。")
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} 必须位于 [0,1] 且不含 NaN/Inf。")


def _mask(mask: Mask, shape: tuple[int, ...]) -> None:
    if mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape != shape[:2]:
        raise ValueError("mask 必须是与图片高宽相同的 bool 数组。")
    if not mask.any():
        raise ValueError("前景遮罩为空；请重新框选人物或添加前景笔划。")


def read_image(path: str | Path) -> FloatImage:
    """读取并按 EXIF 校正方向；灰度转 RGB，已有透明区域铺白底。"""
    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / np.float32(255)


def segment_person(image: FloatImage, selection: Selection) -> Mask:
    """由矩形或四类标签初始化 GrabCut，返回前景归属而非透明度。"""
    _unit_array(image, "image", 3)
    _integer(selection.iterations, "iterations", 1)
    if (selection.rect is None) == (selection.labels is None):
        raise ValueError("rect 与 labels 必须且只能提供一种。")
    height, width = image.shape[:2]
    if selection.rect is not None:
        x, y, w, h = selection.rect
        for value in selection.rect:
            _integer(value, "矩形坐标/边长", 0)
        if w == 0 or h == 0 or x + w > width or y + h > height:
            raise ValueError("矩形必须在图片内且宽高大于 0。")
        labels = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
        labels[y:y + h, x:x + w] = cv2.GC_PR_FGD
    else:
        assert selection.labels is not None
        supplied = selection.labels
        if supplied.dtype != np.uint8 or supplied.shape != (height, width):
            raise ValueError("labels 必须是与图片高宽一致的 uint8 数组。")
        if np.any(supplied > 3):
            raise ValueError("GrabCut 标签只能为 0、1、2、3。")
        labels = supplied.copy()

    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    if np.count_nonzero(foreground) < 5 or np.count_nonzero(~foreground) < 5:
        raise ValueError("GrabCut 初始化需要至少 5 个前景候选和 5 个背景候选像素。")
    pixels = np.rint(image * 255).astype(np.uint8)
    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    # 固定初始化随机种子便于同一环境复现实验；OpenCV 的 RNG 是线程局部状态。
    cv2.setRNGSeed(selection.seed)
    cv2.grabCut(
        bgr, labels, (0, 0, 0, 0), bg_model, fg_model,
        selection.iterations, cv2.GC_INIT_WITH_MASK,
    )
    result = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    _mask(result, image.shape)
    return result


def smooth_foreground(image: FloatImage, mask: Mask, sigma: float = 1.5) -> FloatImage:
    """B = G*(M I) / (G*M)；背景输出为 0，其数值不表示前景颜色。"""
    _unit_array(image, "image", 3)
    return np.clip(masked_gaussian(image, mask, sigma), 0, 1)


def _float_array(array: FloatImage, name: str) -> None:
    """空间信号可有符号；与颜色的 [0,1] 验证分开。"""
    if array.ndim not in (2, 3) or array.dtype != np.float32 or array.size == 0:
        raise ValueError(f"{name} 必须是非空 float32 H×W 或 H×W×C 数组。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 不能含 NaN/Inf。")


def masked_gaussian(image: FloatImage, mask: Mask, sigma: float) -> FloatImage:
    """支持有符号单/多通道信号；先分别横纵卷积，再按前景权重归一化。"""
    _float_array(image, "image")
    _mask(mask, image.shape)
    _number(sigma, "sigma", 0)
    if sigma == 0:
        return np.where(mask[..., None] if image.ndim == 3 else mask, image, np.float32(0))

    radius = max(1, int(np.ceil(3 * sigma)))
    kernel = cv2.getGaussianKernel(2 * radius + 1, sigma, cv2.CV_32F)
    weight = mask.astype(np.float32)
    # sepFilter2D 显式体现二维高斯的空间可分离性；RGB 各通道独立计算。
    numerator = cv2.sepFilter2D(
        image * (weight[..., None] if image.ndim == 3 else weight),
        -1, kernel, kernel, borderType=cv2.BORDER_REFLECT_101,
    )
    if image.ndim == 3 and numerator.ndim == 2:
        numerator = numerator[..., None]
    denominator = cv2.sepFilter2D(
        weight, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT_101,
    )
    valid = mask & (denominator > np.finfo(np.float32).eps)
    result = np.zeros_like(image)
    divisor = denominator[valid, None] if image.ndim == 3 else denominator[valid]
    result[valid] = numerator[valid] / divisor
    result[mask & ~valid] = image[mask & ~valid]
    return result


def quantize_colors(
    base: FloatImage, levels: int = 6, amount: float = 0.65,
) -> FloatImage:
    """在 Lab 中量化 L*/100；amount 混合原明度和量化明度，保留 a*、b*。"""
    _unit_array(base, "base", 3)
    _integer(levels, "levels", 2)
    _number(amount, "amount", 0, 1)
    if amount == 0:
        return base.copy()
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2Lab)
    lightness = np.clip(lab[..., 0] / 100, 0, 1)
    quantized = np.rint(lightness * (levels - 1)) / (levels - 1)
    lab[..., 0] = 100 * ((1 - amount) * lightness + amount * quantized)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(rgb, 0, 1).astype(np.float32)


def _extend_foreground(gray: FloatImage, mask: Mask) -> FloatImage:
    """以附近前景值延拓背景，防止 Sobel 读取遮罩外的占位零值。

    OpenCV 的距离标签给出近邻前景像素索引（5×5 距离近似）。
    延拓只为局部滤波提供边界值，不增加人物区域，不修复混合发丝。
    """
    if mask.all():
        return gray.copy()
    _, labels = cv2.distanceTransformWithLabels(
        (~mask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
    )
    labels = labels.astype(np.int32, copy=False)
    lookup = np.zeros((int(labels.max()) + 1, *gray.shape[2:]), dtype=np.float32)
    lookup[labels[mask]] = gray[mask]
    return lookup[labels]


def extract_lines(
    base: FloatImage, mask: Mask, threshold: float = 0.035, softness: float = 0.08,
) -> FloatImage:
    """E=clip((sqrt(gx²+gy²)-threshold)/softness,0,1)；1 为最深描线。

    灰度权重相当于固定 1×1 通道组合；3×3 Sobel 除以 8 固定尺度。
    先延拓前景再计算方向响应，最后保留人物内部的线条。
    """
    _unit_array(base, "base", 3)
    _mask(mask, base.shape)
    _number(threshold, "threshold", 0)
    _number(softness, "softness", 0)
    if softness == 0:
        raise ValueError("softness 必须大于 0。")
    gray = base @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    extended = _extend_foreground(gray, mask)
    gx = cv2.Sobel(
        extended, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8,
        borderType=cv2.BORDER_REFLECT_101,
    )
    gy = cv2.Sobel(
        extended, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8,
        borderType=cv2.BORDER_REFLECT_101,
    )
    strength = np.clip((np.hypot(gx, gy) - threshold) / softness, 0, 1)
    return np.where(mask, strength, np.float32(0)).astype(np.float32)


def compose_rgba(
    colors: FloatImage, lines: FloatImage, alpha: FloatImage, strength: float = 0.8,
) -> FloatImage:
    """RGB=C*(1-strength*E)，A=alpha；输出非预乘 RGBA。"""
    _unit_array(colors, "colors", 3)
    _unit_array(lines, "lines")
    _unit_array(alpha, "alpha")
    _number(strength, "strength", 0, 1)
    if lines.shape != colors.shape[:2] or alpha.shape != colors.shape[:2]:
        raise ValueError("colors、lines、alpha 的高宽必须一致。")
    rgb = colors * (1 - strength * lines[..., None])
    return np.concatenate((rgb, alpha[..., None]), axis=2).astype(np.float32)


def save_png(rgba: FloatImage, path: str | Path) -> Path:
    """将非预乘 RGBA 保存为 8 位 PNG，保留透明通道。"""
    _unit_array(rgba, "rgba", 4)
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError("透明输出必须使用 .png 扩展名。")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgba * 255).astype(np.uint8)).save(target)
    return target
