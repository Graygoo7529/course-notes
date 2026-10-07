"""先逐通道读取空间邻域，再融合非线性结构特征；不训练卷积核。"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .pipeline import (
    FloatImage, Mask, _extend_foreground, _float_array, _mask, _number,
    _unit_array, masked_gaussian,
)


@dataclass(frozen=True)
class FeatureConfig:
    sigma: float = 0.6
    operator: Literal["sobel", "scharr"] = "sobel"

    def validate(self) -> None:
        _number(self.sigma, "sigma", 0)
        if self.operator not in ("sobel", "scharr"):
            raise ValueError("operator 必须为 sobel 或 scharr。")


@dataclass(frozen=True)
class FusionConfig:
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

    def validate(self) -> None:
        weights = np.asarray(self.weights)
        if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("通道权重必须是三个非负有限数。")
        if not np.isclose(weights.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError("通道权重之和必须为 1。")


@dataclass(frozen=True)
class ChannelFeatures:
    guide: FloatImage
    gx: FloatImage
    gy: FloatImage
    channel_strength: FloatImage


@dataclass(frozen=True)
class Structure:
    guide: FloatImage
    jxx: FloatImage
    jxy: FloatImage
    jyy: FloatImage
    strength: FloatImage
    theta: FloatImage
    coherence: FloatImage
    direction_valid: Mask


def prepare_channels(image: FloatImage, mask: Mask, sigma: float = 0.6) -> FloatImage:
    """轻度逐通道高斯，不混合 RGB；背景为无效占位值。"""
    _unit_array(image, "image", 3)
    return np.clip(masked_gaussian(image, mask, sigma), 0, 1)


def channel_gradients(
    guide: FloatImage, mask: Mask, operator: str = "sobel",
) -> tuple[FloatImage, FloatImage]:
    """DW 的连接方式：每通道两方向，单位线性坡度响应归一为 1。"""
    _float_array(guide, "guide")
    _mask(mask, guide.shape)
    if operator not in ("sobel", "scharr"):
        raise ValueError("operator 必须为 sobel 或 scharr。")
    extended = _extend_foreground(guide, mask)
    ksize, scale = (3, 1 / 8) if operator == "sobel" else (-1, 1 / 32)
    responses = []
    for dx, dy in ((1, 0), (0, 1)):
        response = cv2.Sobel(
            extended, cv2.CV_32F, dx, dy, ksize=ksize, scale=scale,
            borderType=cv2.BORDER_REFLECT_101,
        )
        if guide.ndim == 3 and response.ndim == 2:
            response = response[..., None]
        response[~mask] = 0
        responses.append(response.astype(np.float32, copy=False))
    return responses[0], responses[1]


def extract_channel_features(
    image: FloatImage, mask: Mask, config: FeatureConfig = FeatureConfig(),
) -> ChannelFeatures:
    """RGB → 轻度去噪 → 六张方向特征图和三张幅值图。"""
    config.validate()
    guide = prepare_channels(image, mask, config.sigma)
    gx, gy = channel_gradients(guide, mask, config.operator)
    return ChannelFeatures(guide, gx, gy, np.hypot(gx, gy))


def fuse_features(
    features: ChannelFeatures, config: FusionConfig = FusionConfig(),
) -> Structure:
    """平方/乘积特征经固定 PW 融合，再求颜色结构矩阵的主方向。"""
    config.validate()
    _unit_array(features.guide, "guide", 3)
    for name, array in (("gx", features.gx), ("gy", features.gy)):
        _float_array(array, name)
        if array.shape != features.guide.shape:
            raise ValueError("方向特征必须与 guide 形状相同。")
    # 每通道先构造三种二阶特征；@ weights 是固定 1×1 通道融合。
    # float64 中间量减少近重根时的相减误差。
    gx, gy = features.gx.astype(np.float64), features.gy.astype(np.float64)
    weights = np.asarray(config.weights, dtype=np.float64)
    jxx = (gx * gx) @ weights
    jxy = (gx * gy) @ weights
    jyy = (gy * gy) @ weights
    trace = jxx + jyy
    gap = np.hypot(jxx - jyy, 2 * jxy)
    largest = np.maximum((trace + gap) * 0.5, 0)
    smallest = np.maximum((trace - gap) * 0.5, 0)
    coherence = np.divide(
        largest - smallest, trace, out=np.zeros_like(trace), where=trace > 1e-12,
    )
    valid = (trace > 1e-12) & (coherence > 1e-4)
    theta = np.where(valid, 0.5 * np.arctan2(2 * jxy, jxx - jyy), 0)
    return Structure(
        features.guide.copy(), jxx.astype(np.float32), jxy.astype(np.float32),
        jyy.astype(np.float32), np.sqrt(largest).astype(np.float32),
        theta.astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32), valid,
    )


def comparison_strengths(
    features: ChannelFeatures, mask: Mask, feature_config: FeatureConfig,
    fusion_config: FusionConfig,
) -> dict[str, FloatImage]:
    """保持共同响应尺度，输出灰度路线与幅值 PW 融合的对照。"""
    feature_config.validate()
    fusion_config.validate()
    gray = features.guide @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx, gy = channel_gradients(gray, mask, feature_config.operator)
    weights = np.asarray(fusion_config.weights, dtype=np.float32)
    return {
        "gray_strength": np.hypot(gx, gy),
        "magnitude_sum": features.channel_strength @ weights,
    }
