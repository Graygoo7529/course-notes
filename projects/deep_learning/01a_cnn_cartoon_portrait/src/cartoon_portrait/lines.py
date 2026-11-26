"""方向细化、连通筛选与着墨；仅使用共享结构，不读取色块结果。"""

from dataclasses import dataclass

import cv2
import numpy as np

from .features import Structure
from .pipeline import FloatImage, Mask, _float_array, _integer, _mask, _number


@dataclass(frozen=True)
class LineConfig:
    low: float = 0.025
    high: float = 0.06
    width: float = 0.12
    gamma: float = 0.9
    thin: bool = True
    min_coherence: float = 0.1
    outline_width: int = 0
    outline_strength: float = 0.5

    def validate(self) -> None:
        for name, value in (("low", self.low), ("high", self.high), ("width", self.width)):
            _number(value, name, float(np.finfo(np.float32).eps))
        if self.low >= self.high:
            raise ValueError("线条阈值必须满足 0 < low < high。")
        _number(self.gamma, "gamma", float(np.finfo(np.float32).eps), 1)
        _number(self.min_coherence, "min_coherence", 0, 1)
        _integer(self.outline_width, "outline_width", 0)
        _number(self.outline_strength, "outline_strength", 0, 1)


@dataclass(frozen=True)
class LineDebug:
    thinned: FloatImage
    retained: Mask
    ink_internal: FloatImage
    outline: FloatImage


def thin_edges(
    strength: FloatImage, theta: FloatImage, coherence: FloatImage, mask: Mask,
    min_coherence: float = 0.1,
) -> FloatImage:
    """可靠法向上非极大值抑制；模糊方向处保留原响应，交给后续筛选。"""
    for name, value in (("strength", strength), ("theta", theta), ("coherence", coherence)):
        _float_array(value, name)
        if value.shape != mask.shape:
            raise ValueError("方向、幅值、一致性与遮罩必须是相同大小的二维数组。")
    _mask(mask, strength.shape)
    _number(min_coherence, "min_coherence", 0, 1)
    y, x = np.indices(mask.shape, dtype=np.float32)
    dx, dy = np.cos(theta), np.sin(theta)
    field = np.where(mask, strength, np.float32(0))
    # 图像外和遮罩外均视为无效零响应，插值不参与生成前景。
    plus = cv2.remap(field, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    minus = cv2.remap(field, x - dx, y - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    reliable = (coherence >= min_coherence) & (coherence > 1e-4) & (strength > 1e-6)
    # 一侧严格比较，避免理想阶跃的等高双像素带两侧同时保留。
    maxima = (strength >= plus) & (strength > minus)
    return np.where(mask & (~reliable | maxima), strength, np.float32(0))


def hysteresis(
    strength: FloatImage, mask: Mask, low: float, high: float,
) -> Mask:
    """低阈值八连通区域必须包含高阈值响应；不按面积删除短线。"""
    _float_array(strength, "strength")
    _mask(mask, strength.shape)
    _number(low, "low", float(np.finfo(np.float32).eps))
    _number(high, "high", float(np.finfo(np.float32).eps))
    if strength.ndim != 2 or low >= high:
        raise ValueError("需要二维幅值与 0 < low < high。")
    weak = mask & (strength >= low)
    strong = mask & (strength >= high)
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    labels = labels.astype(np.int32, copy=False)
    accepted = np.zeros(count, dtype=bool)
    accepted[labels[strong]] = True
    accepted[0] = False
    return accepted[labels]


def map_ink(
    strength: FloatImage, retained: Mask, low: float, width: float, gamma: float,
) -> FloatImage:
    """筛选和显示深浅分开：E=R*clip((g-low)/width,0,1)^gamma。"""
    _float_array(strength, "strength")
    if retained.dtype != np.bool_ or retained.shape != strength.shape or strength.ndim != 2:
        raise ValueError("retained 必须是与二维幅值对应的 bool 数组。")
    _number(low, "low", 0)
    _number(width, "width", float(np.finfo(np.float32).eps))
    _number(gamma, "gamma", float(np.finfo(np.float32).eps), 1)
    ink = np.power(np.clip((strength - low) / width, 0, 1), gamma)
    return np.where(retained, ink, np.float32(0)).astype(np.float32)


def outline_mask(mask: Mask, width: int) -> FloatImage:
    """可选内轮廓；图像外采用中性前景，避免人为描出画面裁切边。"""
    _mask(mask, mask.shape)
    _integer(width, "outline_width", 0)
    if width == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    eroded = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
        iterations=width, borderType=cv2.BORDER_CONSTANT, borderValue=1,
    )
    return (mask & (eroded == 0)).astype(np.float32)


def make_lines(
    structure: Structure, mask: Mask, config: LineConfig = LineConfig(),
) -> tuple[FloatImage, LineDebug]:
    config.validate()
    _mask(mask, structure.strength.shape)
    thinned = (
        thin_edges(
            structure.strength, structure.theta, structure.coherence, mask,
            config.min_coherence,
        ) if config.thin else np.where(mask, structure.strength, np.float32(0))
    )
    retained = hysteresis(thinned, mask, config.low, config.high)
    internal = map_ink(thinned, retained, config.low, config.width, config.gamma)
    outline = outline_mask(mask, config.outline_width)
    ink = np.maximum(internal, config.outline_strength * outline)
    return ink, LineDebug(thinned, retained, internal, outline)

