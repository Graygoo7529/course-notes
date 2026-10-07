"""区域和笔画在目标画布上重绘；线性 RGB 合成，透明边缘预乘处理。"""

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import FloatImage, Mask, _integer, _number
from .regions import FillPlan, IntMap
from .strokes import StrokeSet


@dataclass(frozen=True)
class RenderConfig:
    scale: int = 4
    supersample: int = 1
    max_pixels: int = 24_000_000

    def validate(self) -> None:
        _integer(self.scale, "render_scale", 1)
        _integer(self.supersample, "supersample", 1)
        _integer(self.max_pixels, "max_render_pixels", 1)
        if self.scale > 8 or self.supersample > 4:
            raise ValueError("render_scale 最大为 8，supersample 最大为 4。")


@dataclass(frozen=True)
class RenderResult:
    rgba: FloatImage
    colors: FloatImage
    ink: FloatImage
    alpha: FloatImage


def srgb_to_linear(values: FloatImage) -> FloatImage:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(values: FloatImage) -> FloatImage:
    values = np.clip(values, 0, 1)
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055).astype(np.float32)


def rasterize_regions(labels: IntMap, factor: int, tolerance: float = 0.3) -> IntMap:
    """近邻标签为覆盖底图，细化网格上的边界折线进行几何重绘，孔洞按奇偶规则保留。"""
    _integer(factor, "factor", 1)
    _number(tolerance, "tolerance", 0, 0.75)
    h, w = labels.shape
    canvas = cv2.resize(labels.astype(np.float32), (w * factor, h * factor), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    if factor == 1:
        return canvas
    # 2 倍标签网格给出接近像素单元边界的路径；背景留给独立 alpha。
    for region in np.unique(labels):
        if region == 0:
            continue
        binary = cv2.resize((labels == region).astype(np.uint8), (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        paths = []
        for contour in contours:
            simple = cv2.approxPolyDP(contour, tolerance * 2, True)
            mapped = ((simple.astype(np.float32) + 0.5) * (factor / 2) - 0.5)
            paths.append(np.rint(mapped * 256).astype(np.int32))
        if paths:
            cv2.fillPoly(canvas, paths, int(region), lineType=cv2.LINE_8, shift=8)
    return canvas


def render_cartoon(strokes: StrokeSet, fills: FillPlan, mask: Mask,
                   config: RenderConfig = RenderConfig()) -> RenderResult:
    config.validate()
    h, w = mask.shape
    factor = config.scale * config.supersample
    if h * w * factor * factor > config.max_pixels:
        raise ValueError("绘制画布超出像素上限，请减小 --render-scale 或 --supersample。")
    shape = (w * factor, h * factor)
    labels = rasterize_regions(fills.labels, factor)
    # 标签零值在前景边缘插值时需要颜色延拓，最终归属只由 alpha 决定。
    if np.any(labels == 0):
        _, closest = cv2.distanceTransformWithLabels((labels == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros(int(closest.max()) + 1, dtype=np.int32)
        lookup[closest[labels > 0]] = labels[labels > 0]
        labels = np.where(labels > 0, labels, lookup[closest]).astype(np.int32)
    shadow = rasterize_regions(fills.shadow.astype(np.int32), factor) > 0
    shadow &= fills.thresholds[labels] >= 0
    colors = np.where(shadow[..., None], fills.shadow_palette[labels], fills.palette[labels]).astype(np.float32)
    if mask.all():
        alpha = np.ones((h * factor, w * factor), dtype=np.float32)
    else:
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        signed = np.where(mask, inside - 0.5, -(outside - 0.5)).astype(np.float32)
        distance = cv2.resize(signed, shape, interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(0.5 + distance * factor, 0, 1).astype(np.float32)
    ink = np.zeros((h * factor, w * factor), dtype=np.float32)
    # 每条路径局部画布，避免为每根线分配一张完整高分辨率图。
    for stroke in strokes.strokes:
        if not len(stroke.points) or stroke.opacity == 0:
            continue
        points = (stroke.points + 0.5) * factor - 0.5
        radius = int(np.ceil(stroke.width * factor / 2)) + 3
        lo = np.maximum(np.floor(points.min(axis=0)).astype(int) - radius, [0, 0])
        hi = np.minimum(np.ceil(points.max(axis=0)).astype(int) + radius + 1, [shape[0], shape[1]])
        tile = np.zeros((int(hi[1] - lo[1]), int(hi[0] - lo[0])), dtype=np.uint8)
        shifted = points - lo
        lengths = np.linalg.norm(np.diff(stroke.points, axis=0), axis=1)
        arclength = np.concatenate(([0.0], np.cumsum(lengths)))
        total = float(arclength[-1])
        if len(points) == 1:
            cv2.circle(tile, tuple(np.rint(shifted[0]).astype(int)), max(1, int(stroke.width * factor / 2)),
                       255, -1, lineType=cv2.LINE_AA)
        else:
            segments = len(points) if stroke.closed else len(points) - 1
            for index in range(segments):
                successor = (index + 1) % len(points)
                width = stroke.width
                if strokes.taper > 0 and not stroke.closed:
                    middle = float((arclength[index] + arclength[successor]) / 2)
                    fade = 1.0
                    if stroke.taper_start:
                        fade = min(fade, middle / strokes.taper)
                    if stroke.taper_end:
                        fade = min(fade, (total - middle) / strokes.taper)
                    width *= 0.35 + 0.65 * np.clip(fade, 0, 1)
                first = tuple(np.rint(shifted[index] * 256).astype(int))
                second = tuple(np.rint(shifted[successor] * 256).astype(int))
                cv2.line(tile, first, second, 255, max(1, int(round(width * factor))), cv2.LINE_AA, shift=8)
        view = ink[lo[1]:hi[1], lo[0]:hi[0]]
        np.maximum(view, tile.astype(np.float32) * np.float32(stroke.opacity / 255), out=view)
    ink_color = srgb_to_linear(np.array(strokes.ink, dtype=np.float32))
    linear_colors = srgb_to_linear(colors)
    combined = linear_colors * (1 - ink[..., None]) + ink_color * ink[..., None]
    if config.supersample > 1:
        final_shape = (w * config.scale, h * config.scale)
        def shrink(values: FloatImage) -> FloatImage:
            return cv2.resize(values, final_shape, interpolation=cv2.INTER_AREA).astype(np.float32)
        premult = shrink(combined * alpha[..., None])
        color_premult = shrink(linear_colors * alpha[..., None])
        ink = shrink(ink)
        alpha = shrink(alpha)
        combined = np.divide(premult, alpha[..., None], out=np.zeros_like(premult), where=alpha[..., None] > 1e-7)
        linear_colors = np.divide(color_premult, alpha[..., None], out=np.zeros_like(color_premult), where=alpha[..., None] > 1e-7)
    rgb = linear_to_srgb(combined)
    rgb[alpha == 0] = 0
    rgba = np.concatenate((rgb, alpha[..., None]), axis=2).astype(np.float32)
    return RenderResult(rgba, linear_to_srgb(linear_colors), ink, alpha)
