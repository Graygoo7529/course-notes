"""Anti-aliased, high-resolution composition; no learned upsampling is needed here."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderConfig:
    scale: int = 4
    line_width: float = 1.35
    line_color: tuple[int, int, int] = (28, 24, 30)
    outline_color: tuple[int, int, int] = (34, 24, 30)
    dark_color: tuple[int, int, int] = (24, 18, 24)


@dataclass
class RenderResult:
    rgb: np.ndarray
    rgba: np.ndarray
    regions: np.ndarray
    strokes: np.ndarray


def compose_image(lines, regions, tone, mask: np.ndarray, config: RenderConfig | None = None) -> RenderResult:
    cfg = config or RenderConfig()
    h, w = mask.shape
    scale = max(1, int(cfg.scale))
    size = (w * scale, h * scale)
    region_img = cv2.resize(tone.flat.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
    stroke_layer = np.zeros((h * scale, w * scale, 3), np.uint8)
    stroke_alpha = np.zeros((h * scale, w * scale), np.float32)
    colors = {"outline": cfg.outline_color, "dark": cfg.dark_color, "edge": cfg.line_color}
    for stroke in lines.strokes:
        points = np.rint(stroke.points * scale).astype(np.int32)
        if len(points) < 2:
            continue
        color = colors.get(stroke.source, cfg.line_color)
        steps = len(points) if stroke.closed else len(points) - 1
        for i in range(steps):
            j = (i + 1) % len(points)
            progress = (i + .5) / max(steps, 1)
            taper_pixels = max(1.0, float(stroke.taper) * scale)
            taper_start = min(1.0, (i + 1) / taper_pixels) if stroke.taper_start else 1.0
            taper_end = min(1.0, (steps - i) / taper_pixels) if stroke.taper_end else 1.0
            width = max(.55, stroke.width * min(taper_start, taper_end) * scale)
            cv2.line(stroke_layer, tuple(points[i]), tuple(points[j]), color,
                     max(1, int(round(width))), cv2.LINE_AA)
            alpha = np.zeros_like(stroke_alpha, dtype=np.uint8)
            cv2.line(alpha, tuple(points[i]), tuple(points[j]),
                     int(np.clip(stroke.opacity, 0, 1) * 255), max(1, int(round(width))), cv2.LINE_AA)
            stroke_alpha = np.maximum(stroke_alpha, alpha.astype(np.float32) / 255.0)
    strokes = stroke_layer.astype(np.float32) / 255.0
    rgb = region_img.copy()
    ink_color = strokes
    rgb = rgb * (1 - stroke_alpha[..., None]) + ink_color * stroke_alpha[..., None]
    alpha = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)
    return RenderResult(np.clip(rgb, 0, 1), rgba, region_img, strokes)
