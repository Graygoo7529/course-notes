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
    width = max(1, int(round(cfg.line_width * scale)))
    for points in lines.strokes:
        pts = np.rint(points * scale).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(stroke_layer, [pts], False, cfg.line_color, width, cv2.LINE_AA)
    strokes = stroke_layer.astype(np.float32) / 255.0
    rgb = region_img.copy()
    # The stored stroke RGB is a colour, not an alpha channel.  Derive the
    # coverage from painted pixels so a dark ink colour remains visible.
    ink = (np.max(stroke_layer, axis=-1, keepdims=True) > 0).astype(np.float32)
    ink_color = np.asarray(cfg.line_color, np.float32)[None, None, :] / 255.0
    rgb = rgb * (1 - .90 * ink) + ink_color * (.90 * ink)
    alpha = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)
    return RenderResult(np.clip(rgb, 0, 1), rgba, region_img, strokes)
