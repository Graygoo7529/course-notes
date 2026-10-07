"""Tone branch: regional base lightness plus restrained shadow accents."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import lab_to_rgb, masked_gaussian


@dataclass(frozen=True)
class ToneConfig:
    shadow_fraction: float = .28
    shadow_depth: float = .18
    min_range: float = .08


@dataclass
class TonePlan:
    base: np.ndarray
    detail: np.ndarray
    shadow_candidate: np.ndarray
    shadow: np.ndarray
    flat: np.ndarray
    colors: np.ndarray


def generate_tone(features, regions, mask: np.ndarray, config: ToneConfig | None = None) -> TonePlan:
    cfg = config or ToneConfig()
    labels = regions.labels
    base = np.zeros_like(features.lightness, dtype=np.float32)
    for i in range(len(regions.palette_lab)):
        vals = features.lightness[(labels == i) & mask]
        if len(vals):
            base[labels == i] = float(np.median(vals))
    # One mild spatially separable convolution removes isolated tonal speckles.
    base = masked_gaussian(base, .55, mask)
    detail = (features.lightness - base) * mask
    candidate = np.zeros_like(base, dtype=bool)
    for i in range(len(regions.palette_lab)):
        where = (labels == i) & mask
        vals = features.lightness[where]
        if len(vals) and float(np.ptp(vals)) >= cfg.min_range:
            threshold = float(np.quantile(vals, cfg.shadow_fraction))
            candidate |= where & (features.lightness < threshold)
    candidate &= mask
    shadow = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    colors = regions.flat.copy()
    # Compress only the local L component; chroma stays from the palette.
    lab = cv2.cvtColor(np.clip(colors, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    lab[..., 0] = np.clip(lab[..., 0] - shadow.astype(np.float32) * (cfg.shadow_depth * 100), 0, 100)
    flat = lab_to_rgb(lab)
    flat[~mask] = 0
    return TonePlan(base, detail, candidate, shadow, regions.flat, flat)
