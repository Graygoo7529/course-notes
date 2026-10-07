"""Region branch: deterministic Lab prototypes plus small-component merging."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import lab_to_rgb, rgb_to_lab


@dataclass(frozen=True)
class RegionConfig:
    palette_size: int = 5
    min_area: int = 18
    compactness: float = 0.12
    palette_style: str = "natural"


@dataclass
class RegionPlan:
    initial_labels: np.ndarray
    labels: np.ndarray
    palette_lab: np.ndarray
    palette_rgb: np.ndarray
    flat: np.ndarray
    barrier: np.ndarray


def _kmeans(lab: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    pixels = lab[mask].reshape(-1, 3).astype(np.float32)
    if len(pixels) == 0:
        return np.zeros(mask.shape, np.int32), np.zeros((1, 3), np.float32)
    k = max(1, min(int(k), len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, .1)
    cv2.setRNGSeed(20261007)
    best = np.empty((len(pixels), 1), np.int32)
    _, labels, centers = cv2.kmeans(pixels, k, best, criteria, 2, cv2.KMEANS_PP_CENTERS)
    out = np.full(mask.shape, -1, np.int32)
    out[mask] = labels.ravel()
    return out, centers.astype(np.float32)


def _merge_small(labels: np.ndarray, lab: np.ndarray, barriers: np.ndarray, mask: np.ndarray,
                 centers: np.ndarray, min_area: int) -> np.ndarray:
    out = labels.copy()
    # Components are measured inside each prototype; otherwise every valid pixel
    # would be one component and tiny colour islands could never be merged.
    for label in range(len(centers)):
        local = ((labels == label) & mask).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=4)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                continue
            ys, xs = np.where(cc == i)
            if len(ys) == 0:
                continue
            candidates: list[tuple[float, int]] = []
            step = max(1, len(ys) // 80)
            for y, x in zip(ys[::step], xs[::step]):
                for dy, dx, b in ((0, -1, barriers[y, max(x-1, 0)]), (0, 1, barriers[y, x]),
                                  (-1, 0, barriers[max(y-1, 0), x]), (1, 0, barriers[y, x])):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]:
                        target = out[ny, nx]
                        if target >= 0 and target != label:
                            d = float(np.linalg.norm(lab[y, x] - centers[target])) + 5.0 * float(b)
                            candidates.append((d, int(target)))
            if candidates:
                target = min(candidates)[1]
                out[cc == i] = target
    out[~mask] = -1
    return out


def generate_regions(features, mask: np.ndarray, config: RegionConfig | None = None) -> RegionPlan:
    cfg = config or RegionConfig()
    barrier = np.maximum(features.right_barrier, features.down_barrier).astype(np.float32)
    initial, centers = _kmeans(features.lab, mask, cfg.palette_size)
    labels = _merge_small(initial, features.lab, barrier, mask, centers, cfg.min_area)
    # Mean color per final connected label makes the color block visibly flat.
    palette = centers.copy()
    for i in range(len(palette)):
        vals = features.lab[(labels == i) & mask]
        if len(vals):
            palette[i] = np.median(vals, axis=0)
    palette_rgb = lab_to_rgb(palette)
    hsv = cv2.cvtColor(palette_rgb[None, ...].astype(np.float32), cv2.COLOR_RGB2HSV)[0]
    style = cfg.palette_style.lower()
    if style == "warm":
        hsv[..., 0] = (hsv[..., 0] + 3.0) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.08, 0, 1)
    elif style == "pastel":
        hsv[..., 1] = np.clip(hsv[..., 1] * .62, 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.06, 0, 1)
    elif style == "noir":
        hsv[..., 1] = np.clip(hsv[..., 1] * .25, 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2] * .92, 0, 1)
    palette_rgb = cv2.cvtColor(hsv[None, ...], cv2.COLOR_HSV2RGB)[0]
    palette = rgb_to_lab(palette_rgb)
    flat_lab = np.zeros_like(features.lab)
    for i, color in enumerate(palette):
        flat_lab[labels == i] = color
    flat = lab_to_rgb(flat_lab)
    flat[~mask] = 0
    return RegionPlan(initial, labels, palette, palette_rgb, flat, barrier)
