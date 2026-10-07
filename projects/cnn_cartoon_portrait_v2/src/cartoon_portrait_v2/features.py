"""Compact feature extraction: one guide, RGB Sobel, structure tensor and DoG."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import as_mask, masked_gaussian, rgb_to_lab, sobel_rgb


@dataclass(frozen=True)
class FeatureConfig:
    sigma: float = 0.8
    dark_sigma: float = 1.6


@dataclass
class FeatureBundle:
    image: np.ndarray
    mask: np.ndarray
    guide_rgb: np.ndarray
    lab: np.ndarray
    lightness: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    jxx: np.ndarray
    jxy: np.ndarray
    jyy: np.ndarray
    edge: np.ndarray
    theta: np.ndarray
    coherence: np.ndarray
    dark: np.ndarray
    right_barrier: np.ndarray
    down_barrier: np.ndarray


def extract_features(image: np.ndarray, mask: np.ndarray, config: FeatureConfig | None = None) -> FeatureBundle:
    cfg = config or FeatureConfig()
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("image must have shape H,W,3")
    valid = as_mask(mask, rgb.shape[:2])
    guide = masked_gaussian(rgb, cfg.sigma, valid)
    lab = rgb_to_lab(guide)
    lightness = lab[..., 0] / 100.0
    gx, gy = sobel_rgb(guide)

    # A fixed 1x1 channel fusion is a pointwise depthwise-to-pointwise analogue.
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    jxx = np.sum(gx * gx * weights, axis=-1)
    jxy = np.sum(gx * gy * weights, axis=-1)
    jyy = np.sum(gy * gy * weights, axis=-1)
    tr = jxx + jyy
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy * jxy, 0))
    lam_max = 0.5 * (tr + disc)
    lam_min = 0.5 * (tr - disc)
    edge = np.sqrt(np.maximum(lam_max, 0))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy + 1e-12)
    coherence = np.divide(lam_max - lam_min, tr + 1e-6)
    coherence = np.clip(coherence, 0, 1).astype(np.float32)

    narrow = masked_gaussian(lightness, cfg.sigma, valid)
    broad = masked_gaussian(lightness, cfg.dark_sigma, valid)
    dark = np.maximum(broad - narrow, 0).astype(np.float32)

    # Lab color barriers are used only to stop regions crossing strong boundaries.
    dl = np.diff(lab, axis=1, append=lab[:, -1:, :])
    dd = np.diff(lab, axis=0, append=lab[-1:, :, :])
    right_barrier = np.sqrt(np.sum((dl / np.array([100, 128, 128], np.float32)) ** 2, axis=-1))
    down_barrier = np.sqrt(np.sum((dd / np.array([100, 128, 128], np.float32)) ** 2, axis=-1))
    for arr in (edge, theta, dark, right_barrier, down_barrier):
        arr[~valid] = 0
    return FeatureBundle(rgb, valid, guide, lab, lightness, gx, gy, jxx, jxy, jyy,
                         edge, theta, coherence, dark, right_barrier, down_barrier)
