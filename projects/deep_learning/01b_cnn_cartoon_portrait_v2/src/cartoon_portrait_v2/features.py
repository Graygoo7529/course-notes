"""FeatureBank Lite: a small multi-scale bank with semantic outputs.

The bank keeps the useful part of v1: scale-aware structure, smoothed
orientation, signed DoG and chroma edges. It deliberately does not expose
every diagnostic tensor to the later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .pipeline import as_mask, masked_gaussian, rgb_to_lab, sobel_rgb


@dataclass(frozen=True)
class FeatureConfig:
    scales: tuple[float, ...] = (0.6, 1.2, 2.4)
    operator: Literal["sobel", "scharr"] = "sobel"
    channel_weights: tuple[float, float, float] = (0.299, 0.587, 0.114)
    direction_sigma: float = 1.0
    dog_ratio: float = 2.0

    def validate(self) -> None:
        if not 1 <= len(self.scales) <= 4:
            raise ValueError("scales 需要 1–4 个递增尺度。")
        if any(float(s) <= 0 or (i and float(s) <= float(self.scales[i - 1]))
               for i, s in enumerate(self.scales)):
            raise ValueError("scales 必须为正数且严格递增。")
        if self.operator not in ("sobel", "scharr"):
            raise ValueError("operator 必须为 sobel 或 scharr。")
        weights = np.asarray(self.channel_weights, dtype=np.float32)
        if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("channel_weights 必须是和为 1 的三个非负数。")
        if self.direction_sigma < 0 or self.dog_ratio <= 1:
            raise ValueError("direction_sigma 不能为负，dog_ratio 必须大于 1。")


@dataclass
class FeatureBundle:
    image: np.ndarray
    mask: np.ndarray
    scales: tuple[float, ...]
    guides: np.ndarray                 # K,H,W,3
    gx: np.ndarray                     # K,H,W,3
    gy: np.ndarray                     # K,H,W,3
    edges: np.ndarray                  # K,H,W; sigma * sqrt(lambda_max)
    dog: np.ndarray                    # K,H,W; signed broad - narrow
    thetas: np.ndarray                 # K,H,W; smoothed normal direction
    coherences: np.ndarray             # K,H,W
    direction_valid: np.ndarray        # K,H,W
    lab: np.ndarray
    lightness: np.ndarray
    chroma_edge: np.ndarray
    jxx: np.ndarray
    jxy: np.ndarray
    jyy: np.ndarray
    edge: np.ndarray                   # fused edge used by the line branch
    theta: np.ndarray                  # selected normal direction
    coherence: np.ndarray              # selected direction reliability
    dark: np.ndarray                   # positive dark candidate used by lines
    right_barrier: np.ndarray
    down_barrier: np.ndarray

    @property
    def guide_rgb(self) -> np.ndarray:
        return self.guides[0]


def _gradients(guide: np.ndarray, operator: str) -> tuple[np.ndarray, np.ndarray]:
    if operator == "sobel":
        return sobel_rgb(guide)
    gx = np.stack([cv2.Sobel(guide[..., c], cv2.CV_32F, 1, 0, ksize=-1) / 32.0
                   for c in range(3)], axis=-1)
    gy = np.stack([cv2.Sobel(guide[..., c], cv2.CV_32F, 0, 1, ksize=-1) / 32.0
                   for c in range(3)], axis=-1)
    return gx, gy


def _structure(gx: np.ndarray, gy: np.ndarray, mask: np.ndarray,
               weights: np.ndarray, smooth_sigma: float) -> tuple[np.ndarray, ...]:
    # The edge amplitude remains local (as in v1); only the tensor used for
    # orientation is smoothed. This preserves fine glasses/eyelid responses.
    raw_jxx = np.sum(gx * gx * weights, axis=-1)
    raw_jxy = np.sum(gx * gy * weights, axis=-1)
    raw_jyy = np.sum(gy * gy * weights, axis=-1)
    raw_trace = raw_jxx + raw_jyy
    raw_gap = np.hypot(raw_jxx - raw_jyy, 2 * raw_jxy)
    raw_largest = np.maximum((raw_trace + raw_gap) * .5, 0)
    tensor = masked_gaussian(np.stack((raw_jxx, raw_jxy, raw_jyy), axis=-1), smooth_sigma, mask)
    jxx, jxy, jyy = np.moveaxis(tensor, -1, 0)
    trace = jxx + jyy
    gap = np.hypot(jxx - jyy, 2 * jxy)
    smallest = np.maximum((trace - gap) * .5, 0)
    coherence = np.divide(gap, trace, out=np.zeros_like(gap), where=trace > 1e-12)
    valid = mask & (trace > 1e-12) & (coherence > 1e-4)
    theta = np.where(valid, .5 * np.arctan2(2 * jxy, jxx - jyy), 0).astype(np.float32)
    return (jxx.astype(np.float32), jxy.astype(np.float32), jyy.astype(np.float32),
            coherence.clip(0, 1).astype(np.float32), valid,
            theta, np.sqrt(raw_largest).astype(np.float32))


def _barriers(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.array([100, 128, 128], np.float32)
    normalized = lab / scale
    right = np.diff(normalized, axis=1, append=normalized[:, -1:, :])
    down = np.diff(normalized, axis=0, append=normalized[-1:, :, :])
    return np.sqrt(np.sum(right * right, axis=-1)), np.sqrt(np.sum(down * down, axis=-1))


def extract_features(image: np.ndarray, mask: np.ndarray,
                     config: FeatureConfig | None = None) -> FeatureBundle:
    cfg = config or FeatureConfig()
    cfg.validate()
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("image must have shape H,W,3")
    valid = as_mask(mask, rgb.shape[:2])
    weights = np.asarray(cfg.channel_weights, dtype=np.float32)
    raw_lab = rgb_to_lab(rgb)
    raw_lightness = raw_lab[..., 0] / 100.0
    guides, gxs, gys, edges, dogs, thetas, coherences, valids = [], [], [], [], [], [], [], []
    jxxs, jxys, jyys = [], [], []
    for sigma in cfg.scales:
        guide = masked_gaussian(rgb, float(sigma), valid)
        gx, gy = _gradients(guide, cfg.operator)
        jxx, jxy, jyy, coherence, direction_valid, theta, strength = _structure(
            gx, gy, valid, weights, cfg.direction_sigma)
        # DoG uses the original lightness, avoiding a second hidden blur of the
        # already smoothed guide. This matches v1's signed multi-scale response.
        narrow = masked_gaussian(raw_lightness, float(sigma), valid)
        broad = masked_gaussian(raw_lightness, float(sigma) * cfg.dog_ratio, valid)
        guides.append(guide)
        gxs.append(gx)
        gys.append(gy)
        # Scale normalization makes fine and coarse responses comparable.
        edges.append((float(sigma) * strength).astype(np.float32))
        dogs.append((broad - narrow).astype(np.float32))
        thetas.append(theta)
        coherences.append(coherence)
        valids.append(direction_valid)
        jxxs.append(jxx)
        jxys.append(jxy)
        jyys.append(jyy)
    guides_a, gx_a, gy_a = np.stack(guides), np.stack(gxs), np.stack(gys)
    edges_a, dog_a = np.stack(edges), np.stack(dogs)
    theta_a, coherence_a, valid_a = np.stack(thetas), np.stack(coherences), np.stack(valids)
    lab = rgb_to_lab(guides_a[0])
    lightness = lab[..., 0] / 100.0
    chroma = lab[..., 1:].astype(np.float32) / 128.0
    chroma_grad = np.dstack((chroma, np.zeros((*chroma.shape[:2], 1), np.float32)))
    cx, cy = _gradients(chroma_grad, "sobel")
    chroma_edge = np.sqrt(np.sum(cx[..., :2] ** 2 + cy[..., :2] ** 2, axis=-1))
    right_barrier, down_barrier = _barriers(lab)
    # Select orientation and coherence belonging to the strongest scale.
    best = np.argmax(edges_a, axis=0)
    selected_theta = np.take_along_axis(theta_a, best[None, ...], axis=0)[0]
    selected_coherence = np.take_along_axis(coherence_a, best[None, ...], axis=0)[0]
    edge = np.max(edges_a, axis=0)
    dark = np.max(np.maximum(dog_a, 0), axis=0)
    for arr in (edge, selected_theta, selected_coherence, dark, chroma_edge,
                right_barrier, down_barrier):
        arr[~valid] = 0
    return FeatureBundle(
        rgb, valid, tuple(float(s) for s in cfg.scales), guides_a, gx_a, gy_a,
        edges_a, dog_a, theta_a, coherence_a, valid_a, lab, lightness, chroma_edge,
        np.stack(jxxs), np.stack(jxys), np.stack(jyys), edge, selected_theta,
        selected_coherence, dark, right_barrier, down_barrier,
    )
