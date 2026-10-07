"""Small image operators shared by the five pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as float32 RGB in [0, 1]."""
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    if shape and raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return raw > 127


def as_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"mask shape {arr.shape} != image shape {shape}")
    return arr.astype(bool)


def normalize01(x: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a)
    lo = float(np.min(a[finite])) if low is None else float(low)
    hi = float(np.max(a[finite])) if high is None else float(high)
    if hi <= lo + 1e-8:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def robust01(x: np.ndarray, mask: np.ndarray | None = None, q=(0.05, 0.98)) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    values = a[mask] if mask is not None and np.any(mask) else a.ravel()
    lo, hi = np.quantile(values[np.isfinite(values)], q) if values.size else (0.0, 1.0)
    return normalize01(a, float(lo), float(hi))


def masked_gaussian(x: np.ndarray, sigma: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Spatially separable Gaussian, normalized near the valid image mask."""
    a = np.asarray(x, dtype=np.float32)
    if sigma <= 0:
        return a.copy()
    k = max(3, int(round(6 * sigma + 1)) | 1)
    if mask is None:
        return cv2.GaussianBlur(a, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    m = mask.astype(np.float32)
    if a.ndim == 3:
        numer = cv2.GaussianBlur(a * m[..., None], (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    else:
        numer = cv2.GaussianBlur(a * m, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    denom = cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    if a.ndim == 3:
        return numer / np.maximum(denom[..., None], 1e-6)
    return numer / np.maximum(denom, 1e-6)


def sobel_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel Sobel derivatives; this is the explicit CNN-like kernel stage."""
    gx = np.stack([cv2.Sobel(rgb[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8.0 for c in range(3)], axis=-1)
    gy = np.stack([cv2.Sobel(rgb[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8.0 for c in range(3)], axis=-1)
    return gx, gy


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    arr = np.clip(rgb, 0, 1).astype(np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return cv2.cvtColor(arr[None, ...], cv2.COLOR_RGB2LAB)[0]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    arr = np.asarray(lab, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return np.clip(cv2.cvtColor(arr[None, ...], cv2.COLOR_LAB2RGB)[0], 0, 1)
    return np.clip(cv2.cvtColor(arr, cv2.COLOR_LAB2RGB), 0, 1)


def save_gray(path: str | Path, image: np.ndarray, mask: np.ndarray | None = None) -> None:
    a = robust01(image, mask)
    cv2.imwrite(str(path), (a * 255).astype(np.uint8))


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(image, 0, 1) * 255.0, cv2.COLOR_RGB2BGR).astype(np.uint8))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
