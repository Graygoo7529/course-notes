"""Image validation and geometry helpers used by the offline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from .types import FaceDetection


def validate_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16:
        raise ValueError("Expected RGB HxWx3 with both dimensions >= 16")
    return np.ascontiguousarray(image)


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba)
        return validate_rgb(np.asarray(image.convert("RGB")).copy())


def resize_long_side(image: np.ndarray, size: int) -> np.ndarray:
    image = validate_rgb(image)
    if size < 32:
        raise ValueError("size must be >= 32")
    h, w = image.shape[:2]
    scale = min(size / max(h, w), 1.0)
    nw, nh = round(w * scale), round(h * scale)
    if min(nw, nh) < 16:
        raise ValueError("Resize would make the short side < 16 pixels")
    if scale == 1:
        return image.copy()
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def expanded_face_crop(
    image: np.ndarray,
    detection: FaceDetection,
    top: float = 0.85,
    bottom: float = 0.55,
    side: float = 0.55,
) -> tuple[np.ndarray, FaceDetection, tuple[int, int, int, int]]:
    """Crop a face with headroom for a hat and context for line-art generation."""

    from .types import FaceDetection

    image = validate_rgb(image)
    if not isinstance(detection, FaceDetection):
        raise TypeError("detection must be a FaceDetection")
    width, height = detection.width, detection.height
    x0 = max(0, int(np.floor(detection.x1 - side * width)))
    y0 = max(0, int(np.floor(detection.y1 - top * height)))
    x1 = min(image.shape[1], int(np.ceil(detection.x2 + side * width)))
    y1 = min(image.shape[0], int(np.ceil(detection.y2 + bottom * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Face crop is empty")
    crop = image[y0:y1, x0:x1].copy()
    return crop, detection.translated(x0, y0), (x0, y0, x1, y1)
