"""统一 RGB、标签图和生成器数值范围。"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from .labels import FaceLabel


def read_rgb(path: Path) -> NDArray[np.uint8]:
    """先校正 EXIF；透明图片合成白底，返回 H×W×3 uint8 RGB。"""
    with Image.open(path) as image:
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return np.array(Image.alpha_composite(background, rgba).convert("RGB"))


def resize_labels(labels: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    """size 为 (width, height)，标签仅使用最近邻插值。"""
    if labels.ndim != 2 or labels.dtype != np.uint8 or labels.size == 0:
        raise ValueError("labels must be a nonempty HxW uint8 array")
    if int(labels.max()) >= len(FaceLabel):
        raise ValueError("labels must use the official IDs 0..18")
    if len(size) != 2 or min(size) < 1:
        raise ValueError("size must contain positive width and height")
    return np.array(Image.fromarray(labels).resize(size, Image.Resampling.NEAREST))


def generator_input(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """已规范化的头像 → 1×3×256×256，[-1, 1]；此处不做对齐裁剪。"""
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError("generator input must be 256x256x3 uint8 RGB")
    scaled = rgb.astype(np.float32) / np.float32(127.5) - np.float32(1)
    return np.ascontiguousarray(scaled.transpose(2, 0, 1)[None])


def generator_output(chw: NDArray[np.float32]) -> NDArray[np.uint8]:
    """3×256×256 → RGB；先检查有限值，clip 后再转 uint8。"""
    if chw.shape != (3, 256, 256) or not np.isfinite(chw).all():
        raise ValueError("generator output must be finite 3x256x256")
    rgb = (chw.astype(np.float32).transpose(1, 2, 0) + 1) * np.float32(127.5)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
