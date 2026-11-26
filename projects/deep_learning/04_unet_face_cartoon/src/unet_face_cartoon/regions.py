"""只在明确的语义区域内编辑，边界软化后仍保护五官。"""

import cv2
import numpy as np
from numpy.typing import NDArray

from .labels import FaceLabel, HEAD_LABELS


def head_foreground(labels: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """保留含主要皮肤的头像连通域，去掉衣领上被误判的孤立头发/帽子。

    不改动原始标签、不按面积删除眼睛等小类别；独立饰品可能被去掉，
    所以同时保存原始前景供核对。这是单人头像规则，不用于多人图。
    """
    raw = np.isin(labels, HEAD_LABELS).astype(np.uint8)
    count, components = cv2.connectedComponents(raw, connectivity=8)
    skin_counts = np.bincount(components[labels == FaceLabel.SKIN], minlength=count)
    if count <= 1 or skin_counts[1:].sum() == 0:
        return np.zeros(labels.shape, dtype=np.float32), raw.astype(np.float32)
    selected = int(np.argmax(skin_counts[1:])) + 1
    return (components == selected).astype(np.float32), raw.astype(np.float32)


def inward_weight(mask: NDArray[np.bool_], width: float = 4) -> NDArray[np.float32]:
    if width <= 0:
        raise ValueError("width must be positive")
    padded = np.pad(mask.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    return np.minimum(distance / width, 1).astype(np.float32) * mask


def edit_regions(
    rgb: NDArray[np.uint8], labels: NDArray[np.uint8], skin_strength: float,
    hair_color: tuple[int, int, int] | None = None, hair_strength: float = 0,
) -> tuple[NDArray[np.uint8], dict[str, NDArray]]:
    if not 0 <= skin_strength <= 1 or not 0 <= hair_strength <= 1:
        raise ValueError("Editing strengths must be in [0, 1]")
    if rgb.shape[:2] != labels.shape:
        raise ValueError("Image and labels must share coordinates")
    if hair_strength and hair_color is None:
        raise ValueError("Hair color is required when hair strength is nonzero")
    skin = inward_weight(labels == FaceLabel.SKIN)
    hair = inward_weight(labels == FaceLabel.HAIR)
    result = rgb.astype(np.float32)
    if skin_strength:
        smooth = cv2.bilateralFilter(rgb, 7, 25, 5).astype(np.float32)
        weight = (skin * skin_strength)[..., None]
        result += weight * (smooth - result)
    if hair_strength and hair_color is not None:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        target = cv2.cvtColor(np.array([[hair_color]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
        lab[..., 1:] = target[1:]
        tint = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        weight = (hair * hair_strength)[..., None]
        result += weight * (tint - result)
    foreground, raw = head_foreground(labels)
    masks = {
        "skin_weight": skin, "hair_weight": hair,
        "foreground": foreground, "foreground_raw": raw,
    }
    return np.clip(np.rint(result), 0, 255).astype(np.uint8), masks


def white_input(rgb: NDArray[np.uint8], foreground: NDArray[np.float32]) -> NDArray[np.uint8]:
    matte = foreground[..., None]
    white = np.rint(rgb.astype(np.float32) * matte + 255 * (1 - matte)).astype(np.uint8)
    return cv2.resize(white, (256, 256), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)


PALETTE = np.array([
    [255, 255, 255], [255, 190, 150], [255, 125, 100], [80, 180, 220],
    [60, 110, 220], [90, 150, 255], [130, 80, 30], [180, 110, 40],
    [240, 180, 200], [210, 140, 170], [140, 20, 50], [220, 30, 90],
    [255, 100, 160], [90, 55, 150], [140, 180, 30], [0, 160, 130],
    [180, 160, 20], [150, 150, 150], [70, 100, 100],
], dtype=np.uint8)
