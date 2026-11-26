"""关键点对齐与官方头像构图；坐标矩阵按像素中心定义。"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class Portrait:
    rgb: NDArray[np.uint8]
    original_to_portrait: NDArray[np.float64]
    landmarks: NDArray[np.float64]
    crop_box: tuple[int, int, int, int]
    rotation_degrees: float


def resize_matrix(sx: float, sy: float) -> NDArray[np.float64]:
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)


def transform_points(points: NDArray, matrix: NDArray) -> NDArray[np.float64]:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix.T)[:, :2]


def limit_image(rgb: NDArray[np.uint8], maximum: int = 1600) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    h, w = rgb.shape[:2]
    if max(h, w) <= maximum:
        return rgb, np.eye(3, dtype=np.float64)
    ratio = maximum / max(h, w)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)
    return resized, resize_matrix(nw / w, nh / h)


def align_crop(rgb: NDArray[np.uint8], points: NDArray, original_to_working: NDArray) -> Portrait:
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError("Expected finite 68x2 landmarks")
    eye = points[45] - points[36]
    if np.linalg.norm(eye) < 4:
        raise ValueError("Invalid or too small eye separation")
    angle = float(np.degrees(np.arctan2(eye[1], eye[0])))
    h, w = rgb.shape[:2]
    rad = np.radians(angle)
    nw = int(w * abs(np.cos(rad)) + h * abs(np.sin(rad)))
    nh = int(h * abs(np.cos(rad)) + w * abs(np.sin(rad)))
    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotation[:, 2] += [nw // 2 - w // 2, nh // 2 - h // 2]
    rotate3 = np.vstack((rotation, [0, 0, 1]))
    rotated_points = transform_points(points, rotate3)
    low, high = rotated_points.min(axis=0), rotated_points.max(axis=0)
    fw, fh = high - low
    if min(fw, fh) < 16:
        raise ValueError("Face too small for a stable portrait crop")
    left, right = int(low[0] - 0.3 * fw), int(high[0] + 0.3 * fw)
    top, bottom = int(low[1] - 0.8 * fh), int(high[1] + 0.3 * fh)
    if bottom - top > right - left:
        left -= ((bottom - top) - (right - left)) // 2
        right = left + bottom - top
    else:
        top -= ((right - left) - (bottom - top)) // 2
        bottom = top + right - left
    side = bottom - top + 1
    if side > 4 * max(h, w):
        raise ValueError("Implausible landmark crop")
    crop3 = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
    working_to_crop = crop3 @ rotate3
    crop = cv2.warpAffine(rgb, working_to_crop[:2], (side, side),
                          flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    interpolation = cv2.INTER_AREA if side > 512 else cv2.INTER_LINEAR
    portrait = cv2.resize(crop, (512, 512), interpolation=interpolation).astype(np.uint8, copy=False)
    working_to_portrait = resize_matrix(512 / side, 512 / side) @ working_to_crop
    return Portrait(
        portrait, working_to_portrait @ original_to_working,
        transform_points(points, working_to_portrait),
        (left, top, right, bottom), angle,
    )
