"""Procedural muted-color hats with landmark-aware affine placement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..types import FaceDetection


@dataclass(frozen=True)
class HatStyle:
    name: str
    body: tuple[int, int, int]
    band: tuple[int, int, int]
    accent: tuple[int, int, int]
    opacity: float = 0.78


HAT_STYLES = {
    "mist_blue_top": HatStyle("mist_blue_top", (137, 162, 177), (93, 119, 138), (210, 177, 124), 0.56),
    "sage_bowler": HatStyle("sage_bowler", (143, 164, 145), (91, 119, 99), (202, 178, 124), 0.58),
    "rose_cloche": HatStyle("rose_cloche", (177, 144, 151), (139, 101, 111), (216, 181, 132), 0.58),
}


def make_hat(style: HatStyle, size: tuple[int, int] = (320, 190)) -> np.ndarray:
    """Create a soft RGBA illustration; alpha keeps the line art visible."""

    width, height = size
    scale = 4
    canvas = np.zeros((height * scale, width * scale, 4), dtype=np.uint8)
    body = tuple(int(v) for v in style.body) + (round(255 * style.opacity),)
    band = tuple(int(v) for v in style.band) + (round(255 * min(1.0, style.opacity + 0.08)),)
    accent = tuple(int(v) for v in style.accent) + (round(255 * style.opacity),)
    def rect(box: tuple[int, int, int, int], color: tuple[int, ...]) -> None:
        cv2.rectangle(canvas, tuple(v * scale for v in box[:2]), tuple(v * scale for v in box[2:]), color, -1, cv2.LINE_AA)
    if style.name.endswith("bowler"):
        cv2.ellipse(canvas, (width * scale // 2, 95 * scale), (105 * scale, 75 * scale), 0, 180, 360, body, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (width * scale // 2, 145 * scale), (145 * scale, 24 * scale), 0, 0, 360, body, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (width * scale // 2, 131 * scale), (100 * scale, 12 * scale), 0, 0, 360, band, -1, cv2.LINE_AA)
    elif style.name.endswith("cloche"):
        cv2.ellipse(canvas, (width * scale // 2, 105 * scale), (115 * scale, 88 * scale), 0, 180, 360, body, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (width * scale // 2, 148 * scale), (152 * scale, 23 * scale), 0, 0, 360, body, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (width * scale // 2, 136 * scale), (103 * scale, 10 * scale), 0, 0, 360, band, -1, cv2.LINE_AA)
        cv2.circle(canvas, (width * scale // 2, 69 * scale), 10 * scale, accent, -1, cv2.LINE_AA)
    else:
        rect((70, 19, 250, 139), body)
        rect((70, 101, 250, 131), band)
        cv2.ellipse(canvas, (width * scale // 2, 145 * scale), (155 * scale, 24 * scale), 0, 0, 360, body, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (width * scale // 2, 145 * scale), (115 * scale, 10 * scale), 0, 0, 360, accent, -1, cv2.LINE_AA)
    small = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    # The brim center is the stable attachment point for all three styles.
    return small


def overlay_hat(image: np.ndarray, detection: FaceDetection, style: HatStyle) -> np.ndarray:
    """Place one muted-color hat on a white-paper sketch or RGB frame."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be RGB uint8")
    asset = make_hat(style)
    target_width = max(24, round(detection.width * 1.75))
    angle = detection.eye_angle_degrees
    eye_y = (detection.landmarks5[0][1] + detection.landmarks5[1][1]) / 2.0
    target = (detection.x1 + detection.width / 2.0, eye_y - detection.height * 0.26)
    # Warp at the actual canvas size rather than the asset's native size.
    height = max(1, round(asset.shape[0] * target_width / asset.shape[1]))
    resized = cv2.resize(asset, (target_width, height), interpolation=cv2.INTER_AREA)
    anchor = (target_width / 2.0, height * 0.76)
    matrix = cv2.getRotationMatrix2D(anchor, angle, 1.0)
    matrix[0, 2] += target[0] - anchor[0]
    matrix[1, 2] += target[1] - anchor[1]
    warped = cv2.warpAffine(resized, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    foreground = warped[:, :, :3].astype(np.float32)
    result = image.astype(np.float32) * (1.0 - alpha) + foreground * alpha
    return np.round(np.clip(result, 0, 255)).astype(np.uint8)


def overlay_hats(image: np.ndarray, detections: list[FaceDetection], style: HatStyle) -> np.ndarray:
    result = image.copy()
    for detection in detections:
        result = overlay_hat(result, detection, style)
    return result
