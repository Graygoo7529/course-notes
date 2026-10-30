"""Shared canvas, a single Sobel edge field, and optional face geometry."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .contracts import Prepared, Transform


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def normalize(field: np.ndarray, mask: np.ndarray, quantiles=(0.1, 0.95), epsilon=1e-6) -> np.ndarray:
    values = field[mask]
    if not len(values):
        return np.zeros_like(field, dtype=np.float32)
    lo, hi = np.quantile(values, quantiles)
    if hi - lo < epsilon:
        return np.zeros_like(field, dtype=np.float32)
    return np.where(mask, np.clip((field - lo) / (hi - lo), 0, 1), 0).astype(np.float32)


class FaceGeometry:
    def __init__(self, model_path: Path, enabled: bool = True):
        self.detector: Any = None
        self.status = "disabled"
        if not enabled:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.face_landmarker import (
                FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections,
            )
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            self.mp = mp
            self.connections = FaceLandmarksConnections
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
                running_mode=VisionTaskRunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                output_face_blendshapes=False, output_facial_transformation_matrixes=False)
            self.detector = FaceLandmarker.create_from_options(options)
            self.status = "ready"
        except Exception as exc:
            self.status = f"unavailable: {type(exc).__name__}: {exc}"

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()

    def detect(self, rgb: np.ndarray) -> tuple[dict, dict, str]:
        h, w = rgb.shape[:2]
        empty = {key: np.zeros((h, w), dtype=bool) for key in ["features", "cheeks", "face", "contour"]}
        if self.detector is None:
            return {}, empty, self.status
        try:
            result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                       data=np.ascontiguousarray(rgb)))
        except Exception as exc:
            return {}, empty, f"detection_error: {type(exc).__name__}: {exc}"
        if not result.face_landmarks:
            return {}, empty, "not_detected"
        xy = np.asarray([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)
        connections = self.connections
        groups = {"landmarks": xy}
        feature_mask = np.zeros((h, w), np.uint8)
        for name, attr in [("left_eye", "LEFT_EYE"), ("right_eye", "RIGHT_EYE"),
                           ("left_brow", "LEFT_EYEBROW"), ("right_brow", "RIGHT_EYEBROW"),
                           ("mouth", "LIPS"), ("oval", "FACE_OVAL")]:
            edges = getattr(connections, f"FACE_LANDMARKS_{attr}")
            segments = np.array([[xy[e.start], xy[e.end]] for e in edges], np.float32)
            groups[name] = segments.reshape(-1, 2)
            if name != "oval":
                for a, b in segments:
                    cv2.line(feature_mask, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), 1, 7)
        face = np.zeros((h, w), np.uint8)
        hull = cv2.convexHull(np.rint(groups["oval"]).astype(np.int32))
        cv2.fillConvexPoly(face, hull, 1)
        contour = cv2.dilate(face, np.ones((9, 9), np.uint8)) - cv2.erode(face, np.ones((9, 9), np.uint8))
        cheeks = np.zeros_like(face)
        for ids in [[50, 101, 205, 206, 216, 212, 214, 192], [280, 330, 425, 426, 436, 432, 434, 416]]:
            cv2.fillConvexPoly(cheeks, cv2.convexHull(np.rint(xy[ids]).astype(np.int32)), 1)
        cheeks[feature_mask > 0] = 0
        return groups, {"features": feature_mask > 0, "face": face > 0,
                        "contour": contour > 0, "cheeks": cheeks > 0}, "detected"


def prepare(image: np.ndarray, config: dict, face_model: FaceGeometry) -> Prepared:
    h, w = image.shape[:2]
    size = int(config["canvas_size"])
    scale = size / max(h, w)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.asarray(Image.fromarray(image).resize((rw, rh), Image.Resampling.LANCZOS))
    left, top = (size - rw) // 2, (size - rh) // 2
    canvas = cv2.copyMakeBorder(resized, top, size - rh - top, left, size - rw - left, cv2.BORDER_REPLICATE)
    valid = np.zeros((size, size), bool)
    valid[top:top + rh, left:left + rw] = True
    rgb = cv2.bilateralFilter(canvas, 5, 18, 2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray[valid]))
    gain = min(1.35, 65 / max(median, 1)) if median < 60 else 1.0
    if config.get("brightness_correction") == "underexposed_only" and gain > 1:
        rgb = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    else:
        gain = 1.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    interior = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    edge = normalize(np.hypot(gx, gy), interior)
    groups, masks, status = face_model.detect(rgb)
    for key in masks:
        masks[key] &= interior
    transform = Transform((w, h), (rw, rh), (size, size), (rw / w, rh / h), (left, top))
    return Prepared(canvas.astype(np.uint8), rgb.astype(np.uint8), valid, transform, edge, np.arctan2(gy, gx).astype(np.float32),
                    groups, status, masks, {"brightness_gain": gain, "input_luminance_median": median,
                                           "face_landmark_count": len(groups.get("landmarks", []))})


def unpad(array: np.ndarray, transform: Transform) -> np.ndarray:
    left, top = transform.padding_xy
    w, h = transform.resized_size
    return array[top:top + h, left:left + w].copy()


def geometry_preview(prepared: Prepared) -> np.ndarray:
    preview = prepared.original_rgb.copy()
    for name, pairs in prepared.face_groups.items():
        if name == "landmarks":
            continue
        color = (54, 139, 143) if name == "oval" else (215, 98, 67)
        for a, b in pairs.reshape(-1, 2, 2):
            cv2.line(preview, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, 1, cv2.LINE_AA)
    return preview
