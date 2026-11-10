"""Offline scene and portrait-card pipeline for the first implementation stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .compositing.hats import HAT_STYLES, HatStyle, overlay_hats
from .detectors.yolov5_face import YoloV5FaceDetector
from .image_ops import expanded_face_crop, resize_long_side, validate_rgb
from .sketchers.lineart import LineartSketcher
from .tracking.simple import TrackedFace
from .types import FaceDetection


@dataclass
class SceneResult:
    input_rgb: np.ndarray
    sketch_rgb: np.ndarray
    scene_rgb: np.ndarray
    detections: list[FaceDetection]
    cards: list[np.ndarray]


class CorePipeline:
    """Run one complete still-image frame; the Web stage will reuse this API."""

    def __init__(
        self,
        detector: YoloV5FaceDetector,
        sketcher: LineartSketcher,
        scene_max_size: int = 512,
        hat_style: HatStyle | str = "mist_blue_top",
    ) -> None:
        self.detector = detector
        self.sketcher = sketcher
        self.scene_max_size = scene_max_size
        self.hat_style = HAT_STYLES[hat_style] if isinstance(hat_style, str) else hat_style

    @classmethod
    def from_project(cls, project_root: str | Path, device: str = "auto", scene_max_size: int = 512) -> "CorePipeline":
        root = Path(project_root)
        source = root / "third_party" / "yolov5-face"
        weights = root / "model" / "yolov5_face" / "yolov5s-face.pt"
        detector = YoloV5FaceDetector(weights, source, device=device, trust_checkpoint=True)
        sketcher = LineartSketcher(root / "model" / "informative_drawings", style="anime_style", device=device)
        return cls(detector, sketcher, scene_max_size=scene_max_size)

    @staticmethod
    def _to_rgb(sketch: np.ndarray) -> np.ndarray:
        return np.repeat(sketch[:, :, None], 3, axis=2)

    def process(self, frame_rgb: np.ndarray, make_cards: bool = False) -> SceneResult:
        frame = resize_long_side(validate_rgb(frame_rgb), self.scene_max_size)
        detections = self.detector(frame)
        return self.render(frame, detections, make_cards=make_cards)

    def render(
        self,
        frame_rgb: np.ndarray,
        detections: list[FaceDetection] | list[TrackedFace],
        make_cards: bool = False,
    ) -> SceneResult:
        frame = validate_rgb(frame_rgb)
        plain_detections = [
            item.detection if isinstance(item, TrackedFace) else item for item in detections
        ]
        sketch = self.sketcher.predict(frame)
        sketch_rgb = self._to_rgb(sketch)
        scene = overlay_hats(sketch_rgb, plain_detections, self.hat_style)
        cards: list[np.ndarray] = []
        if make_cards:
            for detection in plain_detections:
                crop, local_detection, _ = expanded_face_crop(frame, detection)
                card_sketch = self._to_rgb(self.sketcher.predict(crop))
                cards.append(overlay_hats(card_sketch, [local_detection], self.hat_style))
        return SceneResult(frame, sketch_rgb, scene, plain_detections, cards)


def draw_detections(image: np.ndarray, detections: list[FaceDetection]) -> np.ndarray:
    output = image.copy()
    for index, detection in enumerate(detections, 1):
        x1, y1, x2, y2 = (round(value) for value in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (54, 110, 205), 2, cv2.LINE_AA)
        for x, y in detection.landmarks5:
            cv2.circle(output, (round(x), round(y)), 2, (205, 75, 75), -1, cv2.LINE_AA)
        cv2.putText(output, f"face {index} {detection.score:.2f}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (54, 110, 205), 1, cv2.LINE_AA)
    return output
