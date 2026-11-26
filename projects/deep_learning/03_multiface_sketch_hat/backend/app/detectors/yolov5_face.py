"""YOLOv5-Face adapter with explicit coordinate restoration.

The upstream checkpoint is a trusted, full PyTorch module. Loading it with
``weights_only=False`` is intentionally local to this adapter and requires the
caller to opt in with ``trust_checkpoint=True``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from ..image_ops import validate_rgb
from ..types import BBox, FaceDetection, Point


def _landmark_scale(coords: torch.Tensor, input_shape: tuple[int, int], image_shape: tuple[int, int], ratio_pad: tuple[tuple[float, float], tuple[float, float]]) -> torch.Tensor:
    gain = ratio_pad[0][0]
    pad_x, pad_y = ratio_pad[1]
    coords = coords.clone()
    coords[:, 0::2] = (coords[:, 0::2] - pad_x) / gain
    coords[:, 1::2] = (coords[:, 1::2] - pad_y) / gain
    coords[:, 0::2].clamp_(0, image_shape[1])
    coords[:, 1::2].clamp_(0, image_shape[0])
    return coords


class YoloV5FaceDetector:
    """Detect faces and return bbox + five landmarks in RGB-frame coordinates."""

    def __init__(
        self,
        weights: str | Path,
        source_root: str | Path,
        device: str = "auto",
        image_size: int = 640,
        confidence: float = 0.50,
        iou_threshold: float = 0.45,
        trust_checkpoint: bool = False,
    ) -> None:
        if not trust_checkpoint:
            raise ValueError("Pass trust_checkpoint=True only for the verified local YOLOv5-Face checkpoint")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.image_size = image_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.source_root = Path(source_root).resolve()
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        if not self.source_root.is_dir():
            raise FileNotFoundError(self.source_root)

        self._load_upstream_symbols()
        # The checkpoint is a local file from the project manifest. The explicit
        # opt-in above keeps arbitrary pickle execution out of accidental paths.
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        self.model = checkpoint["model"].float().fuse().eval().to(self.device)

    def _load_upstream_symbols(self) -> None:
        root = str(self.source_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        # yolov5-face imports thop only for an unused profiling helper. Keeping a
        # tiny fallback avoids adding an unrelated runtime dependency.
        if "thop" not in sys.modules:
            stub = types.ModuleType("thop")
            setattr(stub, "profile", lambda *args, **kwargs: (0, 0))
            setattr(stub, "clever_format", lambda values, fmt: values)
            sys.modules["thop"] = stub
        import importlib

        datasets = importlib.import_module("utils.datasets")
        general = importlib.import_module("utils.general")
        letterbox = datasets.letterbox
        non_max_suppression_face = general.non_max_suppression_face

        self._letterbox = letterbox
        self._nms = non_max_suppression_face

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> list[FaceDetection]:
        image = validate_rgb(image)
        h, w = image.shape[:2]
        padded, ratio, pad = self._letterbox(
            image, new_shape=(self.image_size, self.image_size), auto=False, scaleFill=False, scaleup=False
        )
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().unsqueeze(0)
        tensor = tensor.to(self.device) / 255.0
        prediction = self.model(tensor)[0]
        detections = self._nms(prediction, self.confidence, self.iou_threshold)[0]
        if len(detections) == 0:
            return []
        detections = detections.clone()
        gain = (float(ratio[0]), float(ratio[1])) if not np.isscalar(ratio) else (float(ratio), float(ratio))
        ratio_pad = (gain, (float(pad[0]), float(pad[1])))
        detections[:, 0] = (detections[:, 0] - pad[0]) / gain[0]
        detections[:, 2] = (detections[:, 2] - pad[0]) / gain[0]
        detections[:, 1] = (detections[:, 1] - pad[1]) / gain[1]
        detections[:, 3] = (detections[:, 3] - pad[1]) / gain[1]
        detections[:, [0, 2]].clamp_(0, w)
        detections[:, [1, 3]].clamp_(0, h)
        landmarks = _landmark_scale(detections[:, 5:15], (self.image_size, self.image_size), (h, w), ratio_pad)
        result: list[FaceDetection] = []
        for row, points in zip(detections, landmarks):
            coords: tuple[Point, Point, Point, Point, Point] = (
                (float(points[0]), float(points[1])),
                (float(points[2]), float(points[3])),
                (float(points[4]), float(points[5])),
                (float(points[6]), float(points[7])),
                (float(points[8]), float(points[9])),
            )
            bbox: BBox = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            result.append(FaceDetection(bbox, float(row[4]), coords))
        return sorted(result, key=lambda item: (item.x1, item.y1))
