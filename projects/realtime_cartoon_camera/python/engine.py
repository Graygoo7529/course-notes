"""Shared course algorithm with resident encoders and bounded video state."""
from __future__ import annotations

import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

APP = Path(__file__).resolve().parents[1]
CORE = Path(os.environ.get("CARTOON_CORE", APP.parent / "cnn_face_cartoon"))
sys.path.insert(0, str(CORE / "src"))
from face_cartoon.contracts import Layers
from face_cartoon.features import Encoder, extract
from face_cartoon.preprocess import FaceGeometry, prepare
from face_cartoon.rendering import color_layer, compose, stylize
from face_cartoon.strokes import nonmaximum


class OpenVINOEncoder:
    def __init__(self, size: int):
        import openvino as ov
        xml = APP / f"models/mobilenetv3_features_{size}.xml"
        manifest = json.loads(xml.with_suffix(".json").read_text(encoding="utf-8"))
        core = ov.Core()
        self.compiled = core.compile_model(str(xml), "CPU", {
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32"})
        self.request = self.compiled.create_infer_request()
        self.mean = np.asarray(manifest["input"]["mean"], np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(manifest["input"]["std"], np.float32).reshape(1, 3, 1, 1)
        self.layers = [{"reduction": item["stride"], "num_chs": item["channels"], "module": item["name"]}
                       for item in manifest["outputs"]]
        self.info = {"backend": "openvino", "device": "CPU", "manifest": manifest}

    def forward(self, rgb: np.ndarray):
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255
        self.request.infer([(tensor - self.mean) / self.std])
        return [torch.from_numpy(self.request.get_output_tensor(i).data.copy()) for i in range(3)]

    def close(self):
        pass


class VideoGeometry(FaceGeometry):
    """Reuse reference geometry groups, VIDEO timestamps and scaled 7/9px bands."""
    def __init__(self):
        self.reference: Any = FaceGeometry(CORE / "model/face_landmarker.task", enabled=False)
        self.detector: Any = None
        self.timestamp = 0
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            self.reference.mp = mp
            self.reference.connections = FaceLandmarksConnections
            self.detector = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=(CORE / "model/face_landmarker.task").read_bytes()),
                running_mode=VisionTaskRunningMode.VIDEO, num_faces=1))
            self.reference.detector = self
        except Exception as exc:
            self.reference.status = f"unavailable: {exc}"

    def detect(self, rgb):
        image = rgb
        # FaceGeometry calls back with mp.Image; the outer call is a numpy frame.
        if not isinstance(image, np.ndarray):
            self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
            return self.detector.detect_for_video(image, self.timestamp)
        size = image.shape[0]
        canonical = cv2.resize(image, (384, 384)) if size != 384 else image
        groups, masks, status = self.reference.detect(canonical)
        if size != 384:
            groups = {key: points * (size / 384) for key, points in groups.items()}
            masks = {key: cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
                     for key, mask in masks.items()}
        return groups, masks, status

    def close(self):
        if self.detector is not None:
            self.detector.close()


class Session:
    def __init__(self, backend="openvino"):
        if backend not in {"openvino", "pytorch"}:
            raise ValueError("Unknown backend")
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        self.base = json.loads((CORE / "config.json").read_text(encoding="utf-8"))
        self.backend = backend
        started = time.perf_counter()
        self.encoders: dict[int, Any] = {}
        for size in [256, 384]:
            self.encoders[size] = OpenVINOEncoder(size) if backend == "openvino" else (
                self.encoders.get(256) or Encoder(CORE, "mobilenetv3", self.base["features"], "cpu"))
            self.encoders[size].forward(np.full((size, size, 3), 128, np.uint8))
        self.geometry = VideoGeometry()
        self.cold_start_ms = (time.perf_counter() - started) * 1000
        self.gain = 1.0
        self.dark = False
        self.threshold: float | None = None
        self.revision = -1

    def reset(self):
        self.gain, self.dark, self.threshold = 1.0, False, None

    def run(self, data: bytes, style: dict, capture=False):
        start = time.perf_counter()
        with Image.open(io.BytesIO(data)) as file:
            if file.width * file.height > 16_000_000:
                raise ValueError("Input exceeds 16 megapixels")
            rgb = np.asarray(ImageOps.exif_transpose(file).convert("RGB")).copy()
        config = copy.deepcopy(self.base)
        size = 384 if capture else 256
        config["preprocess"]["canvas_size"] = size
        config["style"].update(line_density=float(style["lineDensity"]), palette=style["palette"])
        if not 0 <= config["style"]["line_density"] <= 1 or style["palette"] not in {"natural_warm", "pastel"}:
            raise ValueError("Invalid style")
        if style["revision"] != self.revision:
            self.threshold = None
            self.revision = style["revision"]
        # Keep scalar state only. Never reuse previous S or blend moving pixels.
        if not capture:
            median = float(np.median(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
            self.dark = median < (65 if self.dark else 55)
            target = min(1.35, 65 / max(1, median)) if self.dark else 1
            self.gain += .25 * (target - self.gain)
            rgb = np.clip(rgb.astype(np.float32) * self.gain, 0, 255).astype(np.uint8)
            config["preprocess"]["brightness_correction"] = "none"
        decode_ms = (time.perf_counter() - start) * 1000
        stage = time.perf_counter()
        prepared = prepare(rgb, config["preprocess"], self.geometry)
        prepare_ms = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        structure = extract(prepared, self.encoders[size], config["features"])
        cnn_ms = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        colors = color_layer(prepared.rgb, config["style"])
        if capture:
            layers = stylize(prepared, structure, config, colors)
        else:
            reference = nonmaximum(prepared.edge, prepared.edge_direction)
            values = reference[reference > .04]
            density = config["style"]["line_density"]
            high = max(.1, (float(np.quantile(values, .85 - .3 * density)) if len(values) else 1) * (.48 - .2 * density))
            self.threshold = high if self.threshold is None else self.threshold + .25 * (high - self.threshold)
            multiplier = np.ones(prepared.edge.shape, np.float32)
            multiplier[prepared.masks["cheeks"]] = 1.3
            multiplier[prepared.masks["features"]] = .8
            nms = nonmaximum(structure.stroke_score, prepared.edge_direction)
            weak = nms >= self.threshold * .45 * multiplier
            strong = nms >= self.threshold * multiplier
            count, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
            labels = labels.astype(np.int32)
            accepted = np.zeros(count, bool)
            accepted[np.unique(labels[strong])] = True
            protected = np.unique(labels[prepared.masks["features"] & weak])
            accepted &= (stats[:, cv2.CC_STAT_AREA] >= max(2, round(7 * (size / 384) ** 2))) | np.isin(np.arange(count), protected)
            accepted[0] = False
            skeleton = cv2.ximgproc.thinning(accepted[labels].astype(np.uint8) * 255)
            alpha = (skeleton / 255 * prepared.valid_mask).astype(np.float32)
            layers = Layers(colors, alpha, [], None, {"raster_preview": True, "threshold": self.threshold})
        result = compose(layers, prepared.transform, config)
        render_ms = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        if capture and style.get("mirrored"):
            result = replace(result, final_rgb=result.final_rgb[:, ::-1].copy())
        buffer = io.BytesIO()
        Image.fromarray(result.final_rgb).save(buffer, format="PNG" if capture else "JPEG", quality=88)
        return buffer.getvalue(), {"processingMs": (time.perf_counter() - start) * 1000,
            "faceStatus": prepared.face_status, "mime": "image/png" if capture else "image/jpeg",
            "width": result.final_rgb.shape[1], "height": result.final_rgb.shape[0],
            "timings": {"decode": decode_ms, "prepareFace": prepare_ms, "cnnStructure": cnn_ms,
                        "render": render_ms, "encode": (time.perf_counter() - stage) * 1000},
            "renderer": "bounded_spline" if capture else "raster", "sourceSize": list(rgb.shape[:2][::-1])}

    def close(self):
        self.geometry.close()
        for encoder in {id(v): v for v in self.encoders.values()}.values():
            encoder.close()
