"""FastAPI camera endpoint for the second implementation stage."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from ..pipeline import CorePipeline
from ..tracking.simple import SimpleTracker
from .protocol import image_to_base64, pack_response


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _decode_jpeg(payload: bytes):
    import numpy as np

    array = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("WebSocket payload is not a decodable JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _encode_jpeg(image) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise RuntimeError("Unable to encode scene JPEG")
    return encoded.tobytes()


def _detection_metadata(detections, track_ids=None):
    track_ids = track_ids or [None] * len(detections)
    return [
        {
            "trackId": track_id,
            "score": round(item.score, 4),
            "bbox": [round(value, 2) for value in item.bbox],
            "landmarks5": [[round(x, 2), round(y, 2)] for x, y in item.landmarks5],
        }
        for item, track_id in zip(detections, track_ids)
    ]


class CameraSession:
    def __init__(self, pipeline: CorePipeline) -> None:
        self.pipeline = pipeline
        self.tracker = SimpleTracker()
        self.last_frame = None
        self.last_frame_id = 0
        self.last_tracked = None
        self.last_sketch: np.ndarray | None = None
        self.last_sketch_at = 0.0
        # GPU keeps the full-quality Lineart update on every accepted frame.
        # CPU fallback caches it to keep the camera UI responsive.
        default_interval = 0.0 if pipeline.sketcher.device.type == "cuda" else 250.0
        self.sketch_interval_ms = float(os.environ.get("MULTIFACE_SKETCH_INTERVAL_MS", default_interval))

    def process(self, frame, frame_id: int, make_cards: bool = False):
        started = time.perf_counter()
        if make_cards and frame_id == self.last_frame_id and self.last_tracked is not None:
            tracked = self.last_tracked
        else:
            tracked = self.tracker.update(self.pipeline.detector(frame))
        now = time.perf_counter()
        sketch_updated = (
            make_cards
            or self.last_sketch is None
            or self.last_sketch.shape != frame.shape[:2]
            or self.sketch_interval_ms <= 0
            or (now - self.last_sketch_at) * 1000.0 >= self.sketch_interval_ms
        )
        if sketch_updated:
            self.last_sketch = self.pipeline.sketcher.predict(frame)
            self.last_sketch_at = now
        result = self.pipeline.render(
            frame,
            tracked,
            make_cards=make_cards,
            sketch_gray=self.last_sketch,
        )
        cards = [_encode_jpeg(card) for card in result.cards]
        metadata = {
            "kind": "capture" if make_cards else "scene",
            "frameId": frame_id,
            "width": int(result.scene_rgb.shape[1]),
            "height": int(result.scene_rgb.shape[0]),
            "processingMs": round((time.perf_counter() - started) * 1000.0, 1),
            "sketchUpdated": sketch_updated,
            "detections": _detection_metadata(result.detections, [item.track_id for item in tracked]),
            "cards": [image_to_base64(card) for card in cards],
        }
        self.last_frame = frame.copy()
        self.last_frame_id = frame_id
        self.last_tracked = tracked
        return metadata, _encode_jpeg(result.scene_rgb)


@asynccontextmanager
async def lifespan(application: FastAPI):
    device = os.environ.get("MULTIFACE_DEVICE", "auto")
    pipeline = CorePipeline.from_project(PROJECT_ROOT, device=device)
    if os.environ.get("MULTIFACE_WARMUP", "1").lower() not in {"0", "false", "no"}:
        pipeline.warmup()
    application.state.pipeline = pipeline
    yield
    application.state.pipeline = None


app = FastAPI(title="Multiface Sketch and Hat", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    pipeline = getattr(app.state, "pipeline", None)
    device = str(pipeline.detector.device) if pipeline is not None else os.environ.get("MULTIFACE_DEVICE", "auto")
    return JSONResponse({"ok": True, "stage": "web-camera", "device": device})


@app.websocket("/ws/process")
async def process_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = CameraSession(websocket.app.state.pipeline)
    frame_id = 0
    try:
        while True:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, WebSocketDisconnected):
                return
            if message.get("bytes") is not None:
                frame_id += 1
                frame = _decode_jpeg(message["bytes"])
                metadata, scene = await asyncio.to_thread(session.process, frame, frame_id, False)
                await websocket.send_bytes(pack_response(metadata, scene))
                continue
            text = message.get("text")
            if text is None:
                continue
            command = json.loads(text)
            if command.get("kind") != "capture":
                await websocket.send_json({"kind": "error", "message": "Unknown command"})
                continue
            if session.last_frame is None:
                await websocket.send_json({"kind": "error", "message": "No camera frame received"})
                continue
            metadata, scene = await asyncio.to_thread(session.process, session.last_frame, session.last_frame_id, True)
            await websocket.send_bytes(pack_response(metadata, scene))
    except Exception as error:
        # Some Starlette/TestClient versions surface the disconnect class from
        # a reloaded module, so use the class name as a defensive fallback.
        if isinstance(error, (WebSocketDisconnect, WebSocketDisconnected)) or error.__class__.__name__ in {
            "WebSocketDisconnect",
            "WebSocketDisconnected",
        }:
            return
        if isinstance(error, (ValueError, json.JSONDecodeError)):
            await websocket.send_json({"kind": "error", "message": str(error)})
            return
        raise
