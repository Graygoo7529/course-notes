"""FastAPI camera endpoint for the second implementation stage."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

import cv2
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
        self.busy = False

    def process(self, frame, frame_id: int, make_cards: bool = False):
        tracked = self.tracker.update(self.pipeline.detector(frame))
        result = self.pipeline.render(frame, tracked, make_cards=make_cards)
        cards = [_encode_jpeg(card) for card in result.cards]
        metadata = {
            "kind": "capture" if make_cards else "scene",
            "frameId": frame_id,
            "width": int(result.scene_rgb.shape[1]),
            "height": int(result.scene_rgb.shape[0]),
            "detections": _detection_metadata(result.detections, [item.track_id for item in tracked]),
            "cards": [image_to_base64(card) for card in cards],
        }
        self.last_frame = frame.copy()
        self.last_frame_id = frame_id
        return metadata, _encode_jpeg(result.scene_rgb)


@asynccontextmanager
async def lifespan(application: FastAPI):
    device = os.environ.get("MULTIFACE_DEVICE", "auto")
    application.state.pipeline = CorePipeline.from_project(PROJECT_ROOT, device=device)
    yield
    application.state.pipeline = None


app = FastAPI(title="Multiface Sketch and Hat", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "stage": "web-camera", "device": os.environ.get("MULTIFACE_DEVICE", "auto")})


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
