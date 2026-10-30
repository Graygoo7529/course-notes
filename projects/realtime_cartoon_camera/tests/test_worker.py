"""Protocol and PyTorch/OpenVINO parity smoke tests for the binary worker."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import struct
import subprocess
import unittest

import numpy as np
from PIL import Image
from websockets.sync.client import connect
from websockets.typing import Origin

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("CARTOON_PYTHON", r"C:\Anaconda3\envs\d2l\python.exe")
INPUT = ROOT.parent / "cnn_face_cartoon/data/input/grace_hopper.jpg"


def packet(meta: dict, payload: bytes) -> bytes:
    header = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    return struct.pack("!I", len(header)) + header + payload


def unpack(value: bytes) -> tuple[dict, bytes]:
    size = struct.unpack("!I", value[:4])[0]
    return json.loads(value[4:size + 4]), value[size + 4:]


@unittest.skipUnless(INPUT.exists(), "requires the existing cnn_face_cartoon demo input")
class WorkerProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Image.open(INPUT) as source:
            cls.source_size = source.size
            original = io.BytesIO()
            source.convert("RGB").save(original, format="PNG")
            cls.original = original.getvalue()
            preview = source.convert("RGB")
            preview.thumbnail((256, 256))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=90)
            cls.payload = buffer.getvalue()
            width, height = preview.size
        cls.meta = {"sessionId": "test", "frameId": 1, "configRevision": 0, "width": width, "height": height,
                    "pixelFormat": "jpeg", "rotation": 0, "mirrored": False, "kind": "preview",
                    "style": {"revision": 0, "lineDensity": .35, "palette": "natural_warm", "mirrored": False}}

    def run_backend(self, backend: str, capture=False):
        process = subprocess.Popen([PYTHON, "-X", "utf8", "python/worker.py"], cwd=ROOT,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, encoding="utf-8")
        try:
            assert process.stdout and process.stdin
            descriptor = json.loads(process.stdout.readline())
            with connect(descriptor["url"], origin=Origin("http://127.0.0.1:1420"), proxy=None, max_size=24 * 1024 * 1024) as socket:
                socket.send(json.dumps({"token": descriptor["token"], "backend": backend, "sessionId": "test"}))
                ready = json.loads(socket.recv(timeout=60))
                self.assertEqual(ready["model"], "MobileNetV3-Large")
                socket.send(packet(self.meta, self.payload))
                response = socket.recv(timeout=60)
                assert isinstance(response, bytes), response
                meta, image = unpack(response)
                self.assertEqual(meta["faceStatus"], "detected")
                self.assertGreater(meta["processingMs"], 0)
                self.assertGreater(len(image), 1000)
                preview = Image.open(io.BytesIO(image)).convert("RGB")
                if capture:
                    capture_meta = {**self.meta, "frameId": 2, "pixelFormat": "png", "kind": "capture",
                                    "width": self.source_size[0], "height": self.source_size[1]}
                    socket.send(packet(capture_meta, self.original))
                    response = socket.recv(timeout=60)
                    assert isinstance(response, bytes), response
                    capture_info, capture_image = unpack(response)
                    self.assertEqual(capture_info["renderer"], "bounded_spline")
                    self.assertEqual(capture_info["sourceSize"], list(self.source_size))
                    ratio = 384 / max(self.source_size)
                    expected_size = tuple(round(dimension * ratio) for dimension in self.source_size)
                    self.assertEqual(Image.open(io.BytesIO(capture_image)).size, expected_size)
                    self.assertTrue(capture_image.startswith(b"\x89PNG\r\n\x1a\n"))
                return ready, meta, preview
        finally:
            process.stdin.close()
            process.terminate()
            process.wait(timeout=20)
            process.stdout.close()

    def test_openvino_and_pytorch_backend_contract(self):
        pytorch = self.run_backend("pytorch")
        openvino = self.run_backend("openvino", capture=True)
        self.assertEqual(pytorch[0]["model"], openvino[0]["model"])
        self.assertEqual(pytorch[1]["width"], openvino[1]["width"])
        np.testing.assert_allclose(np.asarray(pytorch[2]), np.asarray(openvino[2]), atol=2)


if __name__ == "__main__":
    unittest.main()
