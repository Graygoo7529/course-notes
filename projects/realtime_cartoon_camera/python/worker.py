"""Loopback-only authenticated binary frames. No per-frame disk or Base64 IO."""
import argparse
import hmac
import json
import secrets
import struct
import sys
import threading

from websockets.sync.server import serve
from websockets.typing import Origin

MAX_PACKET = 24 * 1024 * 1024


def pack(meta: dict, image: bytes) -> bytes:
    header = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    return struct.pack("!I", len(header)) + header + image


def unpack(packet: bytes) -> tuple[dict, bytes]:
    if len(packet) < 4 or len(packet) > MAX_PACKET:
        raise ValueError("Invalid packet size")
    size = struct.unpack("!I", packet[:4])[0]
    if not 0 < size <= 8192 or size + 4 >= len(packet):
        raise ValueError("Invalid header length")
    return json.loads(packet[4:4 + size]), packet[4 + size:]


def main():
    # Import tensor runtimes on the main thread before starting connection threads.
    # Windows native modules can deadlock when first imported inside a WS handler.
    from engine import Session
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", action="append", default=[])
    args = parser.parse_args()
    token = secrets.token_urlsafe(32)
    client_lock = threading.Lock()
    origins = args.origin or ["http://localhost:1420", "http://127.0.0.1:1420", "http://tauri.localhost", "https://tauri.localhost", "tauri://localhost"]

    def handler(socket):
        session = None
        acquired = False
        try:
            hello = json.loads(socket.recv(timeout=10))
            if not hmac.compare_digest(str(hello.get("token", "")), token):
                socket.close(1008, "Unauthorized")
                return
            # A just-closed session may still be finishing its last native call.
            acquired = client_lock.acquire(timeout=5)
            if not acquired:
                socket.close(1013, "Camera worker already in use")
                return
            session = Session(hello.get("backend", "openvino"))
            session_id = str(hello["sessionId"])
            last_frame = -1
            socket.send(json.dumps({"type": "ready", "sessionId": session_id, "backend": session.backend,
                "model": "MobileNetV3-Large", "inputSize": 256, "device": "CPU", "coldStartMs": session.cold_start_ms}))
            for packet in socket:
                if isinstance(packet, str):
                    if json.loads(packet).get("type") == "reset":
                        session.reset()
                    continue
                meta, payload = unpack(packet)
                if meta["sessionId"] != session_id or int(meta["frameId"]) <= last_frame:
                    raise ValueError("Stale session or frame")
                last_frame = int(meta["frameId"])
                if meta.get("rotation") != 0 or meta.get("pixelFormat") not in {"jpeg", "png"}:
                    raise ValueError("Expected oriented JPEG/PNG")
                capture = meta.get("kind") == "capture"
                if capture and (meta["pixelFormat"] != "png" or not payload.startswith(b"\x89PNG")):
                    raise ValueError("Capture requires lossless PNG")
                image, result = session.run(payload, meta["style"], capture)
                socket.send(pack({**meta, **result}, image))
        except Exception as exc:
            try:
                socket.send(json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"}))
            except Exception:
                pass
        finally:
            if session is not None:
                session.close()
            if acquired:
                client_lock.release()

    with serve(handler, "127.0.0.1", 0, origins=[Origin(o) for o in origins], max_size=MAX_PACKET, max_queue=1, compression=None) as server:
        port = server.socket.getsockname()[1]
        print(json.dumps({"url": f"ws://127.0.0.1:{port}", "token": token}), flush=True)
        def lifetime():
            for _ in sys.stdin:
                pass
            server.shutdown()
        threading.Thread(target=lifetime, daemon=True).start()
        server.serve_forever()


if __name__ == "__main__":
    main()
