"""Binary WebSocket response framing.

Response format: 4-byte big-endian JSON header length, UTF-8 JSON header, then
one JPEG scene image. Portrait cards are occasional base64 JPEGs in the header.
Camera frames sent by the browser are plain JPEG binary messages.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any


def pack_response(header: dict[str, Any], jpeg: bytes) -> bytes:
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded + jpeg


def image_to_base64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode("ascii")
