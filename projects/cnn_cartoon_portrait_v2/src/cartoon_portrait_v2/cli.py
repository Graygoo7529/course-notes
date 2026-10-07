"""Command-line helpers kept separate so the library remains importable."""

from __future__ import annotations

import http.server
import threading
import webbrowser
from pathlib import Path


def bool_value(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def serve_observer(root: str | Path, port: int = 0, open_browser: bool = True) -> None:
    directory = str(Path(root).resolve())
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    print(f"观察室：{url}")
    if open_browser:
        threading.Timer(.2, lambda: webbrowser.open(url)).start()
    print("按 Ctrl+C 结束观察室。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
