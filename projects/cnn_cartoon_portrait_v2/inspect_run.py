from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from cartoon_portrait_v2.cli import bool_value, serve_observer


def main() -> int:
    p = argparse.ArgumentParser(description="打开 CNN Cartoon v2 观察室")
    p.add_argument("--run", required=True)
    p.add_argument("--view", nargs="?", const=True, default=True, type=bool_value)
    p.add_argument("--port", type=int, default=0)
    args = p.parse_args()
    root = Path(args.run).resolve()
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"找不到观察室：{root / 'index.html'}")
    if args.view:
        serve_observer(root, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
