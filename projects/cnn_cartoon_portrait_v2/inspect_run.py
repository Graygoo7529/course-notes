from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if not DEPS.exists():
    DEPS = ROOT.parent / "cnn_cartoon_portrait" / ".deps"
sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))
from cartoon_portrait_v2.artifacts import export_run_bundle
from cartoon_portrait_v2.cli import bool_value, serve_observer


def main() -> int:
    p = argparse.ArgumentParser(description="打开 CNN Cartoon v2 观察室")
    p.add_argument("--run", required=True)
    p.add_argument("--view", nargs="?", const=True, default=True, type=bool_value)
    p.add_argument("--export", nargs="?", const=True, default=False, type=bool_value,
                   help="生成可传输的 observer_bundle.zip")
    p.add_argument("--port", type=int, default=0)
    args = p.parse_args()
    root = Path(args.run).resolve()
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"找不到观察室：{root / 'index.html'}")
    if args.export:
        print(f"观察室压缩包：{export_run_bundle(root)}")
    if args.view:
        serve_observer(root, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
