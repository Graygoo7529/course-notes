from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if not DEPS.exists():
    DEPS = ROOT.parent / "cnn_cartoon_portrait" / ".deps"
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cartoon_portrait_v2.artifacts import save_run
from cartoon_portrait_v2.cli import bool_value, serve_observer
from cartoon_portrait_v2.comic import V2Config, cartoonize
from cartoon_portrait_v2.features import FeatureConfig
from cartoon_portrait_v2.lines import LineConfig
from cartoon_portrait_v2.pipeline import read_mask, read_rgb
from cartoon_portrait_v2.regions import RegionConfig
from cartoon_portrait_v2.rendering import RenderConfig
from cartoon_portrait_v2.tone import ToneConfig


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CNN-inspired portrait cartoonizer v2")
    p.add_argument("--input", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output", default=str(ROOT / "outputs" / "run"))
    p.add_argument("--render-scale", type=int, default=4)
    p.add_argument("--scales", type=float, nargs="+", default=[.6, 1.2, 2.4],
                   help="多尺度高斯尺度，例如 --scales 0.6 1.2 2.4")
    p.add_argument("--sigma", type=float, default=None,
                   help="兼容旧命令：指定后使用 sigma、2sigma、4sigma")
    p.add_argument("--dark-sigma", type=float, default=None,
                   help="兼容旧命令：作为最大 DoG 尺度；优先级低于 --scales")
    p.add_argument("--operator", choices=["sobel", "scharr"], default="sobel")
    p.add_argument("--palette-size", type=int, default=5)
    p.add_argument("--palette-style", choices=["natural", "warm", "pastel", "noir"], default="natural")
    p.add_argument("--line-width", type=float, default=1.35)
    p.add_argument("--dark-width", type=float, default=1.02)
    p.add_argument("--outline-width", type=float, default=1.30)
    p.add_argument("--min-coherence", type=float, default=.16)
    p.add_argument("--line-low", type=float, default=.16)
    p.add_argument("--line-high", type=float, default=.34)
    p.add_argument("--join-gap", type=int, default=3)
    p.add_argument("--shadow-depth", type=float, default=.18)
    p.add_argument("--shadow-fraction", type=float, default=.28)
    p.add_argument("--view", nargs="?", const=True, default=False, type=bool_value)
    p.add_argument("--port", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    image = read_rgb(args.input)
    mask = read_mask(args.mask, image.shape[:2])
    scales = tuple(args.scales)
    if args.sigma is not None:
        scales = (args.sigma, 2 * args.sigma, 4 * args.sigma)
    feature = FeatureConfig(scales=scales, operator=args.operator)
    if args.dark_sigma is not None and args.sigma is not None:
        feature = FeatureConfig(scales=scales, operator=args.operator,
                                dog_ratio=max(1.01, args.dark_sigma / args.sigma))
    config = V2Config(
        feature=feature,
        line=LineConfig(low=args.line_low, high=args.line_high, join_gap=args.join_gap,
                        width=args.line_width, dark_width=args.dark_width,
                        outline_width=args.outline_width, min_coherence=args.min_coherence),
        region=RegionConfig(palette_size=args.palette_size, palette_style=args.palette_style),
        tone=ToneConfig(args.shadow_fraction, args.shadow_depth),
        render=RenderConfig(args.render_scale, args.line_width),
    )
    result = cartoonize(image, mask, config)
    root = save_run(result, args.output, config, str(Path(args.input).resolve()), str(Path(args.mask).resolve()))
    print(f"已保存 v2 结果：{root / 'cartoon.png'}")
    print(f"五段观察室：{root / 'index.html'}")
    if args.view:
        serve_observer(root, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
