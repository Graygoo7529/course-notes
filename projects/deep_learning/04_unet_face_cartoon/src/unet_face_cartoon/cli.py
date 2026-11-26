import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import torch

from .artifacts import write_report
from .model_runtime import ROOT
from .pipeline import Controls, Pipeline


def color(value: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise argparse.ArgumentTypeError("发色格式为 '#RRGGBB'")
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def main() -> int:
    parser = argparse.ArgumentParser(description="UNet 分区引导的单人照片漫画；默认本地离线推理")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", nargs="+", type=Path)
    group.add_argument("--demo", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skin-strength", type=float, default=0.5)
    parser.add_argument("--hair-color", type=color)
    parser.add_argument("--hair-strength", type=float, default=0)
    parser.add_argument("--fetch-source", action="store_true")
    args = parser.parse_args()
    try:
        controls = Controls(args.skin_strength, args.hair_color, args.hair_strength)
    except ValueError as exc:
        parser.error(str(exc))
    samples = json.loads((ROOT / "samples.json").read_text(encoding="utf-8"))
    sources = {entry["id"]: entry for entry in samples}
    if args.demo:
        paths = [ROOT / "data/public" / f"{entry['id']}.jpg" for entry in samples]
    else:
        paths = args.input
    if any(not path.is_file() for path in paths):
        parser.error("输入文件不存在；演示图片先运行 scripts/download_samples.py")
    # 安全且唯一的输出子目录名，与用户的原始文件名分离。
    ids = [f"{i:02d}-{re.sub(r'[^a-zA-Z0-9_-]', '_', path.stem)}" for i, path in enumerate(paths, 1)]
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录非空；请使用新目录，避免混入旧结果。")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading models on {device}...", flush=True)
    try:
        pipeline = Pipeline(ROOT, device, args.fetch_source)
    except Exception as exc:
        print(f"模型准备失败：{exc}", file=sys.stderr)
        return 2
    records = []
    for identifier, path in zip(ids, paths, strict=True):
        print(f"Processing {path.name}...", flush=True)
        try:
            source = sources.get(path.stem) if path.resolve().parent == (ROOT / "data/public").resolve() else None
            record = pipeline.process(path, args.output / identifier, controls, source)
        except Exception as exc:
            record = {"id": identifier, "status": "error", "input_name": path.name, "error": str(exc)}
            print(f"{path.name}: {exc}", file=sys.stderr)
        records.append(record)
        write_report(args.output, records, pipeline.metadata)
    print(f"Report: {(args.output / 'index.html').resolve()}")
    return int(any(record["status"] != "ok" for record in records))
