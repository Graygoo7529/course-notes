"""Run the first-stage still-image scene and optional portrait-card pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.image_ops import read_rgb  # noqa: E402
from backend.app.pipeline import CorePipeline, draw_detections  # noqa: E402


def save_contact_sheet(output: Path, panels: list[tuple[str, Path]]) -> None:
    cell_width, cell_height = 440, 390
    sheet = Image.new("RGB", (cell_width * 2, cell_height * ((len(panels) + 1) // 2)), "#eef1f4")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(panels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width - 28, cell_height - 58), Image.Resampling.LANCZOS)
        x = (index % 2) * cell_width
        y = (index // 2) * cell_height
        draw.text((x + 14, y + 12), label, fill="#23313b")
        sheet.paste(image, (x + (cell_width - image.width) // 2, y + 42))
    sheet.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "input" / "grace_hopper.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "core_demo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-size", type=int, default=512)
    parser.add_argument("--cards", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    pipeline = CorePipeline.from_project(ROOT, device=args.device, scene_max_size=args.scene_size)
    model_load_seconds = time.perf_counter() - load_start
    process_start = time.perf_counter()
    result = pipeline.process(read_rgb(args.input), make_cards=args.cards)
    process_seconds = time.perf_counter() - process_start
    Image.fromarray(result.input_rgb).save(args.output / "input.png")
    Image.fromarray(draw_detections(result.input_rgb, result.detections)).save(args.output / "detections.png")
    Image.fromarray(result.sketch_rgb).save(args.output / "lineart_fine.png")
    Image.fromarray(result.scene_rgb).save(args.output / "scene_with_muted_hat.png")
    for index, card in enumerate(result.cards, 1):
        Image.fromarray(card).save(args.output / f"portrait_card_{index:02d}.png")
    panels = [
        ("input", args.output / "input.png"),
        ("YOLOv5-Face detections", args.output / "detections.png"),
        ("Lineart fine", args.output / "lineart_fine.png"),
        ("scene + muted-color hat", args.output / "scene_with_muted_hat.png"),
    ]
    if result.cards:
        panels.append(("portrait card 1", args.output / "portrait_card_01.png"))
    save_contact_sheet(args.output / "contact_sheet.png", panels)
    report = {
        "input": str(args.input),
        "scene_size": [int(result.scene_rgb.shape[1]), int(result.scene_rgb.shape[0])],
        "detected_faces": len(result.detections),
        "cards": len(result.cards),
        "model_load_seconds": round(model_load_seconds, 3),
        "process_seconds": round(process_seconds, 3),
        "detections": [
            {"bbox": list(d.bbox), "score": d.score, "landmarks5": [list(point) for point in d.landmarks5]}
            for d in result.detections
        ],
        "note": "First-stage still-image pipeline; temporal tracking and Web transport are stage two.",
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
