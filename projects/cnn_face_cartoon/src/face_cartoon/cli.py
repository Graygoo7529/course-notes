"""Local batch execution with reusable preprocessing and compact CNN response caches."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from .artifacts import export_report, export_result, panel, save_image, write_json
from .contracts import Prepared, Structure, Transform
from .features import Encoder, extract, hash_file
from .preprocess import FaceGeometry, prepare, read_image, unpad
from .rendering import color_layer, compose, stylize

ROOT = Path(__file__).resolve().parents[2]


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def demo_inputs() -> tuple[list[Path], dict]:
    import matplotlib
    folder = ROOT / "data/input"
    folder.mkdir(parents=True, exist_ok=True)
    old = ROOT.parent / "cnn_cartoon_portrait_v2/outputs/selfie-v2/input.png"
    notes = {}
    paths = []
    if old.is_file():
        path = folder / "selfie.png"
        if not path.exists():
            save_image(path, read_image(old))
        paths.append(path)
        notes[path.stem] = {"label": "本地人像 · 正脸与眼镜", "source_note": "复用旧项目 150×197 本地照片；放大只用于画布统一，不恢复原图细节。"}
        dark = folder / "selfie_dim.png"
        if not dark.exists():
            save_image(dark, np.rint(read_image(path).astype(float) * 0.30).astype(np.uint8))
        notes[dark.stem] = {"label": "同图暗光派生 · 压力检查", "source_note": "由本地人像 RGB 乘 0.30 生成的暗光派生图；不是新增人物或真实暗光采集。"}
    grace = folder / "grace_hopper.jpg"
    if not grace.exists():
        source = Path(matplotlib.get_data_path()) / "sample_data/grace_hopper.jpg"
        grace.write_bytes(source.read_bytes())
    paths.append(grace)
    notes[grace.stem] = {"label": "公开示例 · 眼镜与复杂背景", "source_note": "Matplotlib 随包 grace_hopper.jpg 示例照片（512×600）；保留原背景，未做人物分割。"}
    if old.is_file():
        paths.append(folder / "selfie_dim.png")
    write_json(folder / "sources.json", notes)
    return paths, notes


def cache_prepared(folder: Path, item: Prepared, key: str) -> None:
    arrays: dict[str, Any] = {"original_rgb": item.original_rgb, "rgb": item.rgb,
                              "valid_mask": item.valid_mask, "edge": item.edge, "edge_direction": item.edge_direction,
                              **{f"mask_{k}": v for k, v in item.masks.items()},
                              **{f"group_{k}": v for k, v in item.face_groups.items()}}
    np.savez_compressed(folder / "prepared.npz", **arrays)
    write_json(folder / "prepared.json", {"signature": key, "transform": asdict(item.transform),
                                         "face_status": item.face_status, "diagnostics": item.diagnostics})


def load_prepared(folder: Path, key: str) -> Prepared | None:
    if not (folder / "prepared.json").exists() or not (folder / "prepared.npz").exists():
        return None
    meta = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    if meta["signature"] != key:
        return None
    with np.load(folder / "prepared.npz", allow_pickle=False) as arrays:
        return Prepared(arrays["original_rgb"], arrays["rgb"], arrays["valid_mask"], Transform(**meta["transform"]),
                        arrays["edge"], arrays["edge_direction"], {k[6:]: arrays[k] for k in arrays.files if k.startswith("group_")},
                        meta["face_status"], {k[5:]: arrays[k] for k in arrays.files if k.startswith("mask_")}, meta["diagnostics"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", help="One or more portrait files.")
    parser.add_argument("--demo", action="store_true", help="Use the local earlier portrait, Matplotlib sample and a dim derived image.")
    parser.add_argument("--models", nargs="+", choices=["resnet50", "senet", "efficientnetv2", "mobilenetv3"],
                        default=["resnet50", "senet", "efficientnetv2", "mobilenetv3"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--line-density", type=float)
    parser.add_argument("--palette", choices=["natural_warm", "pastel"])
    parser.add_argument("--no-landmarks", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Require existing matching prepared/CNN caches; do not load models.")
    args = parser.parse_args()
    if args.demo and args.input:
        parser.error("Use either --demo or --input")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.demo:
        inputs, notes = demo_inputs()
    else:
        inputs = args.input or sorted(p for p in (ROOT / "data/input").glob("*")
                                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})
        notes_path = ROOT / "data/input/sources.json"
        notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    if not inputs:
        parser.error("No images: use --input path/to/portrait.jpg or --demo")
    if len({p.stem for p in inputs}) != len(inputs):
        parser.error("Input file stems must be unique")
    if args.line_density is not None:
        config["style"]["line_density"] = args.line_density
    if not 0 <= config["style"]["line_density"] <= 1:
        parser.error("--line-density must be in [0,1]")
    if args.palette:
        config["style"]["palette"] = args.palette
    if config["style"].get("shadow_enabled"):
        parser.error("Optional extra shadows are not implemented; leave shadow_enabled=false")
    threads = 6
    torch.set_num_threads(threads)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use device=auto or cpu")
    device = "cuda" if torch.cuda.is_available() and config["device"] != "cpu" else "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.perf_counter()
    packages = ["torch", "torchvision", "timm", "opencv-contrib-python", "mediapipe", "numpy", "scipy", "Pillow"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    run = {"created_at": datetime.now().astimezone().isoformat(), "device": device, "threads": threads,
           "config": config, "models": {k: registry[k] for k in args.models}, "samples": [],
           "versions": versions, "model_load_seconds": {}, "torch_build": torch.__version__,
           "code_sha256": {p.name: hash_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
           "mode": "cached_render" if args.render_only else "inference_or_cache"}
    face: FaceGeometry | None = None
    prepared_items, colors, prepare_keys = {}, {}, {}
    try:
        for path in inputs:
            sample_id = path.stem
            folder = output / sample_id
            folder.mkdir(exist_ok=True)
            face_path = ROOT / config["face"]["model_path"]
            key = signature({"input": hash_file(path), "preprocess": config["preprocess"], "enabled": not args.no_landmarks,
                             "face": hash_file(face_path) if not args.no_landmarks and face_path.exists() else None,
                             "versions": {p: versions[p] for p in ["mediapipe", "Pillow", "opencv-contrib-python", "numpy"]},
                             "code": hash_file(Path(__file__).with_name("preprocess.py"))})
            prepare_keys[sample_id] = key
            start = time.perf_counter()
            prepared = load_prepared(folder, key)
            cached = prepared is not None
            if prepared is None:
                if args.render_only:
                    raise ValueError(f"No matching preprocessing cache for {sample_id}; run without --render-only")
                if face is None:
                    face = FaceGeometry(ROOT / config["face"]["model_path"], not args.no_landmarks)
                prepared = prepare(read_image(path), config["preprocess"], face)
                cache_prepared(folder, prepared, key)
            prep_seconds = time.perf_counter() - start
            prepared_items[sample_id] = prepared
            color_start = time.perf_counter()
            colors[sample_id] = color_layer(prepared.rgb, config["style"])
            color_seconds = time.perf_counter() - color_start
            save_image(folder / "input.png", unpad(prepared.original_rgb, prepared.transform))
            note = notes.get(sample_id, {"label": sample_id, "source_note": "本次提供的本地照片。"})
            run["samples"].append({"id": sample_id, **note, "source": str(path.resolve()), "sha256": hash_file(path),
                                   "face_status": prepared.face_status, "transform": asdict(prepared.transform),
                                   "preprocessing": prepared.diagnostics, "prepare_seconds": prep_seconds,
                                   "preprocessing_cached": cached, "color_seconds": color_seconds, "results": {}})
            print(f"prepared {sample_id}: {prepared.face_status}; cache={cached}", flush=True)
    finally:
        if face:
            face.close()
    for model_key in args.models:
        encoder = None
        try:
            for sample in run["samples"]:
                sample_id = sample["id"]
                prepared = prepared_items[sample_id]
                folder = output / sample_id / model_key
                folder.mkdir(exist_ok=True)
                key = signature({"prepare": prepare_keys[sample_id], "model": registry[model_key], "features": config["features"],
                                 "weights": hash_file(ROOT / "model" / model_key / "model.safetensors"),
                                 "code": hash_file(Path(__file__).with_name("features.py")), "timm": run["versions"]["timm"],
                                 "torch": torch.__version__})
                cached = False
                if (folder / "structure.json").exists() and (folder / "structure.npz").exists():
                    meta = json.loads((folder / "structure.json").read_text(encoding="utf-8"))
                    if meta["signature"] == key:
                        with np.load(folder / "structure.npz", allow_pickle=False) as arrays:
                            structure = Structure(tuple(arrays["responses"]), arrays["support"], arrays["stroke_score"], meta["model_info"], meta["layer_info"])
                        extract_seconds = meta["extract_seconds"]
                        cached = True
                if not cached:
                    if args.render_only:
                        raise ValueError(f"No matching CNN cache for {sample_id}/{model_key}")
                    if encoder is None:
                        start = time.perf_counter()
                        encoder = Encoder(ROOT, model_key, config["features"], device)
                        run["model_load_seconds"][model_key] = time.perf_counter() - start
                        encoder.forward(prepared.rgb)  # warm-up, not included in the reported forward time
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    structure = extract(prepared, encoder, config["features"])
                    if device == "cuda":
                        torch.cuda.synchronize()
                    extract_seconds = time.perf_counter() - start
                    np.savez_compressed(folder / "structure.npz", responses=np.stack(structure.responses),
                                        support=structure.support, stroke_score=structure.stroke_score)
                    write_json(folder / "structure.json", {"signature": key, "model_info": structure.model_info,
                                                          "layer_info": structure.layer_info, "extract_seconds": extract_seconds})
                start = time.perf_counter()
                layers = stylize(prepared, structure, config, colors[sample_id])
                result = compose(layers, prepared.transform, config)
                render_seconds = time.perf_counter() - start + sample["color_seconds"]
                metadata = {"sample_id": sample_id, "display_name": registry[model_key]["display_name"],
                            "model": structure.model_info, "layers": structure.layer_info,
                            "config": config, "diagnostics": layers.diagnostics, "face_status": prepared.face_status,
                            "extract_seconds": extract_seconds, "render_seconds": render_seconds, "cnn_cached": cached}
                export_result(folder, prepared, structure, result, metadata)
                write_json(folder / "paths.json", [p.tolist() for p in layers.paths])
                np.save(folder / "stroke_alpha.npy", layers.stroke_alpha)
                sample["results"][model_key] = {"extract_seconds": extract_seconds, "render_seconds": render_seconds,
                                                 "cnn_cached": cached, "paths": len(layers.paths), "diagnostics": layers.diagnostics}
                print(f"{sample_id}/{model_key}: CNN {extract_seconds:.3f}s; render {render_seconds:.3f}s; paths {len(layers.paths)}; cache={cached}", flush=True)
        finally:
            if encoder:
                encoder.close()
            gc.collect()
    first_model = args.models[0]
    for sample in run["samples"]:
        sample_id = sample["id"]
        prepared = prepared_items[sample_id]
        flat = np.ones_like(prepared.edge)
        baseline_structure = Structure((flat, flat, flat), flat, prepared.edge.copy(), {"key": "no_cnn"}, [])
        layers = stylize(prepared, baseline_structure, config, colors[sample_id])
        result = compose(layers, prepared.transform, config)
        folder = output / sample_id
        save_image(folder / "no_cnn.png", result.final_rgb)
        cnn = np.asarray(Image.open(folder / first_model / "final.png"))
        alpha = np.load(folder / first_model / "stroke_alpha.npy")
        sample["gate_check"] = {"reference_model": first_model,
                                 "changed_stroke_pixels": int(((np.abs(alpha - layers.stroke_alpha) > 0.05) & prepared.valid_mask).sum()),
                                 "cnn_stroke_pixels": int((alpha > 0.1).sum()),
                                 "no_cnn_stroke_pixels": int((layers.stroke_alpha > 0.1).sum()),
                                 "changed_output_pixels": int(np.any(cnn != result.final_rgb, axis=2).sum())}
        panel([unpad(prepared.original_rgb, prepared.transform), cnn, result.final_rgb],
              ["原图", registry[first_model]["display_name"] + " 门控", "关闭 CNN 门控 · Q=E"], 3,
              "CNN 是否真正影响了笔画？", folder / "baseline_comparison.png", "统一阈值参考与渲染器；未进行正式质量评估。")
    export_report(output, run)
    run["total_wall_seconds"] = time.perf_counter() - started
    run["wall_time_scope"] = "Includes prepare, weight load, warm-up, extraction, rendering and figure/HTML export; excludes Python imports."
    write_json(output / "run.json", run)
    print(f"Ready: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
