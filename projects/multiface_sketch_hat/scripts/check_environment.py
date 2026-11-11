"""Inspect dependencies and local model files without downloading or loading weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "torch", "torchvision", "numpy", "opencv-contrib-python", "Pillow", "scipy",
    "fastapi", "uvicorn", "websockets", "PyYAML", "requests", "tqdm",
    "matplotlib", "pandas", "seaborn", "scikit-image", "gdown", "ty",
]


def command_output(program: str, *arguments: str) -> str:
    executable = shutil.which(program)
    if not executable:
        return "MISSING"
    try:
        result = subprocess.run(
            [executable, *arguments], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20, check=False,
        )
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"ERROR: {error}"


def describe_weight(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        prefix = source.read(256).lstrip().lower()
        source.seek(0)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "looks_like_error_page": b"<!doctype html" in prefix or b"<html" in prefix,
        "inference_verified": False,
    }


def inspect() -> dict[str, Any]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    runtime: dict[str, Any] = {}
    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        torch_version = importlib.import_module("torch.version")
        available = torch.cuda.is_available()
        device = "cuda" if available else "cpu"
        runtime.update(torch=torch.__version__, torchvision=torchvision.__version__,
                       cuda_build=torch_version.cuda, cuda_available=available, device=device)
        if available:
            runtime["gpu"] = torch.cuda.get_device_name(0)
        # This exercises the selected device and torchvision's compiled NMS operator.
        layer = torch.nn.Conv2d(3, 4, 3).to(device)
        with torch.inference_mode():
            result = layer(torch.zeros((1, 3, 16, 16), device=device))
        runtime["conv_shape"] = list(result.shape)
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        scores = torch.tensor([0.9, 0.8], device=device)
        runtime["nms_indices"] = torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist()
    except Exception as error:
        runtime["error"] = f"{type(error).__name__}: {error}"
    manifest = json.loads((ROOT / "config/models.json").read_text(encoding="utf-8"))
    models = []
    for entry in manifest["models"]:
        if "file_glob" in entry:
            paths = sorted(ROOT.glob(entry["file_glob"]))
        elif "file" in entry:
            paths = [ROOT / entry["file"]]
        else:
            paths = [ROOT / entry["primary_file"]]
        models.append({
            "id": entry["id"],
            "source_required": entry.get("source_required", True),
            "source_present": (ROOT / entry["source_dir"]).is_dir(),
            "files": [describe_weight(path) for path in paths if path.is_file()],
        })
    return {
        "python": sys.version, "executable": sys.executable, "packages": packages,
        "runtime": runtime, "models": models,
        "node": command_output("node", "--version"),
        "gpu_driver": command_output("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "note": "File presence and local hashes do not verify official provenance or model inference.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report; use an ignored directory.")
    args = parser.parse_args()
    report = inspect()
    content = json.dumps(report, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    incomplete = "MISSING" in report["packages"].values() or "error" in report["runtime"]
    if args.require_models:
        incomplete |= any(
            (entry.get("source_required", True) and not entry["source_present"]) or not entry["files"]
            or any(item["bytes"] == 0 or item["looks_like_error_page"] for item in entry["files"])
            for entry in report["models"]
        )
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
