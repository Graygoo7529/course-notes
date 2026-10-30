"""Prepare a portable native fixture and compare returned native tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from engine import APP, CORE, OpenVINOEncoder, FaceGeometry, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "compare"])
    parser.add_argument("--size", type=int, choices=[256, 384], default=256)
    parser.add_argument("--input", type=Path, default=CORE / "data/input/selfie.png")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    directory = args.directory or APP / f".cache/native_probe_{args.size}"
    if args.action == "prepare":
        directory.mkdir(parents=True, exist_ok=True)
        with Image.open(args.input) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB")).copy()
        config = json.loads((CORE / "config.json").read_text(encoding="utf-8"))["preprocess"]
        config["canvas_size"] = args.size
        geometry = FaceGeometry(CORE / "model/face_landmarker.task", enabled=False)
        prepared = prepare(rgb, config, geometry)
        geometry.close()
        encoder = OpenVINOEncoder(args.size)
        tensor = prepared.rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255
        ((tensor - encoder.mean) / encoder.std).astype("<f4").tofile(directory / "input.f32")
        features = encoder.forward(prepared.rgb)
        for i, feature in enumerate(features):
            np.save(directory / f"expected.{i}.npy", feature.numpy())
        encoder.close()
        print(f"Prepared normalized RGB fixture: {directory}")
    else:
        report = []
        for i in range(3):
            expected = np.load(directory / f"expected.{i}.npy")
            actual = np.fromfile(directory / f"actual.{i}.f32", dtype="<f4").reshape(expected.shape)
            difference = np.abs(actual - expected)
            report.append({"output": i, "shape": list(expected.shape), "max_abs": float(difference.max()),
                           "mean_abs": float(difference.mean())})
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        text = json.dumps(report, indent=2)
        (directory / "comparison.json").write_text(text + "\n", encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
