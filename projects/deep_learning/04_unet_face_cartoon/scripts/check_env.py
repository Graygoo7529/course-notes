"""实际检查 RGB、CUDA 和 torchvision，不下载权重。"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import cv2
import face_alignment
import numpy as np
from PIL import Image
import torch
import torchvision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "pillow", "opencv-python",
                         "face-alignment", "gdown", "ty")
        },
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "landmarks_enum": face_alignment.LandmarksType.TWO_D.name,
    }
    failures: list[str] = []
    try:
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("OpenCV PNG encoding failed")
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("OpenCV PNG decoding failed")
        decoded = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded, np.array(Image.fromarray(rgb)))
        report["rgb_png_roundtrip"] = "passed"
    except Exception as exc:
        failures.append(f"image check: {exc}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        failures.append("CUDA is required but unavailable")
    try:
        if device.type == "cuda":
            report["gpu"] = torch.cuda.get_device_name(0)
            report["capability"] = torch.cuda.get_device_capability(0)
            report["compiled_architectures"] = torch.cuda.get_arch_list()
        torch.manual_seed(0)
        with torch.inference_mode():
            net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.InstanceNorm2d(8), torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(8, 3, 2, stride=2),
            ).to(device).eval()
            for size in (256, 512):
                result = net(torch.rand(1, 3, size, size, device=device))
                if result.shape != (1, 3, size * 2, size * 2):
                    raise RuntimeError("unexpected convolution output shape")
                if not bool(torch.isfinite(result).all()):
                    raise RuntimeError("nonfinite convolution output")
            boxes = torch.tensor([[0, 0, 8, 8], [1, 1, 7, 7]], device=device).float()
            kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
            if kept.tolist() != [0]:
                raise RuntimeError("torchvision NMS check failed")
            if device.type == "cuda":
                torch.cuda.synchronize()
        report["tensor_checks"] = f"passed on {device}"
    except Exception as exc:
        failures.append(f"tensor check: {exc}")
    report["failures"] = failures
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
