"""Export the single MobileNetV3 feature backbone for OpenVINO.

This is intentionally a separate command: it requires installing OpenVINO in
the d2l environment and records the output contract before mobile integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parent / "cnn_face_cartoon"
sys.path.insert(0, str(CORE / "src"))
from face_cartoon.features import Encoder  # noqa: E402


class FeatureWrapper(torch.nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.model = encoder.model

    def forward(self, image):  # ty: ignore[invalid-mutable-override] -- PyTorch defines forward as a callable attribute.
        return tuple(self.model(image))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=256, choices=[256, 384])
    args = parser.parse_args()
    args.output = args.output or ROOT / f"models/mobilenetv3_features_{args.size}"
    try:
        import openvino as ov
    except ImportError as exc:
        raise SystemExit("缺少 OpenVINO：python -m pip install openvino") from exc
    config = json.loads((CORE / "config.json").read_text(encoding="utf-8"))["features"]
    encoder = Encoder(CORE, "mobilenetv3", config, "cpu")
    model = FeatureWrapper(encoder).eval()
    example = torch.zeros(1, 3, args.size, args.size)
    ov_model = ov.convert_model(model, example_input=example)
    # The camera contract is batch=1 and a fixed preview/capture canvas. The
    # conversion API may keep spatial dimensions dynamic even with an example;
    # make the deployment shape explicit before saving the IR.
    ov_model.reshape({ov_model.inputs[0]: [1, 3, args.size, args.size]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, output in enumerate(ov_model.outputs):
        output.get_tensor().set_names({f"features_{i}"})
    ov.save_model(ov_model, args.output.with_suffix(".xml"), compress_to_fp16=False)
    manifest = {"model": "mobilenetv3_large_100.ra_in1k", "backend": "OpenVINO", "size": args.size,
                "input": {"layout": "NCHW", "color": "RGB", "range": "0..1", "mean": encoder.info["mean"], "std": encoder.info["std"]},
                "revision": encoder.info["revision"], "weight_storage": "FP32", "openvino_version": ov.__version__,
                "outputs": [{"index": i, "name": f"features_{i}", "stride": layer["reduction"], "channels": layer["num_chs"]} for i, layer in enumerate(encoder.layers)],
                "note": "仅转换三个特征输出；S、Q、几何和渲染仍由宿主实现。"}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    encoder.close()
    print(f"saved {args.output.with_suffix('.xml')}")


if __name__ == "__main__":
    main()
