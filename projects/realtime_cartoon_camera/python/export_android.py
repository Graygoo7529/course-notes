"""Export the same frozen backbone and shared configuration for Android."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch

from export_openvino import CORE, ROOT, FeatureWrapper
sys.path.insert(0, str(CORE / "src"))
from face_cartoon.features import Encoder
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections


def main():
    destination = ROOT / "models/android"
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((CORE / "config.json").read_text(encoding="utf-8"))
    encoder = Encoder(CORE, "mobilenetv3", config["features"], "cpu")
    wrapper = FeatureWrapper(encoder).eval()
    path = destination / "mobilenetv3_features.onnx"
    torch.onnx.export(wrapper, (torch.zeros(1, 3, 256, 256),), str(path), input_names=["image"],
        output_names=["features_0", "features_1", "features_2"], opset_version=17, dynamo=False,
        dynamic_axes={"image": {2: "height", 3: "width"}, **{f"features_{i}": {2: f"h{i}", 3: f"w{i}"} for i in range(3)}})
    config["model"] = {"file": path.name, "mean": encoder.info["mean"], "std": encoder.info["std"],
        "revision": encoder.info["revision"], "name": "MobileNetV3-Large", "outputs": [dict(x) for x in encoder.layers]}
    config["geometry_connections"] = {key: [[e.start, e.end] for e in getattr(FaceLandmarksConnections, f"FACE_LANDMARKS_{key}")]
        for key in ["LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW", "RIGHT_EYEBROW", "LIPS", "FACE_OVAL"]}
    (destination / "cartoon_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(CORE / "model/face_landmarker.task", destination / "face_landmarker.task")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    report = []
    for size in [256, 384]:
        tensor = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(42))
        with torch.inference_mode(): expected = wrapper(tensor)
        actual = session.run(None, {"image": tensor.numpy()})
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert isinstance(a, np.ndarray)
            np.testing.assert_allclose(a, e.numpy(), rtol=2e-3, atol=2e-4)
            report.append({"size": size, "output": i, "shape": list(a.shape), "max_abs": float(np.abs(a - e.numpy()).max())})
    (destination / "export_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    encoder.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
