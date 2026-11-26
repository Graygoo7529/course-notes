"""用固定上游结构和本地权重做合成输入前向；不等于真人效果验证。"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from urllib.request import urlopen

import torch
import numpy as np
from numpy.typing import NDArray

from .image_contracts import generator_input, generator_output

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "unet": {
        "revision": "3f6c45df5e67130568f9459f24f8d6a5ff836d30",
        "repository": "switchablenorms/CelebAMask-HQ",
        "files": ["face_parsing/model_utils.py", "face_parsing/unet.py"],
    },
    "photo2cartoon": {
        "revision": "6204e27a71620500c2f5b2ffd36646b0e2e5c110",
        "repository": "minivision-ai/photo2cartoon",
        "files": ["models/networks.py"],
    },
}
SOURCE_SHA256 = {
    "model_utils.py": "e68c70f1c2cd9e986cfa7c261a6983f7c80714f696c1770c6e3c996bafbbd60a",
    "unet.py": "05336219618032a6add7bac8d2073053ecb08f7be1345ebf327e00c957f8ba8f",
    "networks.py": "f58419e988519b2f4831e57330412cf738f4431d11daf4f7889b44f9e4f05623",
}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_source(model: str, root: Path, fetch: bool = False) -> tuple[Path, dict[str, str]]:
    source = SOURCES[model]
    cache = root / ".cache/upstream" / model / str(source["revision"])
    source_hashes: dict[str, str] = {}
    for relative in source["files"]:
        path = cache / Path(relative).name
        url = (
            f"https://raw.githubusercontent.com/{source['repository']}/"
            f"{source['revision']}/{relative}"
        )
        if fetch:
            with urlopen(url, timeout=30) as response:
                content = response.read()
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream source {path}; use --fetch-source")
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hashes[path.name] != SOURCE_SHA256[path.name]:
            raise ValueError(f"Upstream source SHA256 mismatch: {path}")
    return cache, source_hashes


def verify_weight(root: Path, name: str) -> tuple[Path, str]:
    manifest = json.loads((root / "models.json").read_text(encoding="utf-8"))
    if name == "photo2cartoon":
        path = root / "models/photo2cartoon/photo2cartoon_weights.pt"
        expected = manifest["photo2cartoon-bundle"]["generator_sha256"]
    else:
        path = root / "models" / manifest[name]["filename"]
        expected = manifest[name]["sha256"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected is None or digest != expected:
        raise ValueError(f"Checkpoint SHA256 not pinned or mismatched: {path}")
    return path, digest


def load_network(model: str, root: Path, device: str, fetch: bool = False):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache, source_hashes = prepare_source(model, root, fetch)
    if model == "unet":
        load_module("model_utils", cache / "model_utils.py")
        module = load_module("official_unet", cache / "unet.py")
        net = module.unet()
    else:
        module = load_module("official_photo2cartoon", cache / "networks.py")
        net = module.ResnetGenerator(ngf=32, img_size=256, light=True)
    checkpoint, digest = verify_weight(root, model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if model == "photo2cartoon":
        state = state["genA2B"]
    net.load_state_dict(state, strict=True)
    return net.to(device).eval(), {
        "revision": SOURCES[model]["revision"],
        "source_sha256": source_hashes, "checkpoint_sha256": digest,
    }


class PortraitModels:
    def __init__(self, root: Path, device: str, fetch_source: bool = False):
        self.device = device
        self.parser, parser_meta = load_network("unet", root, device, fetch_source)
        self.generator, generator_meta = load_network("photo2cartoon", root, device, fetch_source)
        self.metadata = {"unet": parser_meta, "photo2cartoon": generator_meta}

    def parse(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if rgb.shape != (512, 512, 3):
            raise ValueError("Parser expects a 512x512 RGB portrait")
        values = rgb.astype(np.float32) / np.float32(127.5) - 1
        tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)[None]))
        with torch.inference_mode():
            logits = self.parser(tensor.to(self.device))
            if logits.shape != (1, 19, 512, 512) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Invalid parser logits")
            return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def cartoon(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        with torch.inference_mode():
            out = self.generator(torch.from_numpy(generator_input(rgb)).to(self.device))[0][0]
        return generator_output(out.cpu().numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(SOURCES))
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    net, metadata = load_network(args.model, ROOT, args.device, args.fetch_source)
    size, channels = (512, 19) if args.model == "unet" else (256, 3)
    torch.manual_seed(0)
    image = (torch.rand(1, 3, size, size) * 2 - 1).to(args.device)
    with torch.inference_mode():
        result = net(image)
        if args.model == "photo2cartoon":
            result = result[0]
        if tuple(result.shape) != (1, channels, size, size) or not bool(torch.isfinite(result).all()):
            raise RuntimeError("Invalid model output")
        if args.device == "cuda":
            torch.cuda.synchronize()
    report = {
        "model": args.model, **metadata,
        "device": args.device, "torch": str(torch.__version__),
        "strict_load": True, "weights_only": True,
        "input": list(image.shape), "output": list(result.shape), "finite": True,
        "output_range": [float(result.min()), float(result.max())],
        "synthetic_input_only": True, "portrait_quality_verified": False,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
