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

ROOT = Path(__file__).resolve().parents[1]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(SOURCES))
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = SOURCES[args.model]
    cache = ROOT / ".cache/upstream" / args.model / str(source["revision"])
    source_hashes: dict[str, str] = {}
    for relative in source["files"]:
        path = cache / Path(relative).name
        url = (
            f"https://raw.githubusercontent.com/{source['repository']}/"
            f"{source['revision']}/{relative}"
        )
        if args.fetch_source:
            with urlopen(url, timeout=30) as response:
                content = response.read()
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if not path.is_file():
            parser.error(f"Missing upstream source {path}; use --fetch-source")
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hashes[path.name] != SOURCE_SHA256[path.name]:
            parser.error(f"Upstream source SHA256 mismatch: {path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.model == "unet":
        load_module("model_utils", cache / "model_utils.py")
        module = load_module("official_unet", cache / "unet.py")
        net = module.unet()
        checkpoint = ROOT / "models/unet/model.pth"
        size, channels = 512, 19
    else:
        module = load_module("official_photo2cartoon", cache / "networks.py")
        net = module.ResnetGenerator(ngf=32, img_size=256, light=True)
        checkpoint = ROOT / "models/photo2cartoon/photo2cartoon_weights.pt"
        size, channels = 256, 3
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    expected = (
        manifest["unet"]["sha256"] if args.model == "unet"
        else manifest["photo2cartoon-bundle"]["generator_sha256"]
    )
    if digest != expected:
        parser.error("Checkpoint SHA256 does not match models.json")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if args.model == "photo2cartoon":
        state = state["genA2B"]
    net.load_state_dict(state, strict=True)
    net = net.to(args.device).eval()
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
        "model": args.model, "revision": source["revision"],
        "source_sha256": source_hashes, "checkpoint_sha256": digest,
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
