"""Frozen timm encoders loaded exclusively from verified local checkpoints."""
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import timm
import torch
from torch.nn import functional as F

from .contracts import Prepared, Structure
from .preprocess import normalize


def hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Encoder:
    def __init__(self, root: Path, key: str, config: dict, device: str):
        registry = json.loads((root / "models.json").read_text(encoding="utf-8"))[key]
        folder = root / "model" / key
        manifest = json.loads((folder / "download_manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            if hash_file(root / item["path"]) != item["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {item['path']}")
        local = f"local-dir:{folder.resolve()}"
        # Discover actual metadata before choosing output indices. Never assume index == stride.
        probe: Any = timm.create_model(local, pretrained=False, features_only=True)
        metadata = probe.feature_info.get_dicts()
        chosen = [min(range(len(metadata)), key=lambda i: abs(np.log2(metadata[i]["reduction"] / stride)))
                  for stride in config["target_strides"]]
        if len(set(chosen)) != 3:
            raise ValueError("Encoder does not expose three distinct requested feature scales")
        self.layers = [metadata[i] for i in chosen]
        del probe
        self.model: Any = timm.create_model(local, pretrained=True, features_only=True, out_indices=tuple(chosen))
        self.model.requires_grad_(False).to(device).eval()
        self.device = device
        self.mean = torch.tensor(self.model.pretrained_cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.model.pretrained_cfg["std"], device=device).view(1, 3, 1, 1)
        self.info = {**registry, "key": key, "device": device, "mean": self.mean.flatten().tolist(),
                     "std": self.std.flatten().tolist(), "weights": manifest, "training": False,
                     "selected_indices": chosen, "timm_version": timm.__version__}

    def forward(self, rgb: np.ndarray) -> list[torch.Tensor]:
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255
        with torch.inference_mode():
            return self.model((tensor - self.mean) / self.std)

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def feature_difference(feature: torch.Tensor, epsilon: float = 1e-6) -> np.ndarray:
    unit = F.normalize(feature.float(), p=2, dim=1, eps=epsilon)
    dx = torch.diff(unit, dim=3)
    dy = torch.diff(unit, dim=2)
    dx = F.pad(dx.square().sum(dim=1, keepdim=True), (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy.square().sum(dim=1, keepdim=True), (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(dx + dy)[0, 0].cpu().numpy()


def extract(prepared: Prepared, encoder: Encoder, config: dict) -> Structure:
    features = encoder.forward(prepared.rgb)
    h, w = prepared.edge.shape
    responses, layers = [], []
    for feature, layer in zip(features, encoder.layers):
        difference = feature_difference(feature, config["epsilon"])
        fh, fw = difference.shape
        mask = cv2.resize(prepared.valid_mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        response = normalize(difference, mask, config["normalization_quantiles"], config["epsilon"])
        response = cv2.resize(response, (w, h), interpolation=cv2.INTER_LINEAR)
        response[~prepared.valid_mask] = 0
        responses.append(response)
        layers.append({**layer, "shape": list(feature.shape), "valid_cells": int(mask.sum()),
                       "difference_range": [float(difference.min()), float(difference.max())]})
    support = sum(weight * field for weight, field in zip(config["weights"], responses)).astype(np.float32)
    floor = config["edge_floor"]
    score = prepared.edge * (floor + (1 - floor) * support)
    return Structure(tuple(responses), support, score.astype(np.float32), encoder.info, layers)
