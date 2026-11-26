"""A small, device-aware adapter for Informative Drawings Lineart weights."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn

from ..image_ops import validate_rgb


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return x + self.conv_block(x)


class LineartGenerator(nn.Module):
    """The three-residual-block generator used by the downloaded weights."""

    def __init__(self, residual_blocks: int = 3) -> None:
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
        )
        self.model1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(*(ResidualBlock(256) for _ in range(residual_blocks)))
        self.model3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))


STYLE_NAMES = {
    "anime_style": "anime_style/netG_A_latest.pth",
    "opensketch_style": "opensketch_style/netG_A_latest.pth",
    "contour_style": "contour_style/netG_A_latest.pth",
}


def resolve_style_weight(model_root: str | Path, style: str = "anime_style") -> Path:
    root = Path(model_root)
    if style not in STYLE_NAMES:
        raise ValueError(f"Unknown Lineart style: {style}")
    relative = STYLE_NAMES[style]
    candidates = [root / relative, root / "model" / relative]
    for extracted in candidates:
        if extracted.is_file():
            return extracted
    # The old project convention stored the two primary files under these names.
    aliases = {"anime_style": "sk_model.pth", "opensketch_style": "sk_model2.pth"}
    alias = root / aliases.get(style, "__missing__.pth")
    if alias.is_file():
        return alias
    raise FileNotFoundError(
        f"Missing {candidates[0]}. Extract model/informative_drawings/model.zip first."
    )


class LineartSketcher:
    """Convert RGB uint8 images to white-paper grayscale line drawings."""

    def __init__(
        self,
        model_root: str | Path,
        style: str = "anime_style",
        device: str = "auto",
        residual_blocks: int = 3,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.style = style
        path = resolve_style_weight(model_root, style)
        self.net = LineartGenerator(residual_blocks).to(self.device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict in {path}")
        self.net.load_state_dict(state, strict=True)
        self.net.eval()

    @staticmethod
    def _prepare(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        image = validate_rgb(image)
        h, w = image.shape[:2]
        padded = cv2.copyMakeBorder(image, 0, (-h) % 4, 0, (-w) % 4, cv2.BORDER_REFLECT_101)
        return padded, (h, w)

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> np.ndarray:
        padded, original_shape = self._prepare(image)
        tensor = torch.from_numpy(padded.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.float().to(self.device) / 255.0
        paper = self.net(tensor)[0, 0].detach().cpu().numpy()
        h, w = original_shape
        return np.round(np.clip(paper[:h, :w], 0.0, 1.0) * 255.0).astype(np.uint8)

    def predict_batch(self, images: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Run same-size images together; useful for fixed-size person cards."""

        items = [validate_rgb(image) for image in images]
        if not items:
            return []
        shapes = {image.shape for image in items}
        if len(shapes) != 1:
            raise ValueError("predict_batch requires equal image shapes")
        padded = [self._prepare(image)[0] for image in items]
        tensor = torch.from_numpy(np.stack(padded)).permute(0, 3, 1, 2)
        tensor = tensor.float().to(self.device) / 255.0
        paper = self.net(tensor)[:, 0].detach().cpu().numpy()
        return [np.round(np.clip(item[: image.shape[0], : image.shape[1]], 0.0, 1.0) * 255.0).astype(np.uint8)
                for item, image in zip(paper, items)]
