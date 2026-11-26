"""全分辨率固定卷积：多尺度 DW、非线性结构特征与固定 PW。"""

from dataclasses import dataclass

import cv2
import numpy as np

from .features import FeatureConfig, FusionConfig, Structure, channel_gradients, extract_channel_features, fuse_features
from .pipeline import FloatImage, Mask, _mask, _number, _unit_array, masked_gaussian


@dataclass(frozen=True)
class BankConfig:
    scales: tuple[float, ...] = (0.6, 1.2, 2.4)
    direction_sigma: float = 1.0
    dog_ratio: float = 2.0
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    operator: str = "sobel"

    def validate(self) -> None:
        if not 1 <= len(self.scales) <= 6:
            raise ValueError("scales 需要 1–6 个递增尺度。")
        for sigma in self.scales:
            _number(sigma, "scale", 0.2, 20)
        if any(a >= b for a, b in zip(self.scales, self.scales[1:])):
            raise ValueError("scales 必须严格递增。")
        _number(self.direction_sigma, "direction_sigma", 0, 10)
        _number(self.dog_ratio, "dog_ratio", 1.01, 4)
        FusionConfig(self.weights).validate()
        if self.operator not in ("sobel", "scharr"):
            raise ValueError("operator 必须为 sobel 或 scharr。")


@dataclass(frozen=True)
class FeatureBank:
    scales: tuple[float, ...]
    guides: FloatImage             # K,H,W,3
    gx: FloatImage                 # K,H,W,3；有符号，未乘尺度
    gy: FloatImage
    edges: FloatImage              # K,H,W；sigma * sqrt(lambda_max)
    dog: FloatImage                # K,H,W；有符号，宽邻域减窄邻域
    theta: FloatImage              # K,H,W；邻域法向，模 pi
    coherence: FloatImage
    direction_valid: Mask          # K,H,W
    lab: FloatImage                # 原始 Lab，非归一化 a*, b*
    lightness: FloatImage           # L*/100
    chroma_edge: FloatImage         # 原生 Lab / 原图像素
    fine_structure: Structure


def extract_feature_bank(image: FloatImage, mask: Mask, config: BankConfig = BankConfig()) -> FeatureBank:
    config.validate()
    _unit_array(image, "image", 3)
    _mask(mask, image.shape)
    features, structures = [], []
    # FeatureConfig 的 Literal 字段通过两个明确分支构造。
    operator = "sobel" if config.operator == "sobel" else "scharr"
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float32)
    lightness = lab[..., 0] / np.float32(100)
    dogs, angles, reliabilities, validities = [], [], [], []
    for sigma in config.scales:
        feature = extract_channel_features(image, mask, FeatureConfig(sigma, operator))
        structure = fuse_features(feature, FusionConfig(config.weights))
        features.append(feature)
        structures.append(structure)
        tensor = np.stack((structure.jxx, structure.jxy, structure.jyy), axis=-1)
        tensor = masked_gaussian(tensor, mask, config.direction_sigma)
        xx, xy, yy = np.moveaxis(tensor, -1, 0)
        trace, gap = xx + yy, np.hypot(xx - yy, 2 * xy)
        q = np.divide(gap, trace, out=np.zeros_like(trace), where=trace > 1e-12)
        valid = mask & (trace > 1e-12) & (q > 1e-4)
        angles.append(np.where(valid, 0.5 * np.arctan2(2 * xy, xx - yy), 0).astype(np.float32))
        reliabilities.append(np.where(mask, np.clip(q, 0, 1), 0).astype(np.float32))
        validities.append(valid)
        dogs.append((masked_gaussian(lightness, mask, sigma * config.dog_ratio) -
                     masked_gaussian(lightness, mask, sigma)).astype(np.float32))
    chroma = masked_gaussian(lab[..., 1:], mask, config.scales[0])
    ax, ay = channel_gradients(chroma, mask, config.operator)
    return FeatureBank(
        config.scales, np.stack([f.guide for f in features]),
        np.stack([f.gx for f in features]), np.stack([f.gy for f in features]),
        np.stack([s * f.strength for s, f in zip(config.scales, structures)]).astype(np.float32),
        np.stack(dogs), np.stack(angles), np.stack(reliabilities), np.stack(validities),
        lab, lightness, np.sqrt(np.sum(ax * ax + ay * ay, axis=-1)).astype(np.float32), structures[0],
    )
