"""多尺度、三个任务分支与重绘的纯函数入口。"""

from dataclasses import dataclass, field

import numpy as np

from .feature_bank import BankConfig, FeatureBank, extract_feature_bank
from .pipeline import FloatImage, Mask, _mask, _unit_array
from .regions import FillPlan, RegionConfig, RegionScene, design_fills, fuse_region_features, organize_regions
from .rendering import RenderConfig, RenderResult, render_cartoon
from .strokes import LineFeatures, StrokeConfig, StrokeSet, design_strokes, fuse_line_features


@dataclass(frozen=True)
class ComicConfig:
    features: BankConfig = field(default_factory=BankConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    strokes: StrokeConfig = field(default_factory=StrokeConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def validate(self) -> None:
        self.features.validate()
        self.regions.validate()
        self.strokes.validate()
        self.render.validate()


@dataclass(frozen=True)
class ComicResult:
    features: FeatureBank
    line_features: LineFeatures
    scene: RegionScene
    strokes: StrokeSet
    fills: FillPlan
    rendered: RenderResult


def cartoonize(image: FloatImage, mask: Mask, config: ComicConfig = ComicConfig(), *,
               protect: Mask | None = None, suppress: Mask | None = None) -> ComicResult:
    """不修改输入、不读取文件；全部具名中间结果都交给外层决定如何展示。"""
    config.validate()
    _unit_array(image, "image", 3)
    _mask(mask, image.shape)
    if mask.size * (config.render.scale * config.render.supersample) ** 2 > config.render.max_pixels:
        raise ValueError("绘制画布超出像素上限，请减小 --render-scale 或 --supersample。")
    protect = np.zeros_like(mask) if protect is None else protect
    suppress = np.zeros_like(mask) if suppress is None else suppress
    bank = extract_feature_bank(image, mask, config.features)
    line_features = fuse_line_features(bank, mask, protect, suppress, config.strokes)
    region_features = fuse_region_features(bank, mask, config.regions)
    scene = organize_regions(region_features, mask, protect, config.regions)
    strokes = design_strokes(line_features, mask, config.strokes)
    fills = design_fills(bank, scene, mask, protect, config.regions)
    rendered = render_cartoon(strokes, fills, mask, config.render)
    return ComicResult(bank, line_features, scene, strokes, fills, rendered)
