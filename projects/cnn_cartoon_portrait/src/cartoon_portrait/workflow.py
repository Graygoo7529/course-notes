"""函数管道：只编排数组计算，不读写文件或打开窗口。"""

from dataclasses import dataclass, field

from .colors import ColorConfig, ColorDebug, make_colors
from .features import (
    ChannelFeatures, FeatureConfig, FusionConfig, Structure,
    comparison_strengths, extract_channel_features, fuse_features,
)
from .lines import LineConfig, LineDebug, make_lines
from .pipeline import FloatImage, Mask, _number, compose_rgba


@dataclass(frozen=True)
class StyleConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    colors: ColorConfig = field(default_factory=ColorConfig)
    lines: LineConfig = field(default_factory=LineConfig)
    strength: float = 0.55

    def validate(self) -> None:
        self.features.validate()
        self.fusion.validate()
        self.colors.validate()
        self.lines.validate()
        _number(self.strength, "strength", 0, 1)


@dataclass(frozen=True)
class StyleResult:
    features: ChannelFeatures
    structure: Structure
    colors: FloatImage
    lines: FloatImage
    rgba: FloatImage
    color_debug: ColorDebug
    line_debug: LineDebug
    comparisons: dict[str, FloatImage]


def stylize(image: FloatImage, mask: Mask, config: StyleConfig = StyleConfig()) -> StyleResult:
    """原图/遮罩 → 逐通道特征 → 融合 → 色块/线条 → 非预乘 RGBA。"""
    config.validate()
    features = extract_channel_features(image, mask, config.features)
    structure = fuse_features(features, config.fusion)
    colors, color_debug = make_colors(image, mask, structure, config.colors)
    lines, line_debug = make_lines(structure, mask, config.lines)
    rgba = compose_rgba(colors, lines, mask.astype("float32"), config.strength)
    comparisons = comparison_strengths(features, mask, config.features, config.fusion)
    return StyleResult(features, structure, colors, lines, rgba, color_debug, line_debug, comparisons)
