"""无需预训练模型的自拍人像漫画化管道。"""

from .colors import ColorConfig, edge_aware_smooth, make_colors, map_lightness
from .features import FeatureConfig, FusionConfig, extract_channel_features, fuse_features
from .lines import LineConfig, make_lines
from .selection import ViewConfig, collect_selection, make_preview, view_to_image
from .workflow import StyleConfig, StyleResult, stylize
from .comic import ComicConfig, ComicResult, cartoonize
from .feature_bank import BankConfig, extract_feature_bank
from .regions import RegionConfig, design_fills, organize_regions
from .rendering import RenderConfig, render_cartoon
from .strokes import StrokeConfig, design_strokes

from .pipeline import (
    Config,
    Selection,
    compose_rgba,
    extract_lines,
    quantize_colors,
    read_image,
    save_png,
    segment_person,
    smooth_foreground,
)

__all__ = [
    "Config", "Selection", "read_image", "segment_person",
    "smooth_foreground", "quantize_colors", "extract_lines",
    "compose_rgba", "save_png",
    "ColorConfig", "FeatureConfig", "FusionConfig", "LineConfig", "StyleConfig", "StyleResult",
    "ViewConfig", "collect_selection", "make_preview", "view_to_image",
    "extract_channel_features", "fuse_features", "make_colors", "make_lines",
    "edge_aware_smooth", "map_lightness", "stylize",
    "ComicConfig", "ComicResult", "BankConfig", "RegionConfig", "StrokeConfig", "RenderConfig",
    "cartoonize", "extract_feature_bank", "organize_regions", "design_strokes", "design_fills", "render_cartoon",
]
