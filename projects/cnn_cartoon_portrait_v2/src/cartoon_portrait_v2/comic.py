"""Five-stage public API for the v2 portrait cartoonizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FeatureBundle, FeatureConfig, extract_features
from .lines import LineConfig, LinePlan, generate_lines
from .regions import RegionConfig, RegionPlan, generate_regions
from .rendering import RenderConfig, RenderResult, compose_image
from .tone import ToneConfig, TonePlan, generate_tone


@dataclass(frozen=True)
class V2Config:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    line: LineConfig = field(default_factory=LineConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    tone: ToneConfig = field(default_factory=ToneConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


@dataclass
class CartoonResult:
    features: FeatureBundle
    lines: LinePlan
    regions: RegionPlan
    tone: TonePlan
    rendered: RenderResult
    stages: dict[str, dict[str, np.ndarray]]


def cartoonize(image: np.ndarray, mask: np.ndarray, config: V2Config | None = None) -> CartoonResult:
    cfg = config or V2Config()
    features = extract_features(image, mask, cfg.feature)
    lines = generate_lines(features, features.mask, cfg.line)
    regions = generate_regions(features, features.mask, cfg.region)
    tone = generate_tone(features, regions, features.mask, cfg.tone)
    rendered = compose_image(lines, regions, tone, features.mask, cfg.render)
    stages = {
        "features": {
            "input": features.image, "guide": features.guide_rgb,
            "gx_R": features.gx[..., 0], "gx_G": features.gx[..., 1], "gx_B": features.gx[..., 2],
            "edge": features.edge, "theta": features.theta, "coherence": features.coherence,
            "dark": features.dark, "color_barrier": np.maximum(features.right_barrier, features.down_barrier),
        },
        "lines": {
            "edge_raw": lines.edge, "dark_raw": lines.dark, "nms_edge": lines.nms_edge,
            "nms_dark": lines.nms_dark, "score": lines.score, "retained": lines.retained,
            "skeleton": lines.skeleton, "bridged": lines.bridged,
        },
        "regions": {
            "lab_lightness": features.lightness, "barrier": regions.barrier,
            "initial_labels": regions.initial_labels, "labels": regions.labels, "flat": regions.flat,
        },
        "tone": {
            "lightness": features.lightness, "base": tone.base, "detail": tone.detail,
            "shadow_candidate": tone.shadow_candidate, "shadow": tone.shadow, "colors": tone.flat,
        },
        "compose": {
            "regions": rendered.regions, "strokes": rendered.strokes, "cartoon": rendered.rgb,
        },
    }
    return CartoonResult(features, lines, regions, tone, rendered, stages)
