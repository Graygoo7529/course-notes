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
    feature_images = {
        "input": features.image,
        "guide_fine": features.guides[0],
        "gx_R_fine": features.gx[0, ..., 0], "gx_G_fine": features.gx[0, ..., 1],
        "gx_B_fine": features.gx[0, ..., 2],
        "response_R_fine": np.hypot(features.gx[0, ..., 0], features.gy[0, ..., 0]),
        "response_G_fine": np.hypot(features.gx[0, ..., 1], features.gy[0, ..., 1]),
        "response_B_fine": np.hypot(features.gx[0, ..., 2], features.gy[0, ..., 2]),
        "edge_fused": features.edge, "theta_fused": features.theta,
        "coherence_fused": features.coherence, "dark_candidate": features.dark,
        "lightness": features.lightness, "chroma_edge": features.chroma_edge,
        "color_barrier": np.maximum(features.right_barrier, features.down_barrier),
    }
    for index, sigma in enumerate(features.scales):
        feature_images[f"guide_sigma_{sigma:g}"] = features.guides[index]
        feature_images[f"edge_sigma_{sigma:g}"] = features.edges[index]
        feature_images[f"dog_signed_sigma_{sigma:g}"] = features.dog[index]
        feature_images[f"theta_sigma_{sigma:g}"] = features.thetas[index]
        feature_images[f"coherence_sigma_{sigma:g}"] = features.coherences[index]
    stages = {
        "features": feature_images,
        "lines": {
            "edge_fused": lines.edge, "dark_candidate": lines.dark, "nms_edge": lines.nms_edge,
            "outline": lines.outline, "internal_candidate": lines.internal,
            "nms_dark": lines.nms_dark, "score": lines.score, "source": lines.source,
            "retained": lines.retained, "skeleton": lines.skeleton, "bridged": lines.bridged,
        },
        "regions": {
            "lab_lightness": features.lightness, "chroma_edge": features.chroma_edge,
            "barrier": regions.barrier,
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
