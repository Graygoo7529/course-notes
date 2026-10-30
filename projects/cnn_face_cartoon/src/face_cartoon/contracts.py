"""Four-stage data contracts. These describe interfaces, not generated results.

RGB arrays: uint8 H×W×3. Responses/alpha: float32 H×W in [0, 1].
Landmark/path coordinates: float32 N×2 in canvas (x, y) order.
"""
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

RGB = NDArray[np.uint8]
Field = NDArray[np.float32]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class Transform:
    original_size: tuple[int, int]  # width, height after EXIF orientation
    resized_size: tuple[int, int]
    canvas_size: tuple[int, int]
    scale_xy: tuple[float, float]  # actual rounded resize factors
    padding_xy: tuple[int, int]   # left, top


@dataclass(frozen=True)
class Prepared:
    rgb: RGB
    valid_mask: Mask
    transform: Transform
    edge: Field
    edge_direction: Field  # gradient normal in radians, not tangent
    face_groups: dict[str, Field]
    face_status: str


@dataclass(frozen=True)
class Structure:
    responses: tuple[Field, Field, Field]
    support: Field
    stroke_score: Field
    model_info: dict[str, Any]
    layer_info: list[dict[str, Any]]


@dataclass(frozen=True)
class Layers:
    color: RGB
    stroke_alpha: Field
    paths: list[Field]
    shadow: Field | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Result:
    final_rgb: RGB
    stages: dict[str, RGB]
    timings: dict[str, float]
    warnings: tuple[str, ...]


class Prepare(Protocol):
    def __call__(self, image: RGB, config: dict[str, Any], face_model: Any) -> Prepared: ...


class Extract(Protocol):
    def __call__(self, prepared: Prepared, encoder: Any, config: dict[str, Any]) -> Structure: ...


class Stylize(Protocol):
    def __call__(self, prepared: Prepared, structure: Structure, config: dict[str, Any]) -> Layers: ...


class Compose(Protocol):
    def __call__(self, layers: Layers, transform: Transform, config: dict[str, Any]) -> Result: ...
