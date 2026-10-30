"""Shared flat colors and antialiased ink; CNN only controls stroke support."""
import cv2
import numpy as np
from PIL import Image

from .contracts import Layers, Prepared, Result, Structure, Transform
from .preprocess import unpad
from .strokes import select_lines, smooth_path, trace_paths


def color_layer(rgb: np.ndarray, style: dict) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    smooth = cv2.edgePreservingFilter(bgr, flags=1, sigma_s=55, sigma_r=0.30)
    smooth = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)
    palette = np.asarray(style["palette_rgb"], np.uint8)
    if style["palette"] == "pastel":
        palette = np.clip(palette.astype(float) * 0.85 + np.array([255, 245, 233]) * 0.15, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(smooth.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    palette_lab = cv2.cvtColor(palette[None].astype(np.float32) / 255, cv2.COLOR_RGB2Lab)[0]
    distances = ((lab[:, :, None] - palette_lab[None, None]) ** 2).sum(axis=3)
    labels = distances.argmin(axis=2).astype(np.uint8)
    # True local majority, not a median over arbitrary palette indices.
    counts = np.stack([cv2.boxFilter((labels == k).astype(np.float32), -1, (3, 3), normalize=False)
                       for k in range(len(palette))], axis=2)
    majority = counts.argmax(axis=2)
    labels = np.where(counts.max(axis=2) >= 5, majority, labels)
    return palette[labels]


def stylize(prepared: Prepared, structure: Structure, config: dict, color: np.ndarray | None = None) -> Layers:
    style, face_config = config["style"], config["face"]
    skeleton, diagnostics = select_lines(structure.stroke_score, prepared.edge, prepared.edge_direction,
                                         prepared.masks, face_config, style["line_density"])
    raw_paths = trace_paths(skeleton)
    h, w = prepared.edge.shape
    scale = int(style["supersample"])
    ink = np.zeros((h * scale, w * scale), np.uint8)
    paths, fits, widths = [], [], []
    for points, closed in raw_paths:
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if length < 3:
            continue
        fitted, info = smooth_path(points, closed, style["max_spline_deviation_px"])
        xy = np.rint(points).astype(int)
        near_contour = np.mean(prepared.masks["contour"][xy[:, 1], xy[:, 0]]) > 0.5
        # Long paths outside the coarse face can describe hair/clothing outlines.
        outside_face = np.mean(prepared.masks["face"][xy[:, 1], xy[:, 0]]) < 0.1
        width = style["contour_width"] if near_contour or (outside_face and length > 40) else style["line_width"]
        poly = np.rint(fitted * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(ink, [poly], closed, 255, max(1, round(width * scale)), cv2.LINE_AA)
        paths.append(fitted)
        fits.append(info)
        widths.append(width)
    alpha = (np.asarray(Image.fromarray(ink).resize((w, h), Image.Resampling.LANCZOS)) / 255).astype(np.float32)
    alpha *= prepared.valid_mask
    diagnostics.update(path_count=len(paths), smoothed_paths=sum(i["smoothed"] for i in fits),
                       spline_fallbacks=sum(i["reason"] == "bounded_fallback" for i in fits),
                       max_deviation_px=max((i["deviation_px"] for i in fits), default=0.0),
                       stroke_pixels=int((alpha > 0.1).sum()), widths=widths,
                       geometry_supplement_strokes=0, shadow_enabled=False)
    return Layers(color if color is not None else color_layer(prepared.rgb, style), alpha, paths, None, diagnostics)


def compose(layers: Layers, transform: Transform, config: dict) -> Result:
    alpha = layers.stroke_alpha[..., None]
    ink = np.asarray(config["style"]["ink_rgb"], np.float32)
    image = np.rint((1 - alpha) * layers.color + alpha * ink).clip(0, 255).astype(np.uint8)
    lines = np.rint(255 * (1 - alpha.repeat(3, axis=2))).astype(np.uint8)
    stages = {"color": layers.color, "lines": lines}
    if config["output"]["remove_padding"]:
        image = unpad(image, transform)
        stages = {key: unpad(array, transform) for key, array in stages.items()}
    return Result(image, stages, {}, ())
