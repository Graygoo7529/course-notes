"""NMS, graph paths and bounded B-spline smoothing, without inventing links."""
import warnings

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree


def nonmaximum(score: np.ndarray, direction: np.ndarray) -> np.ndarray:
    y, x = np.indices(score.shape, dtype=np.float32)
    dx, dy = np.cos(direction), np.sin(direction)
    plus = cv2.remap(score, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    minus = cv2.remap(score, x - dx, y - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return np.where((score >= plus) & (score > minus), score, 0).astype(np.float32)


def select_lines(score: np.ndarray, edge: np.ndarray, direction: np.ndarray,
                 masks: dict, face_config: dict, density: float) -> tuple[np.ndarray, dict]:
    nms = nonmaximum(score, direction)
    # Use the shared image edge as the threshold reference; do not rescale each Q
    # back to the same histogram and accidentally cancel the effect of CNN gating.
    reference = nonmaximum(edge, direction)
    values = reference[reference > 0.04]
    high = float(np.quantile(values, 0.85 - 0.30 * density)) if len(values) else 1.0
    high = max(0.10, high * (0.48 - 0.20 * density))
    multiplier = np.ones(score.shape, dtype=np.float32)
    multiplier[masks["cheeks"]] = face_config["cheek_threshold_multiplier"]
    multiplier[masks["features"]] = face_config["eye_mouth_threshold_multiplier"]
    weak = nms >= high * 0.45 * multiplier
    strong = nms >= high * multiplier
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    labels = labels.astype(np.int32, copy=False)
    accepted = np.zeros(count, bool)
    accepted[np.unique(labels[strong])] = True
    accepted[0] = False
    # Retain small eye/mouth details only if they also contain a strong response.
    protected = np.unique(labels[masks["features"] & weak])
    accepted &= (stats[:, cv2.CC_STAT_AREA] >= 7) | np.isin(np.arange(count), protected)
    accepted[0] = False
    retained = accepted[labels].astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(retained * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    return skeleton, {"threshold_high": high, "threshold_low": high * 0.45,
                      "threshold_reference": "shared image Sobel; fixed across encoders for the same input",
                      "nms_pixels": int((nms > 0).sum()), "skeleton_pixels": int(skeleton.sum())}


def trace_paths(skeleton: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Adapted from the local v1 experiment: visit each undirected graph edge once."""
    h, w = skeleton.shape
    nodes = set(int(i) for i in np.flatnonzero(skeleton))
    neighbors = {}
    for node in sorted(nodes):
        y, x = divmod(node, w)
        adjacent = []
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            yy, xx = y + dy, x + dx
            candidate = yy * w + xx
            if not (0 <= yy < h and 0 <= xx < w) or candidate not in nodes:
                continue
            if dx and dy and (y * w + xx in nodes or yy * w + x in nodes):
                continue
            adjacent.append(candidate)
        neighbors[node] = adjacent
    visited = set()
    paths = []
    for start in sorted(nodes, key=lambda i: (len(neighbors[i]) == 2, i)):
        for next_node in neighbors[start]:
            edge = tuple(sorted((start, next_node)))
            if edge in visited:
                continue
            path, previous, current = [start], start, next_node
            visited.add(edge)
            while True:
                path.append(current)
                if current == start or len(neighbors[current]) != 2:
                    break
                successor = next(v for v in neighbors[current] if v != previous)
                edge = tuple(sorted((current, successor)))
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, successor
            points = np.asarray([[v % w, v // w] for v in path], np.float32)
            paths.append((points, path[-1] == start))
    return paths


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distances) > 1e-6]
    distances, points = distances[keep], points[keep]
    if len(points) < 2:
        return points
    samples = np.linspace(0, distances[-1], max(2, int(np.ceil(distances[-1] / spacing)) + 1))
    return np.column_stack([np.interp(samples, distances, points[:, i]) for i in range(2)]).astype(np.float32)


def smooth_path(points: np.ndarray, closed: bool, bound: float) -> tuple[np.ndarray, dict]:
    """Bound the sampled symmetric path distance; fall back if smoothing moves it."""
    original = points.copy()
    points = resample(points, 1.0)
    if len(points) < 5:
        return original, {"smoothed": False, "deviation_px": 0.0, "reason": "short"}
    # Large turns over a 3-pixel span become anchors; fit independent segments.
    if not closed and len(points) > 8:
        before, after = points[3:-3] - points[:-6], points[6:] - points[3:-3]
        cos = (before * after).sum(axis=1) / np.maximum(np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-6)
        corners = np.flatnonzero(cos < 0.35) + 3
        if len(corners):
            anchors = [0]
            for index in corners:
                if index - anchors[-1] >= 4 and len(points) - 1 - index >= 4:
                    anchors.append(int(index))
            if len(anchors) > 1:
                anchors.append(len(points) - 1)
                chunks, infos = [], []
                for a, b in zip(anchors[:-1], anchors[1:]):
                    fitted, info = _fit_segment(points[a:b + 1], False, bound)
                    chunks.append(fitted if not chunks else fitted[1:])
                    infos.append(info)
                info = {"smoothed": any(i["smoothed"] for i in infos), "reason": "corner_anchors"}
                return _check_original(original, np.concatenate(chunks), info, bound)
    fitted, info = _fit_segment(points, closed, bound)
    return _check_original(original, fitted, info, bound)


def _check_original(original: np.ndarray, fitted: np.ndarray, info: dict, bound: float) -> tuple[np.ndarray, dict]:
    # Resampling can cut corners too, so also check against the original skeleton
    # polyline, not only against the resampled points passed to the spline fitter.
    dense = resample(original, 0.2)
    distance = max(float(KDTree(dense).query(fitted)[0].max()), float(KDTree(fitted).query(dense)[0].max()))
    if not np.isfinite(fitted).all() or distance > bound:
        return original, {"smoothed": False, "deviation_px": 0.0, "reason": "bounded_fallback"}
    return fitted, {**info, "deviation_px": distance}


def _fit_segment(points: np.ndarray, closed: bool, bound: float) -> tuple[np.ndarray, dict]:
    if len(points) < 5:
        return points.copy(), {"smoothed": False, "deviation_px": 0.0, "reason": "short"}
    dense_original = resample(points, 0.2)
    tree = KDTree(dense_original)
    for strength in [0.28, 0.08, 0.0]:
        try:
            weights = np.ones(len(points))
            if not closed:
                weights[[0, -1]] = 100
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                tck, _ = splprep(points.T, w=weights, s=len(points) * strength, k=3, per=closed)
            sampled = np.asarray(splev(np.linspace(0, 1, max(10, len(dense_original))), tck)).T.astype(np.float32)
            sampled[[0, -1]] = points[[0, -1]]
            distance = max(float(tree.query(sampled)[0].max()), float(KDTree(sampled).query(dense_original)[0].max()))
            if np.isfinite(sampled).all() and distance <= bound:
                return sampled, {"smoothed": True, "deviation_px": distance, "reason": "accepted"}
        except (ValueError, TypeError, RuntimeWarning):
            continue
    return points.copy(), {"smoothed": False, "deviation_px": 0.0, "reason": "bounded_fallback"}
