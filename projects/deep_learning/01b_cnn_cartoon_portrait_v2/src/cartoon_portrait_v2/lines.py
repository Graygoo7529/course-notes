"""Comic line branch: semantic candidates, graph paths and tapered strokes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import robust01


@dataclass(frozen=True)
class LineConfig:
    low: float = 0.16
    high: float = 0.34
    join_gap: int = 3
    tangent_cos: float = 0.65
    min_coherence: float = 0.16
    width: float = 0.68
    dark_width: float = 0.82
    outline_width: float = 1.05
    opacity: float = 0.92
    min_length: float = 2.8
    fit_error: float = 0.18
    taper: float = 2.0
    outline_strength: float = 0.78


@dataclass(frozen=True)
class Stroke:
    points: np.ndarray
    source: str
    width: float
    opacity: float
    closed: bool
    taper_start: bool
    taper_end: bool
    taper: float = 2.0


@dataclass
class LinePlan:
    edge: np.ndarray
    dark: np.ndarray
    outline: np.ndarray
    internal: np.ndarray
    nms_edge: np.ndarray
    nms_dark: np.ndarray
    score: np.ndarray
    source: np.ndarray
    retained: np.ndarray
    skeleton: np.ndarray
    bridged: np.ndarray
    strokes: list[Stroke]


def _sample(a: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    h, w = a.shape
    return cv2.remap(a.astype(np.float32), np.clip(x, 0, w - 1).astype(np.float32),
                     np.clip(y, 0, h - 1).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def oriented_nms(value: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Keep a response when it is a local maximum along its normal direction."""
    yy, xx = np.mgrid[:value.shape[0], :value.shape[1]].astype(np.float32)
    nx, ny = np.cos(theta), np.sin(theta)
    plus = _sample(value, yy + ny, xx + nx)
    minus = _sample(value, yy - ny, xx - nx)
    # Strict on one side avoids a two-pixel band at ideal step edges.
    return np.where((value >= plus) & (value > minus), value, 0).astype(np.float32)


def hysteresis(score: np.ndarray, low: float, high: float, mask: np.ndarray) -> np.ndarray:
    """Keep weak responses only when their 8-connected component has a strong seed."""
    weak = (score >= low) & mask
    strong = (score >= high) & mask
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    accepted = np.zeros(count, dtype=bool)
    accepted[labels[strong]] = True
    accepted[0] = False
    return accepted[labels]


def zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Topology-preserving thinning used for ordered comic strokes."""
    result = binary.astype(bool).copy()
    for _ in range(max(binary.shape) + 1):
        changed = False
        padded = np.pad(result.astype(np.uint8), 1)
        p = (padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
             padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2])
        count = sum(p)
        transitions = sum(((p[i] == 0) & (p[(i + 1) % 8] > 0)).astype(np.uint8) for i in range(8))
        remove = result & (count >= 2) & (count <= 6) & (transitions == 1)
        remove &= (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
        if np.any(remove):
            result[remove] = False
            changed = True
        padded = np.pad(result.astype(np.uint8), 1)
        p = (padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
             padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2])
        count = sum(p)
        transitions = sum(((p[i] == 0) & (p[(i + 1) % 8] > 0)).astype(np.uint8) for i in range(8))
        remove = result & (count >= 2) & (count <= 6) & (transitions == 1)
        remove &= (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
        if np.any(remove):
            result[remove] = False
            changed = True
        if not changed:
            break
    return result


def bridge_short_gaps(skeleton: np.ndarray, theta: np.ndarray, score: np.ndarray,
                      mask: np.ndarray, max_gap: int = 2, tangent_cos: float = .65) -> np.ndarray:
    """Join only nearby endpoints whose local tangents agree."""
    out = skeleton.copy()
    # Two short passes are enough for a one-pixel photograph: the first pass
    # closes the obvious gaps, the second pass can then continue the same
    # contour without using a broad morphological closing operation.
    for _ in range(2):
        ys, xs = np.where(out)
        endpoints: list[tuple[int, int]] = []
        for y, x in zip(ys, xs):
            n = int(np.sum(out[max(0, y - 1):y + 2, max(0, x - 1):x + 2])) - 1
            if n == 1:
                endpoints.append((y, x))
        used: set[int] = set()
        for i, (y1, x1) in enumerate(endpoints):
            if i in used:
                continue
            for j in range(i + 1, len(endpoints)):
                if j in used:
                    continue
                y2, x2 = endpoints[j]
                distance = float(np.hypot(y2 - y1, x2 - x1))
                if distance < 1.1 or distance > max_gap + 1.01:
                    continue
                direction = np.array([x2 - x1, y2 - y1], np.float32)
                direction /= max(float(np.linalg.norm(direction)), 1e-6)
                tangent1 = np.array([-np.sin(theta[y1, x1]), np.cos(theta[y1, x1])])
                tangent2 = np.array([-np.sin(theta[y2, x2]), np.cos(theta[y2, x2])])
                if max(abs(float(direction @ tangent1)), abs(float(direction @ tangent2))) < tangent_cos:
                    continue
                t = np.linspace(0, 1, int(round(distance)) + 1)
                xx = np.rint(x1 + t * (x2 - x1)).astype(int)
                yy = np.rint(y1 + t * (y2 - y1)).astype(int)
                if np.all(mask[yy, xx]) and float(np.mean(score[yy, xx])) >= .02:
                    out[yy, xx] = True
                    used.update((i, j))
                    break
    return out


def trace_paths(skeleton: np.ndarray) -> list[tuple[np.ndarray, bool, bool, bool]]:
    """Trace every skeleton graph edge once, splitting at junctions."""
    height, width = skeleton.shape
    nodes = set(int(i) for i in np.flatnonzero(skeleton))
    neighbors: dict[int, list[int]] = {}
    for node in sorted(nodes):
        y, x = divmod(node, width)
        adjacent: list[int] = []
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1)):
            yy, xx = y + dy, x + dx
            candidate = yy * width + xx
            if not (0 <= yy < height and 0 <= xx < width) or candidate not in nodes:
                continue
            if dx and dy and ((y * width + xx in nodes) or (yy * width + x in nodes)):
                continue
            adjacent.append(candidate)
        neighbors[node] = adjacent
    visited: set[tuple[int, int]] = set()
    paths: list[tuple[np.ndarray, bool, bool, bool]] = []
    starts = sorted(nodes, key=lambda i: (len(neighbors[i]) == 2, i))
    for start in starts:
        if not neighbors[start]:
            paths.append((np.array([[start % width, start // width]], np.int32), False, True, True))
        for next_node in neighbors[start]:
            edge = (min(start, next_node), max(start, next_node))
            if edge in visited:
                continue
            path = [start]
            previous, current = start, next_node
            visited.add(edge)
            while True:
                path.append(current)
                if current == start or len(neighbors[current]) != 2:
                    break
                successor = next(v for v in neighbors[current] if v != previous)
                edge = (min(current, successor), max(current, successor))
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, successor
            closed = path[-1] == start
            if closed:
                path.pop()
            points = np.array([[v % width, v // width] for v in path], np.int32)
            paths.append((points, closed, len(neighbors[start]) == 1, len(neighbors[current]) == 1))
    return paths


def _fit(points: np.ndarray, closed: bool, error: float) -> np.ndarray:
    if len(points) < 3 or error <= 0:
        return points.astype(np.float32)
    if closed:
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)
    else:
        previous = np.vstack([points[0], points[:-1]])
        following = np.vstack([points[1:], points[-1]])
    delta = (previous + following) * .5 - points
    before, after = points - previous, following - points
    cosine = np.sum(before * after, axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-6)
    delta[cosine < .4] = 0
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(.5, error / np.maximum(length, 1e-6))[:, None]
    if not closed:
        delta[0] = delta[-1] = 0
    return (points + delta).astype(np.float32)


def _shared_scale(values: np.ndarray, mask: np.ndarray) -> float:
    sample = values[:, mask] if values.ndim == 3 else values[mask]
    finite = sample[np.isfinite(sample)]
    return max(.01, float(np.quantile(np.abs(finite), .98))) if finite.size else 1.0


def _make_strokes(skeleton: np.ndarray, source: np.ndarray, score: np.ndarray,
                  config: LineConfig) -> list[Stroke]:
    strokes: list[Stroke] = []
    for points, closed, start_tip, end_tip in trace_paths(skeleton):
        if len(points) < 2:
            continue
        xy = points[:, [0, 1]]
        length = float(np.linalg.norm(np.diff(xy.astype(np.float32), axis=0), axis=1).sum())
        values = source[points[:, 1], points[:, 0]]
        # 1=outline, 2=internal edge, 3=dark accent.
        counts = np.bincount(values, minlength=4).tolist()
        source_id = max((1, 2, 3), key=lambda label: counts[label])
        if length < config.min_length and source_id != 1:
            continue
        name = ("outline", "edge", "dark")[source_id - 1]
        width = {1: config.outline_width, 2: config.width, 3: config.dark_width}[source_id]
        strength = float(np.mean(score[points[:, 1], points[:, 0]]))
        width *= .82 + .36 * np.clip(strength, 0, 1)
        strokes.append(Stroke(_fit(xy, closed, config.fit_error), name, width,
                              config.opacity, closed, start_tip and not closed,
                              end_tip and not closed, config.taper))
    return strokes


def generate_lines(features, mask: np.ndarray, config: LineConfig | None = None) -> LinePlan:
    cfg = config or LineConfig()
    if cfg.low <= 0 or cfg.low >= cfg.high:
        raise ValueError("line thresholds must satisfy 0 < low < high")
    # A common scale keeps fine/coarse responses comparable instead of making
    # every scale independently look equally strong.
    edge_stack = features.edges.astype(np.float32)
    scale = _shared_scale(edge_stack, mask)
    normalized_edges = np.clip(edge_stack / scale, 0, 1)
    fine = normalized_edges[0]
    coarse = np.max(normalized_edges[1:], axis=0) if len(normalized_edges) > 1 else fine
    reliability = np.clip(features.coherence, 0, 1)
    fine_ratio = fine / (fine + .70 * coarse + 1e-6)
    internal = fine * (.5 + .5 * reliability) * (.45 + .55 * fine_ratio)
    dark_stack = np.maximum(features.dog.astype(np.float32), 0)
    dark_scale = _shared_scale(dark_stack, mask)
    dark = np.clip(dark_stack[0] / dark_scale, 0, 1) * (.60 + .40 * reliability)
    nms_edge = oriented_nms(internal, features.theta)
    nms_dark = oriented_nms(dark, features.theta)
    # A broad dark ridge is kept as tone; it only gently suppresses a duplicate
    # structural edge instead of turning into a thick black band.
    near_dark = cv2.dilate(nms_dark, np.ones((3, 3), np.uint8))
    # A dark ridge already carries a semantic stroke.  Suppress the nearby
    # duplicate edge more strongly so a mouth or eyelid does not become two
    # parallel ink bands.
    nms_edge *= 1 - .50 * np.clip(near_dark, 0, 1)
    edge_allowed = reliability >= cfg.min_coherence
    nms_edge *= edge_allowed
    nms_dark *= (.70 + .30 * edge_allowed)
    outline = (mask & ~(
        cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                  borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0)).astype(np.float32)
    outline_score = outline * cfg.outline_strength
    score = np.maximum(np.maximum(outline_score, nms_edge), nms_dark).astype(np.float32)
    source = np.argmax(np.stack((outline_score, nms_edge, nms_dark), axis=-1), axis=-1).astype(np.uint8)
    # Source ids are 1=outline, 2=edge, 3=dark.
    source = np.where(score > 0, source + 1, 0).astype(np.uint8)
    retained = hysteresis(score, cfg.low, cfg.high, mask)
    skeleton = zhang_suen(retained)
    bridged = bridge_short_gaps(skeleton, features.theta, score, mask,
                                cfg.join_gap, cfg.tangent_cos)
    strokes = _make_strokes(bridged, source, score, cfg)
    return LinePlan(fine, dark, outline, internal, nms_edge, nms_dark, score, source,
                    retained, skeleton, bridged, strokes)
