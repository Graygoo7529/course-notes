"""线条任务：有来源的候选、骨架图、受约束路径与独立笔画属性。"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .feature_bank import FeatureBank
from .lines import hysteresis, thin_edges
from .pipeline import FloatImage, Mask, _number


@dataclass(frozen=True)
class StrokeConfig:
    edge_scale: float = 0.075
    dark_scale: float = 0.07
    low: float = 0.20
    high: float = 0.48
    width: float = 0.8
    outline_width: float = 1.1
    opacity: float = 0.92
    ink: tuple[float, float, float] = (0.12, 0.10, 0.14)
    min_length: float = 2.0
    fit_error: float = 0.3
    taper: float = 2.0

    def validate(self) -> None:
        for name, value in (("edge_scale", self.edge_scale), ("dark_scale", self.dark_scale)):
            _number(value, name, 0.001, 2)
        _number(self.low, "line_low", 0.001, 1)
        _number(self.high, "line_high", self.low + 0.001, 1)
        _number(self.width, "line_width", 0.1, 10)
        _number(self.outline_width, "outline_width", 0, 20)
        _number(self.opacity, "ink_opacity", 0, 1)
        _number(self.min_length, "min_length", 0, 50)
        _number(self.fit_error, "fit_error", 0, 0.75)
        _number(self.taper, "taper", 0, 20)
        if len(self.ink) != 3:
            raise ValueError("ink 必须是 RGB 三元组。")
        for value in self.ink:
            _number(value, "ink_rgb", 0, 1)


@dataclass(frozen=True)
class LineFeatures:
    edge: FloatImage
    dark: FloatImage
    score: FloatImage
    source: NDArray[np.uint8]       # 1=edge; 2=dark; 0=none
    theta: FloatImage
    coherence: FloatImage
    scale_support: FloatImage
    retained: Mask
    skeleton: Mask
    protected: Mask
    suppressed: Mask


@dataclass(frozen=True)
class Stroke:
    points: FloatImage             # N,2; original pixel center coordinates
    source: str
    width: float
    opacity: float
    closed: bool
    taper_start: bool
    taper_end: bool


@dataclass(frozen=True)
class StrokeSet:
    strokes: tuple[Stroke, ...]
    accepted: Mask
    rejected: Mask
    ink: tuple[float, float, float]
    taper: float


def skeletonize(binary: Mask) -> Mask:
    """Zhang–Suen 拓扑细化，保持原图大小，边界外为背景。"""
    result = binary.copy()
    for _ in range(max(binary.shape) + 1):
        changed = False
        for phase in (0, 1):
            padded = np.pad(result.astype(np.uint8), 1)
            p = (padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
                 padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2])
            count = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i + 1) % 8] > 0)).astype(np.uint8) for i in range(8))
            if phase == 0:
                condition = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                condition = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = result & (count >= 2) & (count <= 6) & (transitions == 1) & condition
            if remove.any():
                result[remove] = False
                changed = True
        if not changed:
            break
    # 极小实心块可能在并行删除时消失；恢复到该分量中最靠近质心的像素。
    count, components, _, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    components = components.astype(np.int32, copy=False)
    present = np.bincount(components[result], minlength=count) > 0
    for component in range(1, count):
        if not present[component]:
            y, x = np.nonzero(components == component)
            nearest = int(np.argmin((x - centroids[component, 0]) ** 2 + (y - centroids[component, 1]) ** 2))
            result[y[nearest], x[nearest]] = True
    return result


def trace_line_features(bank: FeatureBank, mask: Mask, protected: Mask, suppressed: Mask,
                        config: StrokeConfig) -> tuple[LineFeatures, dict[str, FloatImage | Mask]]:
    """与生产管道共用计算，保留抑制前后的状态；数组不作为可变工作区复用。"""
    config.validate()
    for name, hint in (("protected", protected), ("suppressed", suppressed)):
        if hint.shape != mask.shape or hint.dtype != np.bool_:
            raise ValueError(f"{name} 必须与 mask 同形状且为 bool。")
        if np.any(hint & ~mask):
            raise ValueError(f"{name} 不能包含人物遮罩外的位置。")
    if np.any(protected & suppressed):
        raise ValueError("保护与删除标记不能重叠。")
    theta, q = bank.theta[0], bank.coherence[0]
    fine = bank.edges[0]
    coarse = np.max(bank.edges, axis=0)
    # 跨尺度支持只调整现有幅值，不生成连接；最大值也包括最细尺度。
    support = np.clip(coarse / np.float32(config.edge_scale), 0, 1)
    raw_dark = np.maximum(bank.dog[0], 0)
    dark_normalized = raw_dark / np.float32(config.dark_scale)
    dark_clipped = np.clip(dark_normalized, 0, 1)
    dark = thin_edges(dark_clipped, theta, q, mask)
    ridge_near = cv2.dilate(dark, np.ones((3, 3), dtype=np.uint8)).astype(np.float32)
    edge_normalized = fine / np.float32(config.edge_scale)
    edge_clipped = np.clip(edge_normalized, 0, 1)
    edge_thinned = thin_edges(edge_clipped, theta, q, mask)
    # 单独保留颜色边界，同时避免同一条窄暗线被重复画成两侧轮廓。
    edge_supported = edge_thinned * (np.float32(0.75) + np.float32(0.25) * support)
    edge = np.where(ridge_near >= config.high, 0, edge_supported).astype(np.float32)
    score = np.maximum(dark, np.float32(0.65) * edge).astype(np.float32)
    source = np.where(dark >= 0.65 * edge, 2, 1).astype(np.uint8)
    # 遮罩外占位不参与响应；最外一圈交由明确的轮廓笔画处理。
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0
    allowed = mask & (interior | protected) & ~suppressed
    score_before_gate = score.copy()
    score[~allowed] = 0
    source[score == 0] = 0
    retained = hysteresis(score, mask, config.low, config.high)
    retained |= protected & (score >= config.low * 0.4)
    retained &= allowed
    skeleton = skeletonize(retained)
    features = LineFeatures(edge, dark, score, source, theta.copy(), q.copy(), support,
                            retained, skeleton, protected.copy(), suppressed.copy())
    stages: dict[str, FloatImage | Mask] = {
        "dark_normalized": dark_normalized, "dark_clipped": dark_clipped,
        "dark_nms": dark, "ridge_near": ridge_near,
        "edge_normalized": edge_normalized, "edge_clipped": edge_clipped,
        "edge_nms": edge_thinned, "edge_supported": edge_supported,
        "edge_after_ridge": edge, "ridge_removed": (edge_supported > 0) & (edge == 0),
        "score_before_gate": score_before_gate, "allowed": allowed, "score": score,
        "weak": (score >= config.low) & allowed, "strong": (score >= config.high) & allowed,
        "retained": retained, "skeleton": skeleton,
    }
    return features, stages


def fuse_line_features(bank: FeatureBank, mask: Mask, protected: Mask, suppressed: Mask,
                       config: StrokeConfig) -> LineFeatures:
    return trace_line_features(bank, mask, protected, suppressed, config)[0]


def trace_paths(skeleton: Mask) -> list[tuple[NDArray[np.int32], bool, bool, bool]]:
    """骨架转无向图，每条图边只追踪一次；分叉与闭环分别处理。"""
    height, width = skeleton.shape
    nodes = set(int(i) for i in np.flatnonzero(skeleton))
    neighbors: dict[int, list[int]] = {}
    for node in sorted(nodes):
        y, x = divmod(node, width)
        adjacent = []
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            yy, xx = y + dy, x + dx
            candidate = yy * width + xx
            if not (0 <= yy < height and 0 <= xx < width) or candidate not in nodes:
                continue
            # 存在正交路线时不增加三角形对角边，避免伪分叉。
            if dx and dy and (y * width + xx in nodes or yy * width + x in nodes):
                continue
            adjacent.append(candidate)
        neighbors[node] = adjacent
    visited: set[tuple[int, int]] = set()
    paths = []
    starts = sorted(nodes, key=lambda i: (len(neighbors[i]) == 2, i))
    for start in starts:
        if not neighbors[start]:
            paths.append((np.array([[start % width, start // width]], dtype=np.int32), False, True, True))
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
                candidates = [v for v in neighbors[current] if v != previous]
                successor = candidates[0]
                edge = (min(current, successor), max(current, successor))
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, successor
            closed = path[-1] == start
            if closed:
                path.pop()
            xy = np.array([[v % width, v // width] for v in path], dtype=np.int32)
            paths.append((xy, closed, len(neighbors[start]) == 1, len(neighbors[current]) == 1))
    return paths


def _fit(points: FloatImage, closed: bool, error: float) -> FloatImage:
    """有限偏移的折线平滑；保护急弯，端点不动。不是无约束样条拟合。"""
    if len(points) < 3 or error == 0:
        return points.copy()
    previous, following = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    delta = (previous + following) * 0.5 - points
    before, after = points - previous, following - points
    denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.sum(before * after, axis=1) / np.maximum(denom, 1e-6)
    delta[cosine < 0.4] = 0
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(0.5, error / np.maximum(length, 1e-6))[:, None]
    if not closed:
        delta[0] = delta[-1] = 0
    return (points + delta).astype(np.float32)


def design_strokes(lines: LineFeatures, mask: Mask, config: StrokeConfig) -> StrokeSet:
    config.validate()
    accepted = np.zeros_like(mask)
    strokes = []
    for path, closed, start_tip, end_tip in trace_paths(lines.skeleton):
        x, y = path[:, 0], path[:, 1]
        length = float(np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1).sum())
        protected = bool(lines.protected[y, x].any())
        if length < config.min_length and not protected:
            continue
        is_dark = bool(np.mean(lines.source[y, x] == 2) >= 0.5)
        points = _fit(path.astype(np.float32), closed, config.fit_error)
        # 平滑后若落到遮罩外或删除提示中，则回退该点的原位置。
        rounded = np.rint(points).astype(np.int32)
        invalid = ~mask[rounded[:, 1], rounded[:, 0]] | lines.suppressed[rounded[:, 1], rounded[:, 0]]
        points[invalid] = path[invalid]
        accepted[y, x] = True
        strokes.append(Stroke(points, "protected" if protected else ("dark" if is_dark else "edge"),
                              config.width * (1 if is_dark or protected else 0.65), config.opacity,
                              closed, start_tip and not closed, end_tip and not closed))
    if config.outline_width > 0:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        h, w = mask.shape
        for contour in contours:
            points = contour[:, 0, :].astype(np.float32)
            if len(points) < 2:
                continue
            following = np.roll(points, -1, axis=0)
            frame_edge = (((points[:, 0] == 0) & (following[:, 0] == 0)) |
                          ((points[:, 0] == w - 1) & (following[:, 0] == w - 1)) |
                          ((points[:, 1] == 0) & (following[:, 1] == 0)) |
                          ((points[:, 1] == h - 1) & (following[:, 1] == h - 1)))
            if not frame_edge.any():
                strokes.append(Stroke(_fit(points, True, config.fit_error), "outline", config.outline_width,
                                      config.opacity, True, False, False))
            else:
                # 图像裁切边不描黑；将其余轮廓拆成开放路径。
                first = int(np.flatnonzero(frame_edge)[0])
                run: list[FloatImage] = []
                for step in range(1, len(points) + 1):
                    index = (first + step) % len(points)
                    if not run:
                        run.append(points[index])
                    if frame_edge[index]:
                        if len(run) >= 2:
                            strokes.append(Stroke(_fit(np.stack(run), False, config.fit_error), "outline",
                                                  config.outline_width, config.opacity, False, False, False))
                        run = []
                    else:
                        run.append(following[index])
    return StrokeSet(tuple(strokes), accepted, lines.retained & ~accepted, config.ink, config.taper)
