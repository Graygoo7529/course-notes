"""区域任务：邻接权重、确定性颜色原型、连通区域、区域内明暗与配色。"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .colors import ColorConfig, build_boundary_weights, edge_aware_smooth
from .feature_bank import FeatureBank
from .pipeline import FloatImage, Mask, _integer, _number, _mask

IntMap = NDArray[np.int32]


@dataclass(frozen=True)
class RegionConfig:
    colors: int = 6
    smoothness: float = 6.0
    merge_area: int = 8
    merge_distance: float = 0.65
    palette: tuple[tuple[float, float, float], ...] = ()
    shadow_depth: float = 9.0       # Lab L* units
    shadow_fraction: float = 0.25
    shadow_min_range: float = 0.09  # L*/100
    shadow_style: str = "warm"

    def validate(self) -> None:
        _integer(self.colors, "palette_size", 2)
        if self.colors > 24:
            raise ValueError("palette_size 不超过 24。")
        _number(self.smoothness, "region_smoothness", 0, 100)
        _integer(self.merge_area, "merge_area", 0)
        _number(self.merge_distance, "merge_distance", 0, 5)
        _number(self.shadow_depth, "shadow_depth", 0, 40)
        _number(self.shadow_fraction, "shadow_fraction", 0, 0.5)
        _number(self.shadow_min_range, "shadow_min_range", 0.001, 1)
        if self.shadow_style not in ("warm", "cool", "neutral"):
            raise ValueError("shadow_style 必须为 warm、cool 或 neutral。")
        if len(self.palette) > 32:
            raise ValueError("自定义调色板不超过 32 色。")
        for color in self.palette:
            if len(color) != 3:
                raise ValueError("调色板颜色必须有三个 RGB 分量。")
            for value in color:
                _number(value, "palette_rgb", 0, 1)


@dataclass(frozen=True)
class RegionFeatures:
    descriptor: FloatImage
    right: FloatImage
    down: FloatImage
    affinity: FloatImage
    solver_residual: float


@dataclass(frozen=True)
class RegionScene:
    features: RegionFeatures
    initial_labels: IntMap
    labels: IntMap                  # 0 background, 1..N connected regions
    clusters: IntMap                # color group, 0 background
    area: NDArray[np.int64]


@dataclass(frozen=True)
class FillPlan:
    labels: IntMap
    palette: FloatImage             # N+1,3; row zero is a placeholder
    shadow_palette: FloatImage
    base: FloatImage
    detail: FloatImage
    shadow_candidate: Mask
    shadow: Mask
    thresholds: FloatImage
    flat: FloatImage
    colors: FloatImage
    solver_residual: float


def fuse_region_features(bank: FeatureBank, mask: Mask, config: RegionConfig) -> RegionFeatures:
    config.validate()
    _mask(mask, bank.lightness.shape)
    right, down, affinity = build_boundary_weights(
        bank.guides[0], bank.fine_structure, mask,
        ColorConfig(tau_lightness=14, tau_chroma=10, tau_gradient=0.18),
    )
    descriptor = np.zeros_like(bank.lab)
    residuals = []
    for channel in range(3):
        result, residual, _ = edge_aware_smooth(
            bank.lab[..., channel], right, down, mask, config.smoothness,
        )
        descriptor[..., channel] = result
        residuals.append(residual)
    return RegionFeatures(descriptor, right, down, affinity, max(residuals))


def _prototypes(points: FloatImage, count: int) -> FloatImage:
    """只拟合当前图的颜色；确定性最远点初始化，不修改 OpenCV RNG。"""
    samples = points[::max(1, len(points) // 12000)]
    centers = [np.median(samples, axis=0).astype(np.float32)]
    distances = np.sum((samples - centers[0]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(distances))
        if distances[index] < 1e-5:
            break
        centers.append(samples[index].copy())
        distances = np.minimum(distances, np.sum((samples - centers[-1]) ** 2, axis=1))
    values = np.stack(centers)
    for _ in range(15):
        labels = np.argmin(np.sum((samples[:, None] - values[None]) ** 2, axis=2), axis=1)
        updated = values.copy()
        for index in range(len(values)):
            selected = samples[labels == index]
            if len(selected):
                updated[index] = np.mean(selected, axis=0)
        if np.max(np.abs(updated - values)) < 1e-4:
            break
        values = updated
    return values


def _connected_groups(groups: IntMap, mask: Mask) -> IntMap:
    result = np.zeros(mask.shape, dtype=np.int32)
    offset = 0
    for group in np.unique(groups[mask]):
        count, labels = cv2.connectedComponents((mask & (groups == group)).astype(np.uint8), connectivity=4)
        selected = labels > 0
        result[selected] = labels[selected].astype(np.int32) + offset
        offset += count - 1
    return result


def organize_regions(features: RegionFeatures, mask: Mask, protected: Mask, config: RegionConfig) -> RegionScene:
    """空间连通的颜色初分区，按邻接亲和度合并小区域；保留受保护细节。"""
    config.validate()
    _mask(mask, features.descriptor.shape)
    if protected.shape != mask.shape or protected.dtype != np.bool_:
        raise ValueError("protected 必须与 mask 同形状且为 bool。")
    scale = np.array([45, 14, 14], dtype=np.float32)
    points = features.descriptor[mask] / scale
    centers = _prototypes(points, config.colors)
    groups = np.zeros(mask.shape, dtype=np.int32)
    assigned = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), 65536):
        distances = np.sum((points[start:start + 65536, None] - centers[None]) ** 2, axis=2)
        assigned[start:start + 65536] = np.argmin(distances, axis=1).astype(np.int32) + 1
    groups[mask] = assigned
    labels = _connected_groups(groups, mask)
    initial = labels.copy()
    # 区域邻接图；小区域仅可并入实际相邻、颜色足够接近的区域。
    count = int(labels.max()) + 1
    area = np.bincount(labels.ravel(), minlength=count)
    means = np.zeros((count, 3), dtype=np.float64)
    for c in range(3):
        means[:, c] = np.bincount(labels.ravel(), weights=(features.descriptor[..., c] / scale[c]).ravel(), minlength=count)
    means /= np.maximum(area[:, None], 1)
    pinned = np.zeros(count, dtype=bool)
    pinned[np.unique(labels[protected & mask])] = True
    edges: dict[tuple[int, int], list[float]] = {}
    for first, second, weights in ((labels[:, :-1], labels[:, 1:], features.right),
                                    (labels[:-1], labels[1:], features.down)):
        select = (first > 0) & (second > 0) & (first != second)
        pairs = np.sort(np.stack((first[select], second[select]), axis=1), axis=1)
        if not len(pairs):
            continue
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        totals = np.bincount(inverse, weights=weights[select])
        counts = np.bincount(inverse)
        for pair, total, number in zip(unique, totals, counts):
            key = (int(pair[0]), int(pair[1]))
            acc = edges.setdefault(key, [0.0, 0.0])
            acc[0] += float(total)
            acc[1] += float(number)
    neighbors: dict[int, list[tuple[int, float]]] = {}
    for (a, b), (total, number) in edges.items():
        neighbors.setdefault(a, []).append((b, total / number))
        neighbors.setdefault(b, []).append((a, total / number))
    parent = np.arange(count, dtype=np.int32)

    def root(index: int) -> int:
        while parent[index] != index:
            index = int(parent[index])
        return index

    for index in np.argsort(area[1:]) + 1:
        index = int(index)
        if area[index] >= config.merge_area or pinned[index]:
            continue
        choices = []
        for neighbor, affinity in neighbors.get(index, []):
            target = root(neighbor)
            distance = float(np.linalg.norm(means[index] - means[target]))
            if target != index and affinity > 0.1 and distance < config.merge_distance:
                choices.append((affinity / (0.1 + distance), target))
        if choices:
            target = max(choices)[1]
            means[target] = (means[target] * area[target] + means[index] * area[index]) / (area[target] + area[index])
            area[target] += area[index]
            parent[index] = target
    lookup = np.array([root(i) for i in range(count)], dtype=np.int32)
    merged = lookup[labels]
    # 合并只沿邻接进行；再次连通标号使接口保证每个编号是一块连通区域。
    labels = _connected_groups(merged, mask)
    return RegionScene(features, initial, labels, groups, np.bincount(labels.ravel()).astype(np.int64))


def design_fills(bank: FeatureBank, scene: RegionScene, mask: Mask, protected: Mask,
                 config: RegionConfig) -> FillPlan:
    config.validate()
    labels = scene.labels
    right = scene.features.right * (labels[:, :-1] == labels[:, 1:])
    down = scene.features.down * (labels[:-1] == labels[1:])
    base, residual, _ = edge_aware_smooth(bank.lightness, right, down, mask, config.smoothness)
    detail = np.where(mask, bank.lightness - base, 0).astype(np.float32)
    count = int(labels.max()) + 1
    palette_lab = np.zeros((count, 3), dtype=np.float32)
    candidate = np.zeros(mask.shape, dtype=bool)
    shadow = candidate.copy()
    thresholds = np.full(count, -1, dtype=np.float32)
    for region in range(1, count):
        selected = labels == region
        values = base[selected]
        low, high = np.quantile(values, [0.1, 0.9])
        samples = selected & (base >= low) & (base <= high) & ~protected & (bank.dog[0] < 0.025)
        if not samples.any():
            samples = selected
        palette_lab[region] = np.median(bank.lab[samples], axis=0)
        if config.shadow_fraction == 0 or config.shadow_depth == 0 or high - low < config.shadow_min_range:
            continue
        threshold = float(np.quantile(values, config.shadow_fraction))
        thresholds[region] = threshold
        seed = selected & (base < threshold) & ~protected
        candidate |= seed
        n, components, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
        minimum = max(2, int(np.count_nonzero(selected) * 0.01))
        for component in range(1, n):
            if stats[component, cv2.CC_STAT_AREA] >= minimum:
                shadow |= components == component
    palette = np.clip(cv2.cvtColor(palette_lab[None], cv2.COLOR_Lab2RGB)[0], 0, 1).astype(np.float32)
    if config.palette:
        custom = np.array(config.palette, dtype=np.float32)
        custom_lab = cv2.cvtColor(custom[None], cv2.COLOR_RGB2Lab)[0]
        match = np.argmin(np.sum((palette_lab[:, None] - custom_lab[None]) ** 2, axis=2), axis=1)
        palette = custom[match].copy()
        palette_lab = cv2.cvtColor(palette[None], cv2.COLOR_RGB2Lab)[0].astype(np.float32)
    shadow_lab = palette_lab.copy()
    shadow_lab[:, 0] = np.maximum(0, shadow_lab[:, 0] - config.shadow_depth)
    if config.shadow_style == "warm":
        shadow_lab[:, 1:] += np.array([2, 2], dtype=np.float32)
    elif config.shadow_style == "cool":
        shadow_lab[:, 1:] += np.array([2, -4], dtype=np.float32)
    dark_palette = np.clip(cv2.cvtColor(shadow_lab[None], cv2.COLOR_Lab2RGB)[0], 0, 1).astype(np.float32)
    palette[0] = dark_palette[0] = 0
    flat = palette[labels]
    colors = np.where(shadow[..., None], dark_palette[labels], flat).astype(np.float32)
    return FillPlan(labels.copy(), palette, dark_palette, base, detail, candidate, shadow,
                    thresholds, flat, colors, max(residual, scene.features.solver_residual))
