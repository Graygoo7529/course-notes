"""Line branch: oriented thinning, hysteresis and short, direction-aware bridges."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import robust01


@dataclass(frozen=True)
class LineConfig:
    low: float = 0.16
    high: float = 0.34
    join_gap: int = 2
    tangent_cos: float = 0.65


@dataclass
class LinePlan:
    edge: np.ndarray
    dark: np.ndarray
    nms_edge: np.ndarray
    nms_dark: np.ndarray
    score: np.ndarray
    retained: np.ndarray
    skeleton: np.ndarray
    bridged: np.ndarray
    strokes: list[np.ndarray]


def _sample(a: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    h, w = a.shape
    mapx = np.clip(x, 0, w - 1).astype(np.float32)
    mapy = np.clip(y, 0, h - 1).astype(np.float32)
    return cv2.remap(a.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def oriented_nms(value: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Keep a response when it is a local maximum along its normal direction."""
    yy, xx = np.mgrid[:value.shape[0], :value.shape[1]].astype(np.float32)
    nx, ny = np.cos(theta), np.sin(theta)
    plus = _sample(value, yy + ny, xx + nx)
    minus = _sample(value, yy - ny, xx - nx)
    return np.where((value >= plus) & (value >= minus), value, 0).astype(np.float32)


def hysteresis(score: np.ndarray, low: float, high: float, mask: np.ndarray) -> np.ndarray:
    strong = (score >= high) & mask
    weak = (score >= low) & mask
    if not np.any(strong):
        return weak
    grown = strong.copy()
    kernel = np.ones((3, 3), np.uint8)
    while True:
        nxt = (cv2.dilate(grown.astype(np.uint8), kernel) > 0) & weak
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return grown


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Morphological skeleton; it has no learned parameters and is stable on small images."""
    img = (binary.astype(np.uint8) > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(img):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.bitwise_and(img, cv2.bitwise_not(opened)))
        if np.array_equal(eroded, img):
            break
        img = eroded
    return skel > 0


def bridge_short_gaps(skeleton: np.ndarray, theta: np.ndarray, score: np.ndarray,
                      mask: np.ndarray, max_gap: int = 2, tangent_cos: float = .65) -> np.ndarray:
    """Join only nearby endpoints whose local tangents agree."""
    out = skeleton.copy()
    ys, xs = np.where(skeleton)
    endpoints: list[tuple[int, int]] = []
    for y, x in zip(ys, xs):
        n = int(np.sum(skeleton[max(0, y-1):y+2, max(0, x-1):x+2])) - 1
        if n == 1:
            endpoints.append((y, x))
    used: set[int] = set()
    for i, (y1, x1) in enumerate(endpoints):
        if i in used:
            continue
        best = None
        for j in range(i + 1, len(endpoints)):
            if j in used:
                continue
            y2, x2 = endpoints[j]
            d = float(np.hypot(y2-y1, x2-x1))
            if d < 1.1 or d > max_gap + 1.01:
                continue
            vx, vy = float(x2-x1), float(y2-y1)
            norm = max(np.hypot(vx, vy), 1e-6)
            # theta is the normal direction, hence rotate it for a tangent.
            t1 = np.array([-np.sin(theta[y1, x1]), np.cos(theta[y1, x1])])
            t2 = np.array([-np.sin(theta[y2, x2]), np.cos(theta[y2, x2])])
            direction = np.array([vx, vy]) / norm
            agreement = max(abs(float(direction @ t1)), abs(float(direction @ t2)))
            if agreement < tangent_cos:
                continue
            line = np.linspace(0, 1, int(round(d)) + 1)
            xx = np.rint(x1 + line * (x2-x1)).astype(int)
            yy2 = np.rint(y1 + line * (y2-y1)).astype(int)
            if np.all(mask[yy2, xx]) and float(np.mean(score[yy2, xx])) > .02:
                best = j
                out[yy2, xx] = True
                break
        if best is not None:
            used.update((i, best))
    return out


def _strokes_from_mask(binary: np.ndarray, score: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes: list[np.ndarray] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        strokes.append(contour[:, 0, :].astype(np.float32))
    return strokes


def generate_lines(features, mask: np.ndarray, config: LineConfig | None = None) -> LinePlan:
    cfg = config or LineConfig()
    reliability = .35 + .65 * np.clip(features.coherence, 0, 1)
    edge = robust01(features.edge, mask) * reliability
    dark = robust01(features.dark, mask) * (.55 + .45 * reliability)
    nms_edge = oriented_nms(edge, features.theta)
    nms_dark = oriented_nms(dark, features.theta)
    near_dark = cv2.dilate(nms_dark, np.ones((3, 3), np.uint8))
    # Dark creases are authoritative; reduce a nearby edge gently instead of deleting it.
    soft_edge = nms_edge * (1.0 - .45 * np.clip(near_dark, 0, 1))
    score = np.maximum(nms_dark, .72 * soft_edge)
    retained = hysteresis(score, cfg.low, cfg.high, mask)
    skeleton = skeletonize(retained)
    bridged = bridge_short_gaps(skeleton, features.theta, score, mask, cfg.join_gap, cfg.tangent_cos)
    return LinePlan(edge, dark, nms_edge, nms_dark, score, retained, skeleton, bridged,
                    _strokes_from_mask(bridged, score))
