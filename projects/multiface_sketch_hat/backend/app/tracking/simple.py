"""Small IoU/center tracker for a webcam demo.

It deliberately provides stable short-lived IDs, not identity recognition. A
crossing or a long occlusion may create a new ID; the renderer must tolerate
that explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..types import BBox, FaceDetection


def _iou(left: BBox, right: BBox) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 1e-6 else 0.0


def _center_distance(left: BBox, right: BBox) -> float:
    lx, ly = (left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0
    rx, ry = (right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0
    diagonal = max(math.hypot(left[2] - left[0], left[3] - left[1]), 1.0)
    return math.hypot(lx - rx, ly - ry) / diagonal


@dataclass
class TrackedFace:
    track_id: int
    detection: FaceDetection
    missed: int = 0


class SimpleTracker:
    """Greedy association suitable for a low-rate educational camera preview."""

    def __init__(self, max_missed: int = 8, iou_threshold: float = 0.15, distance_threshold: float = 1.25) -> None:
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self._next_id = 1
        self._tracks: dict[int, TrackedFace] = {}

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        unmatched_tracks = set(self._tracks)
        result: list[TrackedFace] = []
        for detection in detections:
            best_id: int | None = None
            best_score = -1.0
            for track_id in unmatched_tracks:
                previous = self._tracks[track_id].detection
                iou = _iou(previous.bbox, detection.bbox)
                distance = _center_distance(previous.bbox, detection.bbox)
                score = iou + max(0.0, 0.35 - distance * 0.2)
                if (iou >= self.iou_threshold or distance <= self.distance_threshold) and score > best_score:
                    best_id, best_score = track_id, score
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            else:
                unmatched_tracks.remove(best_id)
            tracked = TrackedFace(best_id, detection, missed=0)
            self._tracks[best_id] = tracked
            result.append(tracked)
        for track_id in unmatched_tracks:
            self._tracks[track_id].missed += 1
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if track.missed <= self.max_missed
        }
        return sorted(result, key=lambda item: (item.detection.x1, item.detection.y1))
