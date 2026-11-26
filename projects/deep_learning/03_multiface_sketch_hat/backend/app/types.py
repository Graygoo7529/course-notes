"""Small immutable records shared by the detector and renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceDetection:
    """A face detection in the coordinate system of the input RGB frame."""

    bbox: BBox
    score: float
    landmarks5: Tuple[Point, Point, Point, Point, Point]

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def eye_angle_degrees(self) -> float:
        """Roll estimated from the first two landmarks (left/right eye)."""

        left, right = self.landmarks5[:2]
        import math

        return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))

    def translated(self, dx: float, dy: float) -> "FaceDetection":
        """Return the same detection in a crop whose origin moved by (dx, dy)."""

        x1, y1, x2, y2 = self.bbox
        points: Tuple[Point, Point, Point, Point, Point] = (
            (self.landmarks5[0][0] - dx, self.landmarks5[0][1] - dy),
            (self.landmarks5[1][0] - dx, self.landmarks5[1][1] - dy),
            (self.landmarks5[2][0] - dx, self.landmarks5[2][1] - dy),
            (self.landmarks5[3][0] - dx, self.landmarks5[3][1] - dy),
            (self.landmarks5[4][0] - dx, self.landmarks5[4][1] - dy),
        )
        return FaceDetection((x1 - dx, y1 - dy, x2 - dx, y2 - dy), self.score, points)
