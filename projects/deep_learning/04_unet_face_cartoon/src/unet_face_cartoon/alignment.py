"""本地 SFD + 2D FAN；权重先校验，不在推理过程中自动下载。"""

from pathlib import Path

import face_alignment
import numpy as np
from numpy.typing import NDArray
import torch

from .model_runtime import verify_weight


class NoFaceError(ValueError):
    pass


class FaceAligner:
    def __init__(self, root: Path, device: str):
        detector_path, sfd_hash = verify_weight(root, "sfd")
        _, fan_hash = verify_weight(root, "fan")
        torch.hub.set_dir(str(root / "models/torch/hub"))
        self.net = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device=device,
            face_detector="sfd", flip_input=False, compile=False,
            face_detector_kwargs={"path_to_detector": str(detector_path)},
        )
        torch.backends.cudnn.benchmark = False
        self.metadata = {"sfd_sha256": sfd_hash, "fan_sha256": fan_hash, "flip_input": False}

    def landmarks(self, rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
        with torch.inference_mode():
            points = self.net.get_landmarks_from_image(rgb)
        if points is None or len(points) == 0:
            raise NoFaceError("未检测到有效人脸，请使用清晰的单人照片。")
        if len(points) != 1:
            raise NoFaceError(f"检测到 {len(points)} 张人脸；本版只接受单人照片。")
        return np.asarray(points[0], dtype=np.float64)
