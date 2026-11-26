"""可观察的单人照片管线；解析一次，复用同一裁剪与掩膜进行两分支对照。"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .alignment import FaceAligner
from .artifacts import comparison, save_png
from .geometry import align_crop, limit_image
from .image_contracts import read_rgb
from .model_runtime import PortraitModels
from .regions import PALETTE, edit_regions, white_input


@dataclass(frozen=True)
class Controls:
    skin_strength: float = 0.5
    hair_color: tuple[int, int, int] | None = None
    hair_strength: float = 0

    def __post_init__(self) -> None:
        if not 0 <= self.skin_strength <= 1 or not 0 <= self.hair_strength <= 1:
            raise ValueError("处理强度必须在 0～1 之间")
        if self.hair_strength and self.hair_color is None:
            raise ValueError("请在启用发色强度时同时提供 --hair-color")


class Pipeline:
    def __init__(self, root: Path, device: str, fetch_source: bool = False):
        self.device = device
        self.aligner = FaceAligner(root, device)
        self.models = PortraitModels(root, device, fetch_source)
        self.metadata = {
            "device": device, "torch": str(torch.__version__),
            "models": self.models.metadata, "alignment": self.aligner.metadata,
            "foreground_rule": "labels 1..15; keep component with most skin pixels",
            "postprocess": "denormalize, clip, RGB; no input mask pasted onto output",
        }

    def process(self, path: Path, directory: Path, controls: Controls, source: dict | None = None) -> dict:
        directory.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        input_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if source and source["sha256"] != input_digest:
            raise ValueError("公开样例与来源清单的 SHA256 不符，停止生成以免错误署名。")
        rgb = read_rgb(path)
        working, original_to_working = limit_image(rgb)
        save_png(directory / "input.png", working)
        points = self.aligner.landmarks(working)
        portrait = align_crop(working, points, original_to_working)
        labels = self.models.parse(portrait.rgb)
        edited, masks = edit_regions(portrait.rgb, labels, **asdict(controls))
        if float(masks["foreground"].mean()) < 0.01:
            raise ValueError("解析得到的头像前景过小，停止生成，请检查人像构图。")
        baseline_input = white_input(portrait.rgb, masks["foreground"])
        edited_input = white_input(edited, masks["foreground"])
        baseline = self.models.cartoon(baseline_input)
        identical = np.array_equal(baseline_input, edited_input)
        cartoon = baseline.copy() if identical else self.models.cartoon(edited_input)
        difference = np.abs(cartoon.astype(np.int16) - baseline.astype(np.int16))
        images = {
            "portrait": portrait.rgb, "labels": labels,
            "segmentation": PALETTE[labels],
            "overlay": np.rint(portrait.rgb * 0.65 + PALETTE[labels] * 0.35).astype(np.uint8),
            "edited_portrait": edited, "baseline_input": baseline_input,
            "edited_input": edited_input, "baseline": baseline, "cartoon": cartoon,
            "output_difference": np.clip(difference * 4, 0, 255).astype(np.uint8),
        }
        for key, image in images.items():
            save_png(directory / f"{key}.png", image)
        for key, mask in masks.items():
            save_png(directory / f"{key}.png", np.rint(mask * 255).astype(np.uint8))
        comparison([working, PALETTE[labels], baseline, cartoon], directory / "comparison.png")
        if self.device == "cuda":
            torch.cuda.synchronize()
        warnings = [
            "原始生成分辨率为 256×256。",
            "原模型对不同人群覆盖有限，须结合图像判断失真。",
            "区域控制不保证改善质量，也不保证只改变对应输出区域。",
        ]
        if identical:
            warnings.append("处理后的生成输入与基线完全相同，复用基线输出。")
        removed = int((masks["foreground_raw"] - masks["foreground"]).sum())
        if removed:
            warnings.append(f"清理 {removed} 个孤立前景像素；独立饰品也可能被去掉，请核对原始掩膜。")
        if any(np.any(edge) for edge in (
            masks["foreground"][0], masks["foreground"][-1],
            masks["foreground"][:, 0], masks["foreground"][:, -1],
        )):
            warnings.append("头像前景接触裁剪边界，可能存在截发；原图缺失的部分无法恢复。")
        record = {
            "id": directory.name, "status": "ok", "input_name": path.name,
            "input_sha256": input_digest,
            "controls": asdict(controls), "source": source,
            "geometry": {
                "original_size_wh": [rgb.shape[1], rgb.shape[0]],
                "working_size_wh": [working.shape[1], working.shape[0]],
                "original_to_portrait": portrait.original_to_portrait.tolist(),
                "portrait_to_original": np.linalg.inv(portrait.original_to_portrait).tolist(),
                "landmarks_512": portrait.landmarks.tolist(),
                "crop_box": portrait.crop_box,
                "rotation_degrees": portrait.rotation_degrees,
                "coordinate_frame": "EXIF-corrected original RGB; pixel centers",
            },
            "label_counts": np.bincount(labels.ravel(), minlength=19).tolist(),
            "foreground_removed_pixels": removed,
            "input_difference_mae": float(np.abs(edited_input.astype(float) - baseline_input).mean()),
            "output_difference_mae": float(difference.mean()),
            "changed_output_fraction": float(np.any(difference != 0, axis=2).mean()),
            "parser_calls": 1, "generator_calls": 1 if identical else 2,
            "elapsed_seconds_including_io": time.perf_counter() - start,
            "warnings": warnings,
        }
        (directory / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
        return record
