"""命令行编排：交互和文件保存位于纯数组计算函数之外。"""

import argparse
import hashlib
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .artifacts import export_results, load_labels, load_mask, make_demo
from .pipeline import (
    Config, Labels, Selection, compose_rgba, extract_lines, quantize_colors,
    read_image, segment_person, smooth_foreground,
)
from .selection import collect_selection

PROJECT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="自拍去背景并漫画化：GrabCut + 高斯卷积 + Lab 量化 + Sobel。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="输入照片路径；读取后按 EXIF 校正方向。")
    source.add_argument("--demo", action="store_true", help="处理内置合成图，验证管道。")
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--rect", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="原图坐标中的人物矩形。")
    selection.add_argument("--interactive", action="store_true", help="打开框选/前景/背景笔划窗口；无其他选择时默认启用。")
    selection.add_argument("--labels", type=Path, help="复用单通道 GrabCut 标签 PNG（像素值 0/1/2/3）。")
    selection.add_argument("--mask", type=Path, help="复用已确认的二值遮罩（0/255 PNG），跳过 GrabCut。")
    result.add_argument("--output", type=Path, help="新建的结果目录；默认放入项目 outputs 下。")
    result.add_argument("--iterations", type=int, default=5, help="GrabCut 迭代次数。")
    result.add_argument("--sigma", type=float, default=1.5, help="高斯尺度，单位为原图像素；0 关闭平滑。")
    result.add_argument("--levels", type=int, default=6, help="Lab 明度档数，至少 2。")
    result.add_argument("--amount", type=float, default=0.65, help="明度量化混合强度 [0,1]。")
    result.add_argument("--threshold", type=float, default=0.035, help="Sobel/8 响应的起始阈值。")
    result.add_argument("--softness", type=float, default=0.08, help="描线从 0 过渡到 1 的响应宽度，必须 >0。")
    result.add_argument("--strength", type=float, default=0.8, help="黑色描线的合成强度 [0,1]。")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.demo and (args.rect or args.labels or args.mask or args.interactive):
        arguments.error("--demo 自带矩形，不与其他分割输入同时使用。")
    cfg = Config(**{field.name: getattr(args, field.name) for field in fields(Config)})
    try:
        cfg.validate()
        if args.iterations < 1:
            raise ValueError("iterations 至少为 1。")
        directory = args.output or PROJECT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        directory = directory.resolve()
        if directory.exists():
            raise ValueError(f"输出目录已经存在，请使用新目录：{directory}")
        metadata: dict[str, object] = {"schema_version": 1, "grabcut_seed": 0}
        labels: Labels | None = None
        demo_truth = None
        if args.demo:
            image, demo_truth, rect = make_demo()
            metadata["source"] = "synthetic demo; not a selfie or an effect-quality benchmark"
            selection = Selection(rect=rect, iterations=args.iterations)
        else:
            image = read_image(args.input)
            metadata["source"] = str(args.input.resolve())
            metadata["source_file_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
            selection = None
            if args.rect:
                rect = tuple(args.rect)
                selection = Selection(rect=(rect[0], rect[1], rect[2], rect[3]), iterations=args.iterations)
            elif args.labels:
                labels = load_labels(args.labels, image.shape)
                selection = Selection(labels=labels, iterations=args.iterations)
                metadata["labels_source"] = str(args.labels.resolve())

        print(f"输入尺寸：{image.shape[1]} × {image.shape[0]}；保持原始处理尺寸。", flush=True)
        if selection is not None:
            print("正在计算 GrabCut 人物遮罩……", flush=True)
            mask = segment_person(image, selection)
            metadata["selection"] = "labels" if selection.labels is not None else "rectangle"
            metadata["iterations"] = args.iterations
            if selection.rect is not None:
                x, y, width, height = selection.rect
                metadata["rectangle"] = list(selection.rect)
                labels = np.full(image.shape[:2], cv2.GC_BGD, dtype=np.uint8)
                labels[y:y + height, x:x + width] = cv2.GC_PR_FGD
        elif args.mask:
            mask = load_mask(args.mask, image.shape)
            metadata["selection"] = "existing binary mask; GrabCut skipped"
            metadata["mask_source"] = str(args.mask.resolve())
        else:
            mask, labels = collect_selection(image, iterations=args.iterations)
            metadata["selection"] = "interactive labels"
            metadata["iterations"] = args.iterations
        if demo_truth is not None:
            union = np.count_nonzero(mask | demo_truth)
            metadata["synthetic_mask_iou"] = float(np.count_nonzero(mask & demo_truth) / union)

        print("正在计算卷积平滑、色块和线条……", flush=True)
        base = smooth_foreground(image, mask, sigma=cfg.sigma)
        colors = quantize_colors(base, levels=cfg.levels, amount=cfg.amount)
        lines = extract_lines(base, mask, threshold=cfg.threshold, softness=cfg.softness)
        rgba = compose_rgba(colors, lines, mask.astype(np.float32), strength=cfg.strength)
        output = export_results(directory, image, mask, base, colors, lines, rgba, cfg, metadata, labels)
        if demo_truth is not None:
            print(f"合成图分割 IoU：{metadata['synthetic_mask_iou']:.4f}（不代表真实自拍效果）。")
        print(f"已导出透明漫画：{output}")
        print(f"步骤总览：{directory / 'overview.png'}")
        print(f"参数记录：{directory / 'parameters.json'}")
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print(f"处理未完成：{error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
