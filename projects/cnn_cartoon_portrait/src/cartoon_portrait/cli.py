"""命令行编排：交互和文件保存位于纯数组计算函数之外。"""

import argparse
import hashlib
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .artifacts import export_results, load_labels, load_mask, make_demo
from .colors import ColorConfig
from .features import FeatureConfig, FusionConfig
from .lines import LineConfig
from .pipeline import (
    Config, Labels, Selection, compose_rgba, extract_lines, quantize_colors,
    read_image, segment_person, smooth_foreground,
)
from .selection import ViewConfig, collect_selection
from .workflow import StyleConfig, stylize

PROJECT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    defaults = StyleConfig()
    result = argparse.ArgumentParser(
        description="自拍漫画化：逐通道方向卷积 → 结构融合 → 保边色块与线条。",
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
    result.add_argument("--pipeline", choices=("structure", "baseline"), default="structure", help="结构融合路线；baseline 为第一版对照。")
    result.add_argument("--sigma", type=float, help="预平滑尺度，单位原图像素；结构路线默认 0.6，第一版 1.5；0 关闭。")
    result.add_argument("--levels", type=int, default=6, help="Lab 明度档数，至少 2。")
    result.add_argument("--amount", type=float, help="明度分层强度；结构路线默认 0.35，第一版 0.65。")
    result.add_argument("--threshold", "--low-threshold", dest="threshold", type=float, help="结构路线低阈值默认 0.025；第一版起始阈值 0.035。")
    result.add_argument("--softness", type=float, help="着墨过渡宽度；结构路线默认 0.12，第一版 0.08。")
    result.add_argument("--strength", type=float, help="描线合成强度 [0,1]；结构路线默认 0.55，第一版 0.8。")
    structure = result.add_argument_group("结构路线参数（--pipeline structure）")
    structure.add_argument("--operator", choices=("sobel", "scharr"), default="sobel")
    structure.add_argument("--channel-weights", type=float, nargs=3, default=[1 / 3] * 3, metavar=("R", "G", "B"), help="非负通道权重，和为 1。")
    structure.add_argument("--high-threshold", type=float, default=defaults.lines.high, help="连通线条中的强响应阈值。")
    structure.add_argument("--ink-gamma", type=float, default=defaults.lines.gamma, help="着墨曲线指数，范围 (0,1]。")
    structure.add_argument("--no-thin", action="store_true", help="关闭方向细化，作线条对照。")
    structure.add_argument("--outline-width", type=int, default=0, help="内轮廓宽度（原图像素），0 关闭。")
    structure.add_argument("--transition", type=float, default=0.25, help="软明度过渡半宽相对档距，范围 (0,0.5]。")
    structure.add_argument("--detail", type=float, default=0.35, help="保留的明度细节比例 [0,1]。")
    structure.add_argument("--luma-smoothness", type=float, default=8.0, help="WLS 基础明度平滑强度。")
    structure.add_argument("--chroma-smoothness", type=float, default=1.5, help="WLS 色度平滑强度。")
    structure.add_argument("--tau-lightness", type=float, default=8.0, help="边界权重的 Lab L* 差异尺度。")
    structure.add_argument("--tau-chroma", type=float, default=12.0, help="边界权重的 Lab 色度差异尺度。")
    structure.add_argument("--tau-gradient", type=float, default=0.12, help="边界权重的归一化梯度尺度。")
    result.add_argument("--preview-scale", type=float, default=4.0, help="交互预览最大倍率；不改变计算尺寸。")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.demo and (args.rect or args.labels or args.mask or args.interactive):
        arguments.error("--demo 自带矩形，不与其他分割输入同时使用。")
    try:
        defaults = StyleConfig()
        if args.pipeline == "baseline":
            cfg: Config | StyleConfig = Config(
                sigma=1.5 if args.sigma is None else args.sigma,
                levels=args.levels, amount=.65 if args.amount is None else args.amount,
                threshold=.035 if args.threshold is None else args.threshold,
                softness=.08 if args.softness is None else args.softness,
                strength=.8 if args.strength is None else args.strength,
            )
        else:
            cfg = StyleConfig(
                features=FeatureConfig(defaults.features.sigma if args.sigma is None else args.sigma, args.operator),
                fusion=FusionConfig(tuple(args.channel_weights)),
                colors=ColorConfig(
                    levels=args.levels, amount=defaults.colors.amount if args.amount is None else args.amount,
                    transition=args.transition, detail=args.detail,
                    luma_smoothness=args.luma_smoothness, chroma_smoothness=args.chroma_smoothness,
                    tau_lightness=args.tau_lightness, tau_chroma=args.tau_chroma,
                    tau_gradient=args.tau_gradient,
                ),
                lines=LineConfig(
                    low=defaults.lines.low if args.threshold is None else args.threshold,
                    high=args.high_threshold, width=defaults.lines.width if args.softness is None else args.softness,
                    gamma=args.ink_gamma, thin=not args.no_thin, outline_width=args.outline_width,
                ),
                strength=defaults.strength if args.strength is None else args.strength,
            )
        cfg.validate()
        view_config = ViewConfig(max_scale=args.preview_scale)
        view_config.validate()
        if args.iterations < 1:
            raise ValueError("iterations 至少为 1。")
        directory = args.output or PROJECT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        directory = directory.resolve()
        if directory.exists():
            raise ValueError(f"输出目录已经存在，请使用新目录：{directory}")
        metadata: dict[str, object] = {
            "schema_version": 2, "pipeline": args.pipeline, "grabcut_seed": 0,
            "view_config": asdict(view_config),
        }
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
            mask, labels = collect_selection(image, iterations=args.iterations, view_config=view_config)
            metadata["selection"] = "interactive labels"
            metadata["iterations"] = args.iterations
        if demo_truth is not None:
            union = np.count_nonzero(mask | demo_truth)
            metadata["synthetic_mask_iou"] = float(np.count_nonzero(mask & demo_truth) / union)

        started = perf_counter()
        result = None
        if isinstance(cfg, StyleConfig):
            print("正在计算逐通道结构、保边色块和筛选线条……", flush=True)
            result = stylize(image, mask, cfg)
            base, colors, lines, rgba = result.features.guide, result.colors, result.lines, result.rgba
            print(f"保边平滑最大相对残差：{result.color_debug.solver_residual:.2g}", flush=True)
        else:
            print("正在运行第一版高斯、量化和灰度 Sobel 对照……", flush=True)
            base = smooth_foreground(image, mask, sigma=cfg.sigma)
            colors = quantize_colors(base, levels=cfg.levels, amount=cfg.amount)
            lines = extract_lines(base, mask, threshold=cfg.threshold, softness=cfg.softness)
            rgba = compose_rgba(colors, lines, mask.astype(np.float32), strength=cfg.strength)
        metadata["processing_seconds"] = perf_counter() - started
        output = export_results(
            directory, image, mask, base, colors, lines, rgba, cfg, metadata, labels, result=result,
        )
        if demo_truth is not None:
            print(f"合成图分割 IoU：{metadata['synthetic_mask_iou']:.4f}（不代表真实自拍效果）。")
        print(f"已导出透明漫画：{output}")
        print(f"步骤总览：{directory / 'overview.png'}")
        print(f"参数记录：{directory / 'parameters.json'}")
        return 0
    except (ValueError, OSError, ArithmeticError, cv2.error) as error:
        print(f"处理未完成：{error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
