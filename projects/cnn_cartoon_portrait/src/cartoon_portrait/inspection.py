"""从 v0.7 输出建立可复核的观察记录；不覆盖原结果、不运行人物分割。"""

import argparse
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .feature_bank import BankConfig, FeatureBank
from .features import ChannelFeatures, FusionConfig, fuse_features
from .lines import thin_edges
from .pipeline import FloatImage, Mask, _extend_foreground, masked_gaussian, read_image
from .strokes import StrokeConfig, trace_line_features, trace_paths


def tensor_stages(gx: FloatImage, gy: FloatImage, mask: Mask, config: BankConfig
                  ) -> tuple[NDArray[np.float64], FloatImage, FloatImage]:
    """K,H,W,C → 每通道三个二次项 K,H,W,C,3 → PW → 空间平滑。"""
    x, y = gx.astype(np.float64), gy.astype(np.float64)
    terms = np.stack((x * x, x * y, y * y), axis=-1)
    raw = np.einsum("khwct,c->khwt", terms, np.asarray(config.weights)).astype(np.float32)
    smoothed = np.stack([masked_gaussian(t, mask, config.direction_sigma) for t in raw])
    return terms, raw, smoothed


def gaussian_stages(image: FloatImage, mask: Mask, sigma: float) -> dict[str, FloatImage]:
    """展示 G*(MI)/(G*M) 的分子、分母；中途不做归一化。"""
    radius = max(1, int(np.ceil(3 * sigma)))
    kernel = cv2.getGaussianKernel(radius * 2 + 1, sigma, cv2.CV_32F)
    identity = np.ones(1, np.float32)
    weight = mask.astype(np.float32)
    numerator_x = cv2.sepFilter2D(image * weight[..., None], -1, kernel, identity)
    denominator_x = cv2.sepFilter2D(weight, -1, kernel, identity)
    numerator = cv2.sepFilter2D(numerator_x, -1, identity, kernel)
    denominator = cv2.sepFilter2D(denominator_x, -1, identity, kernel)
    guide = np.zeros_like(image)
    valid = mask & (denominator > np.finfo(np.float32).eps)
    guide[valid] = numerator[valid] / denominator[valid, None]
    guide[mask & ~valid] = image[mask & ~valid]
    return {name: value.astype(np.float32, copy=False) for name, value in
            {"numerator_x": numerator_x, "denominator_x": denominator_x,
             "numerator_xy": numerator, "denominator_xy": denominator, "guide": guide}.items()}


def gradient_kernel(operator: str, axis: str) -> FloatImage:
    smooth = np.array([1, 2, 1] if operator == "sobel" else [3, 10, 3], np.float32)
    kernel = np.outer(smooth, np.array([-1, 0, 1], np.float32))
    kernel /= 8 if operator == "sobel" else 32
    return kernel if axis == "x" else kernel.T


def reconstruct_bank(arrays: dict[str, NDArray], config: BankConfig) -> FeatureBank:
    fine = ChannelFeatures(arrays["guides"][0], arrays["gx"][0], arrays["gy"][0],
                           np.hypot(arrays["gx"][0], arrays["gy"][0]))
    return FeatureBank(config.scales, arrays["guides"], arrays["gx"], arrays["gy"], arrays["edges"],
                       arrays["dog"], arrays["theta"], arrays["coherence"], arrays["direction_valid"],
                       arrays["lab"], arrays["lab"][..., 0] / np.float32(100), arrays["chroma_edge"],
                       fuse_features(fine, FusionConfig(config.weights)))


def path_statistics(skeleton: Mask, accepted: Mask, protected: Mask, min_length: float
                    ) -> dict[str, object]:
    lengths, kept_lengths = [], []
    endpoints = set()
    for path, closed, start_tip, end_tip in trace_paths(skeleton):
        length = float(np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1).sum())
        lengths.append(length)
        if length >= min_length or protected[path[:, 1], path[:, 0]].any():
            kept_lengths.append(length)
        if not closed:
            if start_tip:
                endpoints.add(tuple(path[0]))
            if end_tip:
                endpoints.add(tuple(path[-1]))
    return {"graph_paths": len(lengths), "paths_below_min_length": sum(v < min_length for v in lengths),
            "kept_internal_paths": len(kept_lengths), "kept_paths_under_5px": sum(v < 5 for v in kept_lengths),
            "kept_median_length_px": float(np.median(kept_lengths)) if kept_lengths else 0,
            "graph_endpoints": len(endpoints), "lengths_px": lengths,
            "skeleton_components": int(cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0] - 1),
            "accepted_components": int(cv2.connectedComponents(accepted.astype(np.uint8), connectivity=8)[0] - 1)}


def display_plane(value: NDArray, mask: Mask, limit: float, signed: bool) -> NDArray[np.uint8]:
    """同族特征共用量程；有符号响应蓝负红正，背景独立标白。"""
    normalized = np.clip(value.astype(np.float64) / max(limit, 1e-12), -1 if signed else 0, 1)
    if signed:
        magnitude = np.abs(normalized)[..., None]
        color = np.where((normalized >= 0)[..., None], [198, 43, 57], [38, 95, 178])
        rgb = (1 - magnitude) * 245 + magnitude * color
    else:
        rgb = np.repeat((255 * (1 - normalized))[..., None], 3, axis=2)
    rgb[~mask] = 255
    return np.rint(rgb).astype(np.uint8)


def _encoded(array: NDArray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def export_inspection(run: Path, output: Path, crop: tuple[int, int, int, int] | None = None) -> Path:
    record = json.loads((run / "parameters.json").read_text(encoding="utf-8"))
    if record.get("pipeline") != "comic" or record.get("schema_version") != 3:
        raise ValueError("观察器目前读取 v0.7 comic 的 schema_version=3 结果。")
    if output.exists():
        raise ValueError("观察目录已存在，请使用新的 --output；不会覆盖已有记录。")
    feature_config = BankConfig(**record["config"]["features"])
    stroke_config = StrokeConfig(**record["config"]["strokes"])
    feature_config.validate()
    stroke_config.validate()
    with np.load(run / "features.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mask = arrays["mask"]
    image = read_image(run / "input.png")
    height, width = mask.shape
    if crop is None:
        # 数值浏览限定窗口，PNG 和 NPZ 仍保留全图。窗口不缩放，保证卷积演示坐标准确。
        yy, xx = np.nonzero(mask)
        cw, ch = min(256, width), min(256, height)
        cx = min(max(0, int(np.median(xx)) - cw // 2), width - cw)
        cy = min(max(0, int(np.median(yy)) - ch // 2), height - ch)
        crop = (cx, cy, cw, ch)
    cx, cy, cw, ch = crop
    if min(cx, cy) < 0 or min(cw, ch) < 1 or cw > 256 or ch > 256 or cx + cw > width or cy + ch > height:
        raise ValueError("--crop X Y W H 必须在原图内，宽高各不超过 256；不对数值窗口降采样。")
    view = np.s_[cy:cy + ch, cx:cx + cw]
    bank = reconstruct_bank(arrays, feature_config)
    lines, stages = trace_line_features(bank, mask, arrays["protected"], arrays["suppressed"], stroke_config)
    # 对照存档，防止新版代码悄悄把重算结果冒充旧处理过程。
    errors = {}
    for key, actual in {"line_edge": lines.edge, "line_dark": lines.dark, "line_score": lines.score,
                        "retained": lines.retained, "skeleton": lines.skeleton}.items():
        error = float(np.max(np.abs(actual.astype(float) - arrays[key].astype(float))))
        errors[key] = error
        if error > 2e-6:
            raise ValueError(f"{key} 与存档不一致（最大差 {error:g}），不能按当前公式解释该结果。")
    terms, tensor, smoothed = tensor_stages(bank.gx, bank.gy, mask, feature_config)
    stages["thinning_removed"] = lines.retained & ~lines.skeleton
    stages["path_removed"] = lines.skeleton & ~arrays["accepted"]
    stages["accepted"] = arrays["accepted"]
    # 一个变量的对照：先用未截断幅值做 NMS，再截断；后续绘制不使用此结果。
    alternatives = {}
    for name in ("dark", "edge"):
        alternative = np.clip(thin_edges(stages[f"{name}_normalized"].astype(np.float32), lines.theta,
                                          lines.coherence, mask), 0, 1)
        alternatives[f"{name}_nms_before_clip"] = alternative
        alternatives[f"{name}_nms_difference"] = alternative - stages[f"{name}_nms"]
    stats = path_statistics(lines.skeleton, arrays["accepted"], arrays["protected"], stroke_config.min_length)
    def correlation(a: NDArray, b: NDArray) -> float | None:
        if a.size < 2 or min(float(np.std(a)), float(np.std(b))) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    magnitudes = np.hypot(bank.gx[0], bank.gy[0])
    gray_weights = np.array([.299, .587, .114], np.float32)
    gray_strength = np.hypot(bank.gx[0] @ gray_weights, bank.gy[0] @ gray_weights) * bank.scales[0]
    stats["rgb_magnitude_correlations"] = [[correlation(magnitudes[..., a][mask], magnitudes[..., b][mask])
                                           for b in range(3)] for a in range(3)]
    stats["tensor_weighted_gray_strength_correlation"] = correlation(bank.edges[0][mask], gray_strength[mask])
    stats.update({"retained_pixels": int(lines.retained.sum()), "skeleton_pixels": int(lines.skeleton.sum()),
                  "thinning_removed_pixels": int(stages["thinning_removed"].sum()),
                  "path_removed_pixels": int(stages["path_removed"].sum()),
                  "ridge_removed_pixels": int(stages["ridge_removed"].sum()),
                  "ridge_removed_above_low": int((stages["ridge_removed"].astype(bool) &
                                                  (stages["edge_supported"] * .65 >= stroke_config.low)).sum()),
                  "replay_max_errors": errors,
                  "nms_order_changed_pixels": {name: int((np.abs(alternatives[f"{name}_nms_difference"]) > 1e-6).sum())
                                               for name in ("edge", "dark")}})
    gaussians = [gaussian_stages(image, mask, sigma) for sigma in bank.scales]
    stats["saved_input_gaussian_max_error"] = max(float(np.max(np.abs(g["guide"] - bank.guides[k])))
                                                   for k, g in enumerate(gaussians))
    output.mkdir(parents=True)
    maps = output / "maps"
    maps.mkdir()
    np.savez_compressed(output / "trace.npz", allow_pickle=False, tensor_terms=terms, tensor_raw=tensor, tensor_smoothed=smoothed,
                        **stages, **alternatives,
                        **{f"gaussian_{k}_{name}": value for k, g in enumerate(gaussians) for name, value in g.items()})
    for name in ("input.png", "preview_light.png", "lines.png", "regions.png", "flat_colors.png", "colors.png"):
        shutil.copyfile(run / name, output / name)
    (output / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    catalog: list[dict[str, object]] = []
    pixel_data: dict[str, str] = {}

    def add(key: str, value: NDArray, title: str, group: str, explanation: str, unit: str,
            limit: float = 1, signed: bool = False, scale: int = -1, channel: int = -1,
            valid: Mask | None = None) -> None:
        filename = key + ".png"
        display_mask = mask if valid is None else valid
        Image.fromarray(display_plane(value, display_mask, limit, signed)).save(maps / filename)
        catalog.append({"key": key, "file": "maps/" + filename, "title": title, "group": group,
                        "explanation": explanation, "unit": unit, "limit": float(limit), "signed": signed,
                        "scale": scale, "channel": channel})
        pixel_data[key] = _encoded(value[view])

    derivative_limit = max(.01, float(max(np.abs(bank.gx).max(), np.abs(bank.gy).max())))
    term_limit = max(1e-6, float(np.abs(terms).max()))
    edge_limit = max(.01, float(bank.edges.max()))
    dog_limit = max(.01, float(np.abs(bank.dog).max()))
    all_valid = np.ones_like(mask)
    padded_guides = []
    for k, sigma in enumerate(bank.scales):
        prefix = f"s{k}_"
        padded = cv2.copyMakeBorder(_extend_foreground(bank.guides[k], mask), 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
        padded_guides.append(_encoded(padded[cy:cy + ch + 2, cx:cx + cw + 2]))
        for c, color in enumerate(("R", "G", "B")):
            for key, title in (("numerator_x", "横向高斯：分子"), ("numerator_xy", "再纵向高斯：分子")):
                add(f"{prefix}{key}_{c}", gaussians[k][key][..., c], f"{title} · {color}", "gaussian",
                    "从保存的 input.png 重算 MI 的可分离卷积；边界处随后需要除以 G*M。尚未归一化的横向结果不等于最终底色。",
                    "RGB [0,1] 加权和", scale=k, channel=c, valid=all_valid)
            add(f"{prefix}guide_{c}", bank.guides[k, ..., c], f"归一化引导图 · {color}", "gaussian",
                "直接读取生产管道保存的逐通道引导图。G*(MI)/(G*M)，背景为无效占位。", "RGB [0,1]", scale=k, channel=c)
            for axis, value in (("x", bank.gx), ("y", bank.gy)):
                add(f"{prefix}g{axis}_{c}", value[k, ..., c], f"{color} 通道 g{axis}", "channels",
                    "每个通道独立做方向互相关。蓝负、红正；梯度方向为法向，不是笔画走向。各尺度与通道共用显示量程。",
                    "RGB / 原图像素", derivative_limit, True, k, c)
            add(f"{prefix}magnitude_{c}", np.hypot(bank.gx[k, ..., c], bank.gy[k, ..., c]) * sigma,
                f"{color} 通道尺度归一幅值", "channels", "σ√(gx²+gy²)，非线性幅值；比较尺度时共用显示上限。",
                "σ × 梯度幅值", max(.01, float((np.hypot(bank.gx, bank.gy) * np.asarray(bank.scales)[:, None, None, None]).max())),
                scale=k, channel=c)
            for t, label in enumerate(("xx", "xy", "yy")):
                add(f"{prefix}term_{label}_{c}", terms[k, ..., c, t], f"{color}：{label} 二次项", "fusion",
                    "gx²、gx·gy、gy² 是逐点非线性特征，尚未乘通道权重。xy 可以为负，不能当作普通正响应图。",
                    "梯度²", term_limit, t == 1, k, c)
        for key, title in (("denominator_x", "横向前景权重"), ("denominator_xy", "横纵前景权重")):
            add(prefix + key, gaussians[k][key], title, "gaussian", "对 M 使用相同卷积；在最终二维卷积之后做分子/分母归一化。",
                "权重 [0,1]", scale=k, valid=all_valid)
        for t, label in enumerate(("xx", "xy", "yy")):
            for stem, value, title in (("j", tensor, "PW 融合"), ("js", smoothed, "张量空间平滑")):
                add(f"{prefix}{stem}{label}", value[k, ..., t], f"{title} J{label}", "fusion",
                    "PW 将三个通道同类二次项加权汇总。原始 J 用来求强度；另一路对 J 做高斯平滑后求稳定方向，二者不能混用。",
                    "梯度²", term_limit, t == 1, k)
        add(prefix + "edge", bank.edges[k], "融合结构强度", "fusion", "σ√λmax(J)，使用空间平滑前的 J。细尺度进入边界候选；跨尺度最大值只为候选提供支持。",
            "σ × 梯度幅值", edge_limit, scale=k)
        add(prefix + "coherence", bank.coherence[k], "方向一致性 q", "fusion",
            "(λmax−λmin)/(λmax+λmin)，使用平滑后的 J。高一致性不等于强边缘，也不等于五官语义概率。",
            "[0,1]", scale=k)
        add(prefix + "dog", bank.dog[k], "DoG：宽高斯减窄高斯", "fusion",
            "L*/100 上的有符号差分。正响应表示局部较暗；现有线条融合只读取最细尺度，较粗尺度仍是可观察的候选。",
            "L*/100", dog_limit, True, k)
    stage_info = [
        ("dark_normalized", "暗结构正部／幅值归一", "尚未截断；最细尺度 DoG 正部除以 dark_scale。"),
        ("dark_clipped", "暗结构截断", "超过 1 的值变成平台；平台可能改变后续局部极大值的位置。"),
        ("dark_nms", "暗线方向细化", "沿法向比较 ±1 像素的双线性采样；一侧严格大于。方向不可靠时暂时保留。"),
        ("edge_normalized", "边界幅值归一", "最细尺度结构强度除以 edge_scale。"),
        ("edge_clipped", "边界截断", "当前算法在 NMS 前截断；观察旁边的顺序对照，但不预设其优劣。"),
        ("edge_nms", "边界方向细化", "局部细化只选择峰值，不沿笔画连接缺口。"),
        ("edge_supported", "跨尺度支持", "乘 0.75+0.25×support；只调整已有候选幅值，不生成连接。"),
        ("ridge_near", "暗线邻域最大值", "3×3 形态学膨胀：邻域取最大值；不是空洞卷积。"),
        ("ridge_removed", "暗线附近删除的边界", "强暗线附近边界直接归零。图中标记所有被删除的非零候选，不等于它们本来一定会成为笔画。"),
        ("edge_after_ridge", "抑制重复后的边界", "避免一条暗线两侧都描黑，但当前不检查是否真是同一条线。"),
        ("score_before_gate", "暗线／边界取最大值", "max(dark, 0.65×edge)：没有连续性约束；两类响应位置可能不同。"),
        ("allowed", "允许绘制范围", "去掉最外一圈与删除提示，外轮廓另行绘制；保护提示可以保留边界。"),
        ("score", "范围限制后的分数", "分数属于规则设计的强度，不是网络预测概率。"),
        ("weak", "低阈值候选", "候选构成八连通区域。"),
        ("strong", "高阈值种子", "低阈值连通区域中至少存在一个强种子才能通过一般筛选。"),
        ("retained", "连通筛选后", "滞后阈值保留与强种子相连的弱响应；保护提示有额外规则。"),
        ("thinning_removed", "正常细化移除", "从宽响应带变成骨架去掉的像素；不能据此认定有效结构被误删。"),
        ("skeleton", "骨架", "拓扑细化得到像素图；分叉节点会拆出多条路径。"),
        ("path_removed", "未被任何保留路径覆盖", "骨架中不在 accepted 的像素；与正常细化移除分开统计。"),
        ("accepted", "保留路径的原始位置", "路径拟合前的覆盖；不含从人物遮罩单独提取的外轮廓。"),
    ]
    for name, title, explanation in stage_info:
        value = stages[name]
        add(name, value, title, "lines", explanation, "bool" if value.dtype == np.bool_ else "归一响应", 1)
    for name in ("edge", "dark"):
        add(name + "_nms_before_clip", alternatives[name + "_nms_before_clip"], f"顺序对照：{name} 先细化", "ablation",
            "单变量实验：未截断值做 NMS 后再截断。尚未进入后续筛选与绘制，不能视为新漫画效果。", "归一响应")
        add(name + "_nms_difference", alternatives[name + "_nms_difference"], f"顺序对照：{name} 差值", "ablation",
            "红色为新顺序更强，蓝色为更弱；并非所有新增响应都应保留。", "新−旧", 1, True)
    for key, title, unit, limit, signed in (
        ("chroma_edge", "Lab 色度边缘", "Lab / 像素", max(1, float(bank.chroma_edge.max())), False),
        ("right_affinity", "向右邻接亲和度", "[0,1]", 1, False),
        ("down_affinity", "向下邻接亲和度", "[0,1]", 1, False),
        ("base_lightness", "区域内基础明度", "L*/100", 1, False),
        ("detail_lightness", "明度细节", "L*/100", max(.01, float(np.abs(arrays["detail_lightness"]).max())), True),
        ("shadow_candidate", "阴影候选", "bool", 1, False), ("shadow", "整理后的阴影", "bool", 1, False)):
        value = arrays[key]
        if key == "right_affinity":
            value = np.pad(value, ((0, 0), (0, 1)))
        elif key == "down_affinity":
            value = np.pad(value, ((0, 1), (0, 0)))
        add(key, value, title, "color", "色度图当前仅作观察；区域分支通过邻接色差与结构权重做 WLS。阴影由区域内基础明度筛选，细节层未全部重加回填色。右/下亲和度末列/末行无邻居，以零占位。",
            unit, limit, signed)
    manifest = {"schema_version": 1, "source_run": str(run.resolve()), "source_parameters": record,
                "source_features_sha256": hashlib.sha256((run / "features.npz").read_bytes()).hexdigest(),
                "original_shape": [height, width], "crop_xywh": list(crop), "scales": bank.scales,
                "weights": feature_config.weights, "operator": feature_config.operator,
                "provenance": "saved feature arrays + checked replay; Gaussian decomposition reconstructed from saved 8-bit input",
                "statistics": stats, "maps": catalog,
                "display": "same-family fixed scale; signed blue-negative/red-positive, white background; PNG is not raw data",
                "array_axes": "tensor_terms K,H,W,C,(xx,xy,yy); tensor_raw/smoothed K,H,W,(xx,xy,yy); line stages H,W"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    browser = {**manifest, "data": pixel_data, "mask": _encoded(mask[view]), "input": _encoded(image[view]),
               "padded_guides": padded_guides,
               "kernels": {axis: gradient_kernel(feature_config.operator, axis).ravel().tolist() for axis in ("x", "y")}}
    # 外部静态 JS 在 file:// 下可用，无网络、无本地文件 fetch、无服务依赖。
    (output / "trace-data.js").write_text("window.TRACE=" + json.dumps(browser, ensure_ascii=False, separators=(",", ":")) + ";", encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("feature_lab.html"), output / "index.html")
    make_stage_animation(output, stage_info)
    return output / "index.html"


def make_stage_animation(output: Path, entries: list[tuple[str, str, str]]) -> None:
    selected = {"dark_clipped", "dark_nms", "edge_nms", "edge_after_ridge", "score", "retained", "skeleton", "accepted"}
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default(size=19)
    frames = []
    for key, title, _ in entries:
        if key not in selected:
            continue
        frame = Image.new("RGB", (680, 470), "#f3f3f1")
        draw = ImageDraw.Draw(frame)
        draw.text((18, 12), "同一张自拍 · " + title, font=font, fill="#172229")
        for x, filename in ((16, "input.png"), (352, "maps/" + key + ".png")):
            with Image.open(output / filename) as opened:
                tile = ImageOps.contain(opened.convert("RGB"), (312, 390), Image.Resampling.NEAREST)
            frame.paste(tile, (x + (312 - tile.width) // 2, 60 + (390 - tile.height) // 2))
        frames.append(frame)
    frames[0].save(output / "line-stages.gif", save_all=True, append_images=frames[1:], duration=1250, loop=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="保存逐通道／融合／线条去留的交互观察页与原始数据")
    parser.add_argument("--run", type=Path, required=True, help="已有 comic 结果目录")
    parser.add_argument("--output", type=Path, help="新建观察目录，默认在 run 旁加 -inspection")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="数值窗口，最大 256×256，单位原图像素")
    args = parser.parse_args(argv)
    output = args.output or args.run.with_name(args.run.name + "-inspection")
    try:
        crop = tuple(args.crop) if args.crop else None
        if crop is not None:
            crop = (crop[0], crop[1], crop[2], crop[3])
        result = export_inspection(args.run, output, crop)
        print(f"观察页：{result.resolve()}")
        print(f"逐步动画：{(output / 'line-stages.gif').resolve()}")
        print(f"原始状态与计算说明：{(output / 'trace.npz').resolve()} / manifest.json")
        return 0
    except (ValueError, OSError, KeyError, cv2.error) as error:
        print(f"观察导出未完成：{error}", file=sys.stderr)
        return 2
