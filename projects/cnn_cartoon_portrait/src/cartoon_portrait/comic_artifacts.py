"""漫画管道的诊断文件：原始特征、筛选过程、配色、路径与阶段对照。"""

import html
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .artifacts import _save_rgb, preview
from .comic import ComicConfig, ComicResult
from .pipeline import FloatImage, Labels, Mask, save_png
from .regions import IntMap


def label_preview(labels: IntMap) -> FloatImage:
    """稳定的离散标签色，不表示最终配色。"""
    index = np.arange(int(labels.max()) + 1, dtype=np.float32)
    hsv = np.stack((np.mod(index * 137.508, 360), np.full_like(index, .5), np.full_like(index, .9)), axis=1)
    rgb = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]
    rgb[0] = 1
    return rgb[labels].astype(np.float32)


def contact_sheet(directory: Path, entries: list[tuple[str, str]], name: str, columns: int = 3) -> None:
    rows = (len(entries) + columns - 1) // columns
    board = Image.new("RGB", (columns * 310, rows * 360), "#eeeeee")
    draw = ImageDraw.Draw(board)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default(size=18)
    for index, (filename, title) in enumerate(entries):
        x, y = (index % columns) * 310, (index // columns) * 360
        draw.text((x + 12, y + 8), title, font=font, fill="#222222")
        with Image.open(directory / filename) as opened:
            image = opened.convert("RGB")
            tile = ImageOps.contain(image, (288, 312), Image.Resampling.LANCZOS)
        board.paste(tile, (x + (310 - tile.width) // 2, y + 40 + (312 - tile.height) // 2))
    board.save(directory / name)


def export_comic(directory: Path, image: FloatImage, mask: Mask, result: ComicResult,
                  config: ComicConfig, metadata: dict[str, object], labels: Labels | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    bank, line, scene, fills, drawing = result.features, result.line_features, result.scene, result.fills, result.rendered
    _save_rgb(image, directory / "input.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / "mask.png")
    if labels is not None:
        Image.fromarray(labels).save(directory / "selection_labels.png")
    save_png(drawing.rgba, directory / "cartoon.png")
    _save_rgb(preview(drawing.rgba, 1), directory / "preview_light.png")
    _save_rgb(preview(drawing.rgba, .12), directory / "preview_dark.png")
    _save_rgb(bank.guides[0], directory / "base.png")
    _save_rgb(bank.guides[0], directory / "guide.png")
    def on_white(values: FloatImage) -> FloatImage:
        return np.where(mask[..., None], values, np.float32(1)).astype(np.float32)
    _save_rgb(on_white(fills.flat), directory / "flat_colors.png")
    _save_rgb(on_white(fills.colors), directory / "colors.png")
    _save_rgb(drawing.colors * drawing.alpha[..., None] + 1 - drawing.alpha[..., None], directory / "rendered_colors.png")
    _save_rgb(label_preview(scene.initial_labels), directory / "regions_initial.png")
    _save_rgb(label_preview(scene.labels), directory / "regions.png")
    scales: dict[str, object] = {}
    arrays = {
        "mask": mask, "scales": np.array(bank.scales), "guides": bank.guides,
        "gx": bank.gx, "gy": bank.gy, "edges": bank.edges, "dog": bank.dog,
        "theta": bank.theta, "coherence": bank.coherence, "direction_valid": bank.direction_valid,
        "lab": bank.lab, "chroma_edge": bank.chroma_edge, "line_edge": line.edge,
        "line_dark": line.dark, "line_score": line.score, "line_source": line.source,
        "line_scale_support": line.scale_support, "retained": line.retained, "skeleton": line.skeleton,
        "accepted": result.strokes.accepted, "rejected": result.strokes.rejected,
        "protected": line.protected, "suppressed": line.suppressed,
        "region_descriptor": scene.features.descriptor, "right_affinity": scene.features.right,
        "down_affinity": scene.features.down, "initial_regions": scene.initial_labels,
        "regions": scene.labels, "clusters": scene.clusters,
        "base_lightness": fills.base, "detail_lightness": fills.detail,
        "shadow_candidate": fills.shadow_candidate, "shadow": fills.shadow,
        "shadow_thresholds": fills.thresholds, "palette": fills.palette,
        "shadow_palette": fills.shadow_palette,
    }
    np.savez_compressed(directory / "features.npz", allow_pickle=False, **arrays)
    np.savez_compressed(directory / "render.npz", alpha=drawing.alpha, ink=drawing.ink)

    def plane(name: str, value: FloatImage) -> None:
        valid = mask if value.shape == mask.shape else np.ones(value.shape, dtype=bool)
        display = np.where(valid, np.clip(value, 0, 1), 1)
        Image.fromarray(np.rint(display * 255).astype(np.uint8)).save(directory / (name + ".png"))

    edge_scale = max(.01, float(np.max(bank.edges)))
    dog_scale = max(.01, float(np.max(np.abs(bank.dog))))
    chroma_scale = max(1.0, float(np.max(bank.chroma_edge)))
    # RGB 通道在同一尺度、不同尺度之间都共用一组显示范围。
    channel_edges = np.hypot(bank.gx, bank.gy) * np.array(bank.scales, dtype=np.float32)[:, None, None, None]
    channel_scale = max(.01, float(np.max(channel_edges)))
    features_entries = []
    for index, sigma in enumerate(bank.scales):
        plane(f"edge_scale_{index}", 1 - bank.edges[index] / edge_scale)
        plane(f"dog_scale_{index}", .5 + .5 * bank.dog[index] / dog_scale)
        features_entries.append((f"edge_scale_{index}.png", f"方向融合 σ={sigma:g}"))
        for channel, name in enumerate(("red", "green", "blue")):
            plane(f"response_{name}_{index}", 1 - channel_edges[index, ..., channel] / channel_scale)
    for name, values in {
        "chroma_edge": 1 - bank.chroma_edge / chroma_scale,
        "dark_candidate": 1 - np.clip(np.maximum(bank.dog[0], 0) / config.strokes.dark_scale, 0, 1),
        "coherence": bank.coherence[0], "boundary_weights": scene.features.affinity,
        "line_edge": 1 - line.edge, "line_dark": 1 - line.dark, "line_score": 1 - line.score,
        "lines_retained": 1 - line.retained.astype(np.float32),
        "skeleton": 1 - line.skeleton.astype(np.float32),
        "lines_accepted": 1 - result.strokes.accepted.astype(np.float32),
        "lines_rejected": 1 - result.strokes.rejected.astype(np.float32),
        "shadow_candidate": 1 - fills.shadow_candidate.astype(np.float32),
        "shadow": 1 - fills.shadow.astype(np.float32), "base_lightness": fills.base,
        "lines": 1 - drawing.ink * drawing.alpha,
    }.items():
        plane(name, values)
    detail_scale = max(.01, float(np.max(np.abs(fills.detail))))
    plane("detail_lightness", .5 + .5 * fills.detail / detail_scale)
    hsv = np.stack((np.mod(bank.theta[0] + np.pi / 2, np.pi) * (360 / np.pi),
                    bank.coherence[0], np.ones(mask.shape, np.float32)), axis=-1).astype(np.float32)
    direction_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    direction_rgb[~bank.direction_valid[0]] = 1
    _save_rgb(direction_rgb, directory / "directions.png")
    # 曲线叠回原图，坐标对照不使用颜色响应图代替。
    overlay = np.rint(image * 255).astype(np.uint8).copy()
    for stroke in result.strokes.strokes:
        color = (20, 135, 195) if stroke.source == "outline" else (200, 35, 70)
        cv2.polylines(overlay, [np.rint(stroke.points * 256).astype(np.int32)], stroke.closed, color,
                      1, cv2.LINE_AA, shift=8)
    Image.fromarray(overlay).save(directory / "stroke_overlay.png")
    stroke_json = [{**asdict(stroke), "points": stroke.points.tolist()} for stroke in result.strokes.strokes]
    (directory / "strokes.json").write_text(json.dumps(stroke_json, ensure_ascii=False, indent=2), encoding="utf-8")
    palette_json = [{"region": i, "area": int(scene.area[i]), "base_rgb": fills.palette[i].tolist(),
                     "shadow_rgb": fills.shadow_palette[i].tolist(), "shadow_threshold": float(fills.thresholds[i])}
                    for i in range(1, len(fills.palette))]
    (directory / "palette.json").write_text(json.dumps(palette_json, ensure_ascii=False, indent=2), encoding="utf-8")
    scales.update(edge_display_scale=edge_scale, channel_display_scale=channel_scale,
                  dog_signed_display_scale=dog_scale, chroma_display_scale=chroma_scale,
                  detail_signed_display_scale=detail_scale)
    details = {**metadata, "schema_version": 3, "config": asdict(config), "shape": list(image.shape),
               "output_shape": list(drawing.rgba.shape), "alpha": "coverage, straight; supersampling in premultiplied linear RGB",
               "feature_axes": "gx/gy/guides: K,H,W,C; edges/dog/theta/coherence: K,H,W",
               "direction": "theta is normal modulo pi; directions.png shows tangent; use direction_valid",
               "response_display": "1-response/scale; signed maps .5+.5*value/scale; invalid background white",
               "display_scales": scales, "regions": len(fills.palette) - 1, "strokes": len(stroke_json),
               "solver_max_relative_residual": fills.solver_residual,
               "algorithm": "fixed convolutions + current-image clustering + graph paths; no trained semantic model",
               "dependencies": {name: version(name) for name in ("numpy", "opencv-python", "Pillow")}}
    (directory / "parameters.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    overview = [("input.png", "输入照片"), ("regions.png", "连通区域（标签色）"),
                ("flat_colors.png", "纯底色"), ("shadow.png", "阴影形状"),
                ("lines.png", "重绘笔画"), ("preview_light.png", "最终漫画")]
    features_entries += [("chroma_edge.png", "Lab 色度变化"), ("dark_candidate.png", "暗结构候选"),
                         ("directions.png", "切向／一致性（色相模 π）")]
    contact_sheet(directory, overview, "overview.png")
    contact_sheet(directory, features_entries, "features_overview.png")
    line_entries = [("line_edge.png", "边界候选"), ("line_dark.png", "暗线候选"),
                    ("lines_retained.png", "连通筛选"), ("lines_rejected.png", "未用于路径的响应"),
                    ("stroke_overlay.png", "路径叠回原图"), ("lines.png", "独立墨色与线宽")]
    contact_sheet(directory, line_entries, "lines_overview.png")
    sections = [("阶段", overview), ("特征", features_entries), ("线条", line_entries)]
    blocks = []
    for title, entries in sections:
        figures = "".join(f'<figure><a href="{html.escape(file)}"><img src="{html.escape(file)}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>' for file, caption in entries)
        blocks.append(f"<h2>{title}</h2><section>{figures}</section>")
    report = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
              '<title>漫画处理过程</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:20px;background:#fafafa;color:#222}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}figure{margin:0}img{width:100%;height:300px;object-fit:contain}h2{font-size:20px}figcaption{text-align:center}</style>'
              '<h1>漫画处理过程</h1><p>照片与特征在原图尺寸分析，笔画与填色在目标画布绘制。不同类别响应单位不同；标签色不是漫画配色。点击图片查看原尺寸。</p>'
              + "".join(blocks) + '</html>')
    (directory / "report.html").write_text(report, encoding="utf-8")
    return directory / "cartoon.png"
