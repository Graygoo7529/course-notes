"""PNG 与可离线打开的报告，不依赖 CDN、外部脚本或原图绝对路径。"""

from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .labels import FaceLabel
from .regions import PALETTE


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def comparison(images: list[np.ndarray], target: Path) -> None:
    canvas = Image.new("RGB", (2048, 558), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    captions = ["Input photo", "UNet segmentation", "Controls OFF", "Controls ON"]
    for i, (image, caption) in enumerate(zip(images, captions, strict=True)):
        view = ImageOps.contain(Image.fromarray(image), (512, 512), Image.Resampling.BICUBIC)
        canvas.paste(view, (512 * i + (512 - view.width) // 2, (512 - view.height) // 2))
        draw.text((512 * i + 12, 525), caption, fill="black", font=font)
    canvas.save(target)


def write_report(directory: Path, records: list[dict], metadata: dict) -> None:
    report = {"metadata": metadata, "results": records}
    (directory / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    sections = []
    for record in records:
        name = escape(record["id"])
        if record["status"] != "ok":
            sections.append(f"<section><h2>{name}</h2><p>未生成：{escape(record['error'])}</p></section>")
            continue
        figures = []
        for filename, title in [
            ("input.png", "原图"), ("segmentation.png", "UNet 语义分割"),
            ("baseline.png", "关闭区域控制"), ("cartoon.png", "开启区域控制"),
        ]:
            src = f"{name}/{filename}"
            figures.append(f'<figure><a href="{src}"><img src="{src}" alt="{title}"></a>'
                           f'<figcaption>{title} · <a href="{src}" download>保存</a></figcaption></figure>')
        details = []
        for filename, title in [
            ("portrait.png", "统一的 512 头像"), ("overlay.png", "标签叠加"),
            ("skin_weight.png", "皮肤处理权重"), ("hair_weight.png", "头发处理权重"),
            ("foreground_raw.png", "原始前景"), ("foreground.png", "清理孤立块后的前景"),
            ("baseline_input.png", "生成输入：关闭"),
            ("edited_input.png", "生成输入：开启"), ("output_difference.png", "输出差异 ×4"),
        ]:
            details.append(f'<figure><img src="{name}/{filename}" alt="{title}">'
                           f"<figcaption>{title}</figcaption></figure>")
        source = record.get("source")
        credit = "用户提供的本地图片；分享前请自行确认授权。"
        if source:
            credit = (f"图片：{escape(source['author'])}；{escape(source['license'])}；"
                      f'<a href="{escape(source["source_page"], quote=True)}">原始出处</a>；'
                      f'<a href="{escape(source["license_url"], quote=True)}">许可条款</a>。'
                      "本报告中的对齐、裁剪和漫画均为算法派生结果，不代表人物背书。"
                      "分享 CC BY-SA 图片的派生作品时需保留署名并遵守相同方式共享条款。")
        info = escape(json.dumps(record, ensure_ascii=False, indent=2))
        sections.append(
            f"<section><h2>{name}</h2><div class='grid'>{''.join(figures)}</div>"
            f"<p>{credit}</p><p>{escape('；'.join(record['warnings']))}</p>"
            f'<a href="{name}/comparison.png" download>保存四图对照</a>'
            f"<details><summary>中间过程与参数</summary><div class='grid'>{''.join(details)}</div>"
            f"<pre>{info}</pre></details></section>"
        )
    legend = "".join(
        f'<span><i style="background:rgb({int(c[0])},{int(c[1])},{int(c[2])})"></i>'
        f"{label.value}: {label.name}</span>" for label, c in zip(FaceLabel, PALETTE.tolist(), strict=True)
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>UNet 人脸漫画实验</title><style>
body{{max-width:1400px;margin:auto;padding:24px;font:16px/1.6 system-ui;color:#202830;background:#f3f5f7}}
h1,h2{{line-height:1.3}}section{{background:white;padding:20px;margin:24px 0;border-radius:12px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}figure{{margin:0}}
img{{width:100%;height:310px;object-fit:contain;background:white}}figcaption{{text-align:center}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;font-size:12px}}i{{display:inline-block;width:12px;height:12px;margin-right:4px}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}img{{height:240px}}}}
@media(max-width:450px){{.grid{{grid-template-columns:1fr}}}}a{{color:#145da0}}summary{{cursor:pointer;margin-top:16px}}
</style><h1>UNet 语义分割引导的人脸漫画</h1>
<p>同一裁剪、同一前景、同一生成器，仅切换区域处理。生成器真实输出为 256×256；
放大显示不代表恢复高清细节。输入局部编辑也可能影响输出的其他区域。</p>
<p>通过命令行的 skin-strength、hair-color、hair-strength 调整并重新运行；本页面只查看已生成结果。</p>
<div class="legend">{legend}</div>{''.join(sections)}
<footer>课程研究演示；没有训练或微调。差异数值不是质量评分。模型许可与图片许可分别适用。</footer></html>"""
    (directory / "index.html").write_text(html, encoding="utf-8")
