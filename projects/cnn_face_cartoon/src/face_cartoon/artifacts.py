"""Three presentation figures and a small, entirely local comparison page."""
import base64
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import Prepared, Result, Structure
from .preprocess import geometry_preview, unpad


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font(size=18):
    for candidate in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def response_rgb(field: np.ndarray) -> np.ndarray:
    # Fixed 0..1 color scale shared by all encoders and samples.
    low, high = np.array([247, 249, 245]), np.array([28, 111, 115])
    return np.rint(low + np.clip(field[..., None], 0, 1) * (high - low)).astype(np.uint8)


def panel(images: list[np.ndarray], labels: list[str], columns: int, title: str,
          path: Path, footer: str = "", cell_width=340, cell_height=384) -> None:
    rows = (len(images) + columns - 1) // columns
    gap, label_height, header, bottom = 16, 36, 72, 48
    width = gap + columns * (cell_width + gap)
    height = header + rows * (cell_height + label_height + gap) + bottom
    canvas = Image.new("RGB", (width, height), "#f5f6f2")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 22), title, font=font(24), fill="#223238")
    for i, (array, label) in enumerate(zip(images, labels)):
        x = gap + (i % columns) * (cell_width + gap)
        y = header + (i // columns) * (cell_height + label_height + gap)
        draw.rectangle((x, y, x + cell_width, y + cell_height), fill="white")
        im = Image.fromarray(array)
        im.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + (cell_width - im.width) // 2, y + (cell_height - im.height) // 2))
        draw.text((x + 4, y + cell_height + 6), label, font=font(16), fill="#223238")
    draw.text((gap, height - 32), footer, font=font(14), fill="#5d6d6f")
    canvas.save(path, dpi=(300, 300))


def export_result(folder: Path, prepared: Prepared, structure: Structure, result: Result, metadata: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    original = unpad(prepared.original_rgb, prepared.transform)
    geometry = unpad(geometry_preview(prepared), prepared.transform)
    support = unpad(response_rgb(structure.support), prepared.transform)
    save_image(folder / "final.png", result.final_rgb)
    for name, image in result.stages.items():
        save_image(folder / f"{name}.png", image)
    save_image(folder / "support.png", support)
    save_image(folder / "edge.png", unpad(response_rgb(prepared.edge), prepared.transform))
    save_image(folder / "score.png", unpad(response_rgb(structure.stroke_score), prepared.transform))
    panel([original, geometry, support, result.stages["lines"], result.stages["color"], result.final_rgb],
          ["01 原图", "02 五官几何辅助", "03 CNN 支持 S · 0→1", "04 筛选与样条笔画", "05 简化颜色层", "06 最终漫画"],
          3, f"{metadata['display_name']} / 六步过程", folder / "process.png",
          "S 为归一化结构响应，非概率；几何辅助预览不直接覆盖到成图。")
    panel([original, result.final_rgb], ["原图", "CNN 引导的漫画"], 2,
          metadata["display_name"], folder / "comparison.png", "颜色取自照片的色板映射，CNN 参与筛选笔画。")
    write_json(folder / "run.json", metadata)


def export_report(output: Path, run: dict) -> None:
    images, labels = [], []
    for sample in run["samples"]:
        image_folder = output / sample["id"]
        images.append(np.asarray(Image.open(image_folder / "input.png")))
        labels.append(sample["label"])
        for key in run["models"]:
            images.append(np.asarray(Image.open(image_folder / key / "final.png")))
            labels.append(run["models"][key]["display_name"])
    panel(images, labels, len(run["models"]) + 1, "同一渲染器，不同 CNN 特征引导", output / "comparison.png",
          "所有模型使用同一画布、配色、密度参数与图像边缘阈值；请同时观察原图和过程面板。", cell_width=240, cell_height=320)
    sample_options = "".join(f'<option value="{html.escape(s["id"])}">{html.escape(s["label"])}</option>' for s in run["samples"])
    model_options = "".join(f'<option value="{key}">{html.escape(value["display_name"])}</option>' for key, value in run["models"].items())
    timing_rows = []
    for sample in run["samples"]:
        for key, metrics in sample["results"].items():
            timing_rows.append(f'<tr><td>{html.escape(sample["label"])}</td><td>{html.escape(run["models"][key]["display_name"])}</td><td>{metrics["extract_seconds"]:.3f} s</td><td>{metrics["render_seconds"]:.3f} s</td><td>{metrics["paths"]}</td></tr>')
    sources = "".join(f'<li>{html.escape(s["label"])}：{html.escape(s["source_note"])}；关键点状态：<code>{html.escape(s["face_status"])}</code></li>' for s in run["samples"])
    models = "".join(f'<li>{html.escape(v["display_name"])} · <a href="https://huggingface.co/{v["hf_repo"]}">{html.escape(v["timm_id"])}</a><br><small>revision {v["revision"]}</small></li>' for v in run["models"].values())
    downloads = {s["id"]: {k: base64.b64encode((output / s["id"] / k / "final.png").read_bytes()).decode("ascii")
                            for k in run["models"]} for s in run["samples"]}
    observations = ('<p class="note">本次观察：四模型整体外观接近，变化主要在眼镜、发丝和细纹的取舍；统一颜色层主导了风格。复杂背景和皱纹仍产生杂线，固定色板使肤色出现较硬的分区；暗光派生图即使温和提亮仍缺少脸部层次，作为局限保留展示。</p>' if {'selfie', 'selfie_dim', 'grace_hopper'} <= {s['id'] for s in run['samples']} else '')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN 人脸漫画 · 作业实践</title><style>
:root{{color-scheme:light;--ink:#223238;--teal:#246f73}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f2;color:var(--ink);font-family:'Microsoft YaHei',sans-serif;line-height:1.8}}main{{max-width:1280px;margin:auto;padding:40px 32px}}header{{padding:30px 0 45px;border-bottom:1px solid #d8dfda}}.eyebrow{{color:var(--teal);font-size:12px;letter-spacing:.12em}}h1{{font-size:46px;line-height:1.3;margin:16px 0}}h2{{font-size:25px;margin:0 0 20px}}p{{max-width:950px}}section{{padding:40px 0;border-bottom:1px solid #d8dfda}}.flow{{display:flex;gap:12px;flex-wrap:wrap}}.flow span{{background:white;border:1px solid #d8dfda;padding:16px 22px;border-radius:8px}}.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}}select,a.button{{padding:10px 16px;border:1px solid #c7d2cc;background:white;border-radius:5px;font:inherit}}a{{color:var(--teal)}}.triplet{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0;text-align:center;min-width:0}}.portrait{{height:420px;width:100%;object-fit:contain;background:white;border:1px solid #e1e7e2}}figcaption,small{{font-size:13px;color:#667572}}.wide{{width:100%;height:auto;display:block}}.frame{{background:white;border-radius:8px;padding:12px;border:1px solid #e1e7e2}}.note{{padding:18px 22px;border-left:3px solid var(--teal);background:#eaf0ec}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d8dfda}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}summary{{cursor:pointer;color:var(--teal)}}footer{{padding:28px 0;font-size:12px;color:#667572}}@media(max-width:700px){{main{{padding:24px 16px}}h1{{font-size:33px}}.portrait{{height:300px}}.pair{{gap:8px}}.flow span{{padding:9px}}}}@media print{{.controls{{display:none}}}}
</style></head><body><main><header><div class="eyebrow">深度学习应用 / CNN PORTRAIT STUDY</div><h1>从照片结构，<br>到漫画色块与笔画。</h1><p>ResNet50、SE-ResNet50、EfficientNetV2-S、MobileNetV3-Large 共用一套渲染器。这里比较预训练 CNN 对边缘取舍的影响；不训练、微调或直接解码生成漫画。</p><div class="flow"><span>01 预处理</span><span>02 CNN 多尺度结构</span><span>03 色块与样条笔画</span><span>04 抗锯齿合成</span></div></header>
<section><h2>01 / 四模型对照</h2><p>每行固定同一输入和渲染参数。更换编码器改变结构支持 S，颜色层保持一致；相近结果也如实展示。</p><div class="frame"><img class="wide" src="comparison.png" alt="原图与四种 CNN 结果对照"></div></section>
<section><h2>02 / 查看单张结果与过程</h2><div class="controls"><label>样例 <select id="sample">{sample_options}</select></label><label>模型 <select id="model">{model_options}</select></label><a class="button" id="save" download="final.png">保存漫画 PNG</a></div><div class="pair"><figure><img class="portrait" id="original" alt="原始照片"><figcaption>原图 · 等比例显示</figcaption></figure><figure><img class="portrait" id="final" alt="漫画结果"><figcaption>统一色板与渲染器 · CNN 引导笔画</figcaption></figure></div><p id="sample-note" class="note"></p><img class="wide" id="process" alt="六格过程面板"><details><summary>查看 S、E 与 Q，以及关闭 CNN 门控的对照</summary><div class="triplet"><figure><img class="portrait" id="support" alt="CNN 结构支持 S"><figcaption>归一化 CNN 响应 S · 0→1</figcaption></figure><figure><img class="portrait" id="edge" alt="图像边缘 E"><figcaption>单尺度 Sobel 边缘 E</figcaption></figure><figure><img class="portrait" id="score" alt="门控候选 Q"><figcaption>Q = E × (0.25 + 0.75 S)</figcaption></figure></div><img class="wide" id="baseline" alt="CNN 门控开关对照"><p>下方对照固定使用本次首个编码器。关闭门控仅令 Q=E，保留颜色层、几何阈值和其余渲染规则。它用于确认 CNN 实际影响笔画，不是漫画质量评测。</p></details></section>
<section><h2>03 / 主要方法</h2><p><b>预处理：</b>EXIF 方向修正，等比缩放与复制边缘填充到 384×384；轻度双边滤波，明显欠曝时最多 1.35 倍温和提亮。模型输入使用各自权重的均值和标准差，保留完整画面。</p><p><b>结构：</b>依据实际 feature_info 选 stride 4、8、16。每个位置的通道向量做 L2 归一化，计算相邻向量差；在有效区域按 q10～q95 归一化，上采样后以 0.5／0.3／0.2 加权为 S。填充不参与统计，近常量响应置零。S 是响应分数，不是边界概率。</p><p><b>笔画：</b>以图像 Sobel 提供精确位置，CNN 门控筛线。沿梯度法向做非极大值抑制与滞后连接，MediaPipe 只用于眼嘴低阈值和脸颊抑制。骨架按端点、交叉点和闭环追踪；参数 B 样条有限平滑，保护端点和急转处，采样偏移上限为 {run['config']['style']['max_spline_deviation_px']} 像素，必要时退回折线。</p><p><b>颜色与后处理：</b>OpenCV 保边平滑后在 Lab 空间映射到固定 12 色暖色板，用局部多数压制孤立色点。深棕色双档线宽在 3 倍画布绘制，再缩回获得抗锯齿。首版不添加额外阴影，不自动去背景，不绘制人脸网格。</p><p class="note">使用的 CNN 是分类预训练骨干，缺少将特征转换为漫画的已训练解码器；因此采用程序化颜色与线条渲染。五官可辨识度、杂线和配色仍取决于照片与启发式规则，不能由 ImageNet 分类成绩推断。</p></section>
<section><h2>04 / 本次运行与观察边界</h2>{observations}<ul>{sources}</ul><p>设备：{run['device']}；线程数：{run['threads']}。下表提取时间来自一次预热后的实际前向与结构计算，渲染时间包含共享颜色层耗时；不包含权重加载、关键点检测和图表导出。复用缓存时，提取栏沿用缓存首次生成的实测时间。完整阶段耗时见 <a href="run.json">run.json</a>，不能据此承诺实时性能。</p><div class="scroll"><table><tr><th>样例</th><th>模型</th><th>CNN＋结构</th><th>颜色＋笔画</th><th>路径数</th></tr>{''.join(timing_rows)}</table></div><p>只有少量演示照片；暗光派生图与原图是同一个人，不计作独立人物样本。关键点位置、眼镜和皱纹细线、固定色板带来的肤色偏移均应结合原图查看。没有人工质量评分，不报告模型优劣排名。</p><details><summary>固定权重与技术资料</summary><ul>{models}</ul><p><a href="https://huggingface.co/docs/timm/feature_extraction">timm 特征接口</a> · <a href="https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python">MediaPipe</a> · <a href="https://docs.opencv.org/4.x/df/dac/group__photo__render.html">OpenCV 保边滤波</a> · <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html">SciPy 样条</a></p></details></section><footer>所有图像在本地生成。目录内 index.html 与相对路径图片一起保留，可离线浏览。</footer></main>
<script id="run-data" type="application/json">{json.dumps(run['samples'], ensure_ascii=False).replace('</', '<\\/')}</script><script>
const samples=JSON.parse(document.getElementById('run-data').textContent);
const downloads={json.dumps(downloads)};
const sample=document.getElementById('sample'), model=document.getElementById('model');
function update(){{const s=samples.find(x=>x.id===sample.value),base=encodeURIComponent(sample.value)+'/'+model.value+'/';for(const k of ['final','process','edge','score','support'])document.getElementById(k).src=base+k+'.png';document.getElementById('original').src=encodeURIComponent(sample.value)+'/input.png';document.getElementById('baseline').src=encodeURIComponent(sample.value)+'/baseline_comparison.png';document.getElementById('save').href='data:image/png;base64,'+downloads[sample.value][model.value];document.getElementById('save').download=sample.value+'_'+model.value+'.png';document.getElementById('sample-note').textContent=s.source_note+' | 关键点：'+s.face_status;}}sample.addEventListener('change',update);model.addEventListener('change',update);update();
</script></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
