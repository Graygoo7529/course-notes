"""Write an image-first run directory and the compact observer manifest."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .comic import CartoonResult, V2Config
from .pipeline import ensure_dir, robust01, save_gray, save_rgb


def _write(path: Path, value: np.ndarray, mask: np.ndarray | None = None) -> str:
    if value.ndim == 3 and value.shape[-1] == 3:
        image = value.copy()
        if mask is not None and mask.shape == value.shape[:2]:
            image[~mask] = 1
        save_rgb(path, image)
    elif value.dtype == bool:
        display = (value.astype(np.float32) > 0).astype(np.uint8) * 255
        if mask is not None and mask.shape == value.shape:
            display[~mask] = 255
        cv2.imwrite(str(path), display)
    elif np.issubdtype(value.dtype, np.integer):
        # Labels are categorical: a grayscale normalization would make label 0
        # indistinguishable from the outside mask, so use a stable false colour map.
        labels = value.astype(np.int32)
        out = np.full((*labels.shape, 3), 255, np.uint8)
        palette = np.array([[232, 92, 80], [84, 156, 230], [104, 190, 120],
                            [224, 179, 74], [166, 118, 206], [92, 188, 188]], np.uint8)
        for i in range(len(palette)):
            out[labels == i] = palette[i]
        cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    elif "dog_signed" in path.stem:
        # Signed DoG: mid-gray is zero, bright/dark show opposite responses.
        values = value.astype(np.float32)
        valid = mask if mask is not None else np.ones(values.shape, bool)
        sample = np.abs(values[valid]) if np.any(valid) else np.array([1], np.float32)
        scale = max(.01, float(np.quantile(sample, .98)))
        display = .5 + .5 * np.clip(values / scale, -1, 1)
        display[~valid] = 1
        cv2.imwrite(str(path), np.rint(display * 255).astype(np.uint8))
    elif "theta" in path.stem:
        # Orientation is periodic modulo pi; hue is more readable than a linear gray map.
        valid = mask if mask is not None else np.ones(value.shape, bool)
        hue = np.mod(value.astype(np.float32), np.pi) / np.pi * 179
        hsv = np.zeros((*value.shape, 3), np.uint8)
        hsv[..., 0] = np.rint(hue).astype(np.uint8)
        hsv[..., 1] = np.where(valid, 210, 0).astype(np.uint8)
        hsv[..., 2] = np.where(valid, 220, 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    else:
        display = robust01(value, mask)
        valid = mask if mask is not None and mask.shape == value.shape else np.ones(value.shape, bool)
        # Match v1's response display: stronger response is darker and the
        # invalid background is white. Signed maps and directions are handled
        # by their branches above.
        if any(token in path.stem for token in ("edge", "response", "score", "dark_candidate", "nms")):
            display = 1 - display
        display[~valid] = 1
        cv2.imwrite(str(path), np.rint(display * 255).astype(np.uint8))
    return path.as_posix()


def save_run(result: CartoonResult, output: str | Path, config: V2Config, input_path: str, mask_path: str) -> Path:
    root = ensure_dir(output)
    image_dir = ensure_dir(root / "images")
    images: dict[str, dict[str, str]] = {}
    shapes: dict[str, list[int]] = {}
    for stage, mapping in result.stages.items():
        images[stage] = {}
        for name, value in mapping.items():
            path = image_dir / f"{stage}__{name}.png"
            rel = _write(path, np.asarray(value), result.features.mask)
            images[stage][name] = str(Path(rel).relative_to(root).as_posix())
            shapes[f"{stage}.{name}"] = list(np.asarray(value).shape)
    save_rgb(root / "input.png", result.features.image)
    cv2.imwrite(str(root / "mask.png"), result.features.mask.astype(np.uint8) * 255)
    save_rgb(root / "cartoon.png", result.rendered.rgb)
    np.savez_compressed(
        root / "features.npz", scales=np.asarray(result.features.scales),
        guides=result.features.guides, gx=result.features.gx, gy=result.features.gy,
        edges=result.features.edges, dog=result.features.dog, theta=result.features.thetas,
        coherence=result.features.coherences, direction_valid=result.features.direction_valid,
        edge=result.features.edge, theta_fused=result.features.theta,
        coherence_fused=result.features.coherence, dark=result.features.dark,
        chroma_edge=result.features.chroma_edge, lightness=result.features.lightness)
    params = asdict(config)
    (root / "parameters.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    scale_names = [f"{s:g}" for s in result.features.scales]
    feature_scale_guides = [f"guide_sigma_{s}" for s in scale_names]
    feature_scale_edges = [f"edge_sigma_{s}" for s in scale_names]
    feature_scale_dogs = [f"dog_signed_sigma_{s}" for s in scale_names]
    feature_scale_thetas = [f"theta_sigma_{s}" for s in scale_names]
    feature_scale_coherences = [f"coherence_sigma_{s}" for s in scale_names]
    nodes = [
        {"id": "features", "title": "特征提取", "purpose": "以多个尺度提取能直接解释轮廓、方向、暗部和色差的特征。",
         "inputs": [{"stage": "features", "name": "input"}],
         "outputs": [{"stage": "features", "name": n} for n in ["edge_fused", "theta_fused", "coherence_fused", "dark_candidate", "lightness", "chroma_edge", "color_barrier"]],
         "ops": [
             {"title": "多尺度空间可分离高斯", "description": "横向和纵向一维高斯卷积分别作用于 RGB，抑制噪声但不混合通道。", "outputs": feature_scale_guides},
             {"title": "逐通道方向卷积", "description": "每个尺度分别计算 R、G、B 的 gx、gy 和梯度幅值，观察哪个通道提供结构。", "outputs": ["gx_R_fine", "gx_G_fine", "gx_B_fine", "response_R_fine", "response_G_fine", "response_B_fine"]},
             {"title": "结构张量与空间平滑", "description": "逐通道二次项加权融合，再平滑 Jxx、Jxy、Jyy，得到稳定方向和一致性。", "outputs": feature_scale_edges + feature_scale_thetas + feature_scale_coherences},
             {"title": "尺度归一化与跨尺度选择", "description": "用 sigma·sqrt(lambda_max) 比较尺度，并在每个像素选择结构最强尺度。", "outputs": ["edge_fused", "theta_fused", "coherence_fused"]},
             {"title": "有符号 DoG", "description": "宽尺度亮度减窄尺度亮度；灰色为零，保留变暗与变亮的方向。", "outputs": feature_scale_dogs + ["dark_candidate"]},
             {"title": "Lab 色差特征", "description": "从 Lab 的 L、a、b 计算亮度、色度梯度和邻域颜色屏障。", "outputs": ["lightness", "chroma_edge", "color_barrier"]},
         ]},
        {"id": "lines", "title": "线条生成", "purpose": "沿方向响应细化线条，并以滞后阈值保留连续笔画。",
         "inputs": [{"stage": "features", "name": n} for n in ["edge_fused", "theta_fused", "coherence_fused", "dark_candidate"]],
         "outputs": [{"stage": "lines", "name": n} for n in ["score", "retained", "skeleton", "bridged"]],
         "ops": [
             {"title": "方向非极大值抑制", "description": "沿特征提取产生的 theta 法线比较两侧响应，保留局部峰值。", "outputs": ["nms_edge", "nms_dark"]},
             {"title": "结构与暗部软融合", "description": "融合几何边缘和 DoG 暗部，避免单独依赖某一种响应。", "outputs": ["score"]},
             {"title": "滞后阈值连通", "description": "强线作为种子吸收相邻弱线，减少断裂并避免全局闭运算粘连。", "outputs": ["retained"]},
             {"title": "骨架化与方向一致短桥", "description": "把线条区域细化为中心线，只连接短距离且切线一致的端点。", "outputs": ["skeleton", "bridged"]},
         ]},
        {"id": "regions", "title": "区域生成", "purpose": "依据亮度、色度边缘和邻域屏障形成连续的大色块。",
         "inputs": [{"stage": "features", "name": n} for n in ["lightness", "chroma_edge", "color_barrier"]],
         "outputs": [{"stage": "regions", "name": n} for n in ["initial_labels", "labels", "flat"]],
         "ops": [
             {"title": "Lab 原型量化", "description": "在 Lab 空间把像素分配到少量代表色，形成初始区域。", "outputs": ["initial_labels"]},
             {"title": "颜色屏障约束与碎片合并", "description": "优先向颜色距离小且屏障弱的相邻区域合并小碎片。", "outputs": ["labels"]},
             {"title": "区域代表色", "description": "用每个最终区域的中位 Lab 颜色生成稳定的平涂底色。", "outputs": ["flat"]},
         ]},
        {"id": "tone", "title": "明暗生成", "purpose": "把区域底色和局部阴影分开，控制明暗细节。",
         "inputs": [{"stage": "features", "name": "lightness"}, {"stage": "regions", "name": "labels"}, {"stage": "regions", "name": "flat"}],
         "outputs": [{"stage": "tone", "name": n} for n in ["base", "detail", "shadow_candidate", "shadow", "colors"]],
         "ops": [
             {"title": "区域基础明度", "description": "从特征提取的 Lab L* 与区域标签求中位亮度，形成平稳底层。", "outputs": ["base"]},
             {"title": "明度细节残差", "description": "计算 lightness−base，单独观察哪些局部明暗被底色抹去。", "outputs": ["detail"]},
             {"title": "区域分位数阴影", "description": "只在有足够明度范围的区域选择低分位像素，并去除孤立点。", "outputs": ["shadow_candidate", "shadow"]},
             {"title": "只修改 L* 的上色", "description": "保持区域色相和饱和度，只用阴影降低 Lab L*。", "outputs": ["colors"]},
         ]},
        {"id": "compose", "title": "图片合成", "purpose": "放大画布、抗锯齿绘制笔画并合成 RGBA 图像。",
         "inputs": [{"stage": "regions", "name": "flat"}, {"stage": "tone", "name": "colors"}, {"stage": "lines", "name": "bridged"}],
         "outputs": [{"stage": "compose", "name": n} for n in ["regions", "strokes", "cartoon"]],
         "ops": [
             {"title": "色块与线条放大绘制", "description": "色块用最近邻保持边界，线条在放大画布上以抗锯齿方式重绘。", "outputs": ["regions", "strokes"]},
             {"title": "图层合成", "description": "将明暗颜色和墨线按覆盖率合成为最终漫画。", "outputs": ["cartoon"]},
         ]},
    ]
    image_info = _image_info(result)
    manifest = {"version": "v2", "input": input_path, "mask": mask_path, "images": images,
                "image_info": image_info,
                "shapes": shapes, "architecture": {"nodes": nodes,
                "edges": [["features", "lines"], ["features", "regions"], ["regions", "tone"], ["lines", "compose"], ["tone", "compose"]]}}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_observer(root, manifest)
    return root


def _image_info(result: CartoonResult) -> dict[str, dict[str, dict[str, str]]]:
    """Short Chinese explanations shown beside every observable feature map."""
    info: dict[str, dict[str, dict[str, str]]] = {
        "features": {
            "input": {"title": "输入照片", "meaning": "原始 RGB 图像。", "purpose": "所有卷积和颜色分析的输入。"},
            "guide_fine": {"title": "细尺度引导图", "meaning": "轻度空间平滑后的照片。", "purpose": "抑制像素噪声，同时保留镜框、眼睑等细节。"},
            "gx_R_fine": {"title": "R 通道水平梯度", "meaning": "红色通道沿 x 方向的 Sobel 响应。", "purpose": "观察红色变化对结构边缘的贡献。"},
            "gx_G_fine": {"title": "G 通道水平梯度", "meaning": "绿色通道沿 x 方向的 Sobel 响应。", "purpose": "观察绿色变化对结构边缘的贡献。"},
            "gx_B_fine": {"title": "B 通道水平梯度", "meaning": "蓝色通道沿 x 方向的 Sobel 响应。", "purpose": "观察蓝色变化对结构边缘的贡献。"},
            "response_R_fine": {"title": "R 通道梯度幅值", "meaning": "R 通道 sqrt(gx²+gy²)，黑色表示响应强。", "purpose": "与 v1 的逐通道幅值图对照。"},
            "response_G_fine": {"title": "G 通道梯度幅值", "meaning": "G 通道 sqrt(gx²+gy²)，黑色表示响应强。", "purpose": "观察绿色结构响应。"},
            "response_B_fine": {"title": "B 通道梯度幅值", "meaning": "B 通道 sqrt(gx²+gy²)，黑色表示响应强。", "purpose": "观察蓝色结构响应。"},
            "edge_fused": {"title": "融合结构强度", "meaning": "各尺度结构张量最大特征值的尺度归一化响应。", "purpose": "提供轮廓和线条候选。"},
            "theta_fused": {"title": "融合法线方向", "meaning": "最强尺度的梯度法线方向，颜色按角度编码。", "purpose": "告诉非极大值抑制沿哪条方向比较邻域。"},
            "coherence_fused": {"title": "方向一致性", "meaning": "结构张量两个特征值的相对差异。", "purpose": "过滤方向不稳定的平坦或噪声区域。"},
            "dark_candidate": {"title": "暗部候选", "meaning": "多尺度 DoG 的正响应。", "purpose": "突出眼睑、嘴缝等适合转成墨线的局部暗部。"},
            "lightness": {"title": "Lab 亮度 L*", "meaning": "引导图的 L 通道归一化结果。", "purpose": "明确提供给区域和明暗分支的亮度来源。"},
            "chroma_edge": {"title": "色度边缘", "meaning": "Lab 的 a、b 通道梯度融合。", "purpose": "发现颜色改变但亮度变化不明显的区域边界。"},
            "color_barrier": {"title": "颜色屏障", "meaning": "相邻像素的 Lab 差异。", "purpose": "阻止区域合并跨越明显色差。"},
        },
        "lines": {
            "edge_fused": {"title": "线条结构输入", "meaning": "进入线条分支的融合结构强度。", "purpose": "提供几何轮廓。"},
            "dark_candidate": {"title": "线条暗部输入", "meaning": "进入线条分支的暗部响应。", "purpose": "补足嘴缝、眼睑等暗线。"},
            "nms_edge": {"title": "方向细化结构", "meaning": "沿 theta 法线只保留局部峰值。", "purpose": "让线条变细并减少双边响应。"},
            "nms_dark": {"title": "方向细化暗线", "meaning": "对暗部候选进行同样的方向筛选。", "purpose": "保留局部暗结构中心。"},
            "score": {"title": "线条综合分数", "meaning": "结构线与暗线的软融合。", "purpose": "为滞后阈值提供统一评分。"},
            "retained": {"title": "滞后阈值结果", "meaning": "强响应连接相邻弱响应后的线条区域。", "purpose": "保持连续笔画。"},
            "skeleton": {"title": "骨架线", "meaning": "线条区域的细化结果。", "purpose": "为路径重绘提供中心线。"},
            "bridged": {"title": "断线修整", "meaning": "只连接短距离且切线方向一致的端点。", "purpose": "修复眼镜框、嘴唇等小断裂而不粘连无关结构。"},
        },
        "regions": {
            "lab_lightness": {"title": "Lab 亮度", "meaning": "引导图的 L 通道。", "purpose": "参与色块原型和明暗分析。"},
            "chroma_edge": {"title": "色度边缘", "meaning": "a、b 通道的颜色变化。", "purpose": "帮助色块沿颜色语义边界分开。"},
            "barrier": {"title": "邻域颜色屏障", "meaning": "水平和垂直邻居的最大 Lab 差异。", "purpose": "控制小区域合并方向。"},
            "initial_labels": {"title": "初始色块标签", "meaning": "Lab 原型量化得到的离散区域。", "purpose": "作为碎片合并前的区域草图。"},
            "labels": {"title": "连通色块标签", "meaning": "合并小碎片后的区域。", "purpose": "生成大面积漫画平涂。"},
            "flat": {"title": "平涂底色", "meaning": "每个区域使用一个代表色。", "purpose": "形成漫画的基础色域。"},
        },
        "tone": {
            "lightness": {"title": "原始明度", "meaning": "Lab L 通道归一化结果。", "purpose": "提供明暗变化来源。"},
            "base": {"title": "区域基础明度", "meaning": "每个色块的中位亮度。", "purpose": "建立稳定的大面明暗。"},
            "detail": {"title": "明度细节残差", "meaning": "原始明度减去区域基础明度。", "purpose": "观察需要保留或抑制的局部细节。"},
            "shadow_candidate": {"title": "阴影候选", "meaning": "区域内部较低分位的明度。", "purpose": "寻找脸颊、鼻侧等阴影位置。"},
            "shadow": {"title": "筛选后阴影", "meaning": "去除孤立噪点后的阴影掩码。", "purpose": "只在稳定区域增加漫画阴影。"},
            "colors": {"title": "底色与阴影", "meaning": "只调整 Lab 的 L 通道后的颜色。", "purpose": "把色块和明暗组合成可绘制颜色。"},
        },
        "compose": {
            "regions": {"title": "放大色块", "meaning": "保持标签边界的最近邻放大结果。", "purpose": "作为画布底层。"},
            "strokes": {"title": "抗锯齿笔画", "meaning": "在放大画布上绘制的线条层。", "purpose": "恢复清晰、连续的手绘轮廓。"},
            "cartoon": {"title": "最终漫画", "meaning": "色块、明暗和线条的合成结果。", "purpose": "输出 RGBA 漫画人像。"},
        },
    }
    for stage, mapping in result.stages.items():
        for name in mapping:
            info.setdefault(stage, {}).setdefault(name, {
                "title": name, "meaning": "该步骤生成的中间特征图。", "purpose": "用于观察算子效果。"})
    for index, sigma in enumerate(result.features.scales):
        for name, meaning, purpose in (
            (f"guide_sigma_{sigma:g}", f"sigma={sigma:g} 的空间平滑引导图。", "分别观察细节尺度和结构尺度。"),
            (f"edge_sigma_{sigma:g}", f"sigma={sigma:g} 的尺度归一化结构强度。", "比较不同尺度对轮廓的响应。"),
            (f"dog_signed_sigma_{sigma:g}", f"sigma={sigma:g} 的有符号 DoG，灰色表示零响应。", "区分局部变暗和变亮。"),
            (f"theta_sigma_{sigma:g}", f"sigma={sigma:g} 的平滑法线方向，颜色表示角度。", "观察该尺度的方向稳定性。"),
            (f"coherence_sigma_{sigma:g}", f"sigma={sigma:g} 的方向一致性。", "判断该尺度的方向是否可信。"),
        ):
            info["features"][name] = {"title": name, "meaning": meaning, "purpose": purpose}
    return info


def write_observer(root: Path, manifest: dict) -> None:
    (root / "index.html").write_text(_html(manifest), encoding="utf-8")


def _html(manifest: dict) -> str:
    data = json.dumps(manifest, ensure_ascii=False)
    css = """<style>
body{font:15px system-ui,sans-serif;margin:0;background:#f6f7fb;color:#20232a}
header{padding:20px 5vw;background:#202c45;color:white}
main{max-width:1180px;margin:22px auto;padding:0 18px}
.route{background:#edf1fa;border-left:4px solid #5074d9;border-radius:8px;padding:11px 14px;line-height:1.8;color:#33415f}
.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0 22px}
.node{border:1px solid #bac7e0;border-radius:14px;background:white;padding:13px 16px;min-width:130px;cursor:pointer;box-shadow:0 2px 8px #15223b12}
.node.active{border-color:#5074d9;box-shadow:0 0 0 3px #5074d933}
.arrow{color:#7a86a2;font-size:24px}
.panel{background:white;border-radius:16px;padding:18px;margin-top:14px;box-shadow:0 3px 14px #15223b10}
.panel h2{margin:0 0 6px}.muted{color:#68728a}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.card{border:1px solid #e1e5ee;border-radius:10px;padding:8px;background:#fbfcff}
.card img{width:100%;height:115px;object-fit:contain;background:#edf0f7;border-radius:7px}
.card b{display:block;margin-top:5px;font-size:13px}.links{margin-top:8px;font-size:12px;color:#68728a}
.op{border-top:1px solid #e1e5ee;padding:10px 0 4px}.op h3{margin:5px 0}.channel-tabs{display:flex;gap:6px;margin:8px 0}.channel-tabs button{border:1px solid #bac7e0;background:#fff;border-radius:7px;padding:5px 13px;cursor:pointer}.channel-tabs button:hover{background:#edf1fa}.hidden{display:none}
</style>"""
    body = """<header><h1>漫画人像 v2 · 观察室</h1><div>五段函数管道：特征 → 线条 / 区域 → 明暗 → 合成。点击节点查看图像证据。</div></header>
<main><div class='route'>特征提取 ├─→ 线条生成 ─────────┐<br>　　　　└─→ 区域生成 ─→ 明暗生成 ─┴─→ 图片合成</div><div id='flow' class='flow'></div><div id='detail' class='panel'></div></main>"""
    script = """<script>const M=""" + data + """;const order=M.architecture.nodes;const flow=document.querySelector('#flow'),detail=document.querySelector('#detail');
function img(stage,name,extra=''){let p=M.images[stage]&&M.images[stage][name];let i=(M.image_info[stage]||{})[name]||{};return p?`<div class='card ${extra}' data-name='${name}'><a href='${p}' target='_blank'><img loading='lazy' src='${p}'></a><b>${i.title||name}</b><div class='muted'>${i.meaning||''}</div><div class='links'>${i.purpose||''}<br>尺寸：${(M.shapes[stage+'.'+name]||[]).join(' × ')}</div></div>`:''}
function gallery(refs){return `<div class='gallery'>${(refs||[]).map(r=>img(r.stage,r.name)).join('')}</div>`}
function channelGallery(stage,names,id){return `<div class='channel-tabs'>${names.map((n,i)=>`<button onclick="pickChannel('${id}','${n}')">${n.split('_')[1]||n}</button>`).join('')}</div><div id='${id}' class='gallery'>${names.map((n,i)=>img(stage,n,i?'hidden':'')).join('')}</div>`}
function pickChannel(id,name){document.querySelectorAll('#'+id+' .card').forEach(x=>x.classList.toggle('hidden',x.dataset.name!==name))}
function opGallery(stage,op,index){let channels=op.outputs.filter(x=>/^response_[RGB]_/.test(x));let rest=op.outputs.filter(x=>!channels.includes(x));return gallery(rest.map(name=>({stage,name})))+(channels.length===3?channelGallery(stage,channels,'channel_'+index):'')}
function show(id){const n=order.find(x=>x.id===id)||order[0];document.querySelectorAll('.node').forEach(x=>x.classList.toggle('active',x.dataset.id===id));let ops=(n.ops||[]).map((op,i)=>`<div class='op'><h3>${i+1}. ${op.title}</h3><div class='muted'>${op.description}</div>${opGallery(id,op,i)}</div>`).join('');detail.innerHTML=`<h2>${n.title}</h2><div class='muted'>${n.purpose}</div><h3>输入图</h3>${gallery(n.inputs)}<h3>输出图</h3>${gallery(n.outputs)}<h3>内部算子与输出</h3>${ops}`}
order.forEach((n,i)=>{let el=document.createElement('div');el.className='node';el.dataset.id=n.id;el.innerHTML=`<b>${n.title}</b><div class='muted'>${n.purpose}</div>`;el.onclick=()=>show(n.id);flow.appendChild(el);if(i<order.length-1){let a=document.createElement('span');a.className='arrow';a.textContent='→';flow.appendChild(a)}});show('features');</script>"""
    return "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>CNN Cartoon v2 观察室</title>" + css + "</head><body>" + body + script + "</body></html>"
