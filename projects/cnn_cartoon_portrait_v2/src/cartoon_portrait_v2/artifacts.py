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
        save_rgb(path, value)
    elif value.dtype == bool:
        cv2.imwrite(str(path), (value.astype(np.float32) > 0).astype(np.uint8) * 255)
    elif np.issubdtype(value.dtype, np.integer):
        # Labels are categorical: a grayscale normalization would make label 0
        # indistinguishable from the outside mask, so use a stable false colour map.
        labels = value.astype(np.int32)
        out = np.zeros((*labels.shape, 3), np.uint8)
        palette = np.array([[232, 92, 80], [84, 156, 230], [104, 190, 120],
                            [224, 179, 74], [166, 118, 206], [92, 188, 188]], np.uint8)
        for i in range(len(palette)):
            out[labels == i] = palette[i]
        cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    else:
        save_gray(path, value, mask)
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
            rel = _write(path, np.asarray(value), result.features.mask if value.ndim < 3 else None)
            images[stage][name] = str(Path(rel).relative_to(root).as_posix())
            shapes[f"{stage}.{name}"] = list(np.asarray(value).shape)
    save_rgb(root / "input.png", result.features.image)
    cv2.imwrite(str(root / "mask.png"), result.features.mask.astype(np.uint8) * 255)
    save_rgb(root / "cartoon.png", result.rendered.rgb)
    np.savez_compressed(root / "features.npz", edge=result.features.edge, theta=result.features.theta,
                        coherence=result.features.coherence, dark=result.features.dark,
                        lightness=result.features.lightness)
    params = asdict(config)
    (root / "parameters.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    nodes = [
        {"id": "features", "title": "特征提取", "purpose": "用一次平滑、逐通道 Sobel 与结构张量提取可解释特征。", "uses": ["input"], "produces": ["guide", "edge", "theta", "dark", "color_barrier"], "ops": ["空间可分离高斯：input → guide", "逐通道 Sobel 卷积：guide → gx_R / gx_G / gx_B", "结构张量逐通道平方与加权融合：gx, gy → edge / theta", "亮度 DoG：lightness → dark", "Lab 邻域差：guide → color_barrier"]},
        {"id": "lines", "title": "线条生成", "purpose": "沿方向响应细化线条，并以滞后阈值保留连续笔画。", "uses": ["edge", "theta", "dark"], "produces": ["retained", "skeleton", "bridged"], "ops": ["方向非极大值抑制：edge, theta → nms_edge", "暗部与结构融合：nms_dark, nms_edge → score", "滞后阈值：score → retained", "骨架化与方向一致短桥：retained → bridged"]},
        {"id": "regions", "title": "区域生成", "purpose": "在 Lab 空间形成少量色块，并用颜色屏障合并碎片。", "uses": ["lab_lightness", "barrier"], "produces": ["labels", "flat"], "ops": ["Lab 原型量化：lab → initial_labels", "颜色屏障约束：initial_labels, barrier → labels", "区域中位色：labels → flat"]},
        {"id": "tone", "title": "明暗生成", "purpose": "把区域底色和局部阴影分开，控制明暗细节。", "uses": ["lightness", "labels"], "produces": ["base", "detail", "shadow", "colors"], "ops": ["区域中位亮度：lightness, labels → base", "明度残差：lightness − base → detail", "区域分位数筛选：lightness, labels → shadow", "只调整 L 通道：flat, shadow → colors"]},
        {"id": "compose", "title": "图片合成", "purpose": "放大画布、抗锯齿绘制笔画并合成 RGBA 图像。", "uses": ["flat", "strokes", "colors"], "produces": ["cartoon"], "ops": ["最近邻放大色块、抗锯齿绘制笔画：flat, strokes → layers", "线条与色块 alpha 合成：layers → cartoon"]},
    ]
    manifest = {"version": "v2", "input": input_path, "mask": mask_path, "images": images,
                "shapes": shapes, "architecture": {"nodes": nodes,
                "edges": [["features", "lines"], ["features", "regions"], ["regions", "tone"], ["lines", "compose"], ["tone", "compose"]]}}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_observer(root, manifest)
    return root


def write_observer(root: Path, manifest: dict) -> None:
    (root / "index.html").write_text(_html(manifest), encoding="utf-8")


def _html(manifest: dict) -> str:
    data = json.dumps(manifest, ensure_ascii=False)
    return """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>CNN Cartoon v2 观察室</title>
<style>body{font:15px system-ui,sans-serif;margin:0;background:#f6f7fb;color:#20232a}header{padding:20px 5vw;background:#202c45;color:white}main{max-width:1180px;margin:22px auto;padding:0 18px}.route{background:#edf1fa;border-left:4px solid #5074d9;border-radius:8px;padding:11px 14px;line-height:1.8;color:#33415f}.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0 22px}.node{border:1px solid #bac7e0;border-radius:14px;background:white;padding:13px 16px;min-width:130px;cursor:pointer;box-shadow:0 2px 8px #15223b12}.node.active{border-color:#5074d9;box-shadow:0 0 0 3px #5074d933}.arrow{color:#7a86a2;font-size:24px}.panel{background:white;border-radius:16px;padding:18px;margin-top:14px;box-shadow:0 3px 14px #15223b10}.panel h2{margin:0 0 6px}.muted{color:#68728a}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}.card{border:1px solid #e1e5ee;border-radius:10px;padding:8px;background:#fbfcff}.card img{width:100%;height:115px;object-fit:contain;background:#edf0f7;border-radius:7px}.card b{display:block;margin-top:5px;font-size:13px}.links{margin-top:8px;font-size:12px;color:#68728a}</style></head><body><header><h1>漫画人像 v2 · 观察室</h1><div>五段函数管道：特征 → 线条 / 区域 → 明暗 → 合成。点击节点查看图像证据。</div></header><main><div class='route'>特征提取 ├─→ 线条生成 ─────────┐<br>　　　　└─→ 区域生成 ─→ 明暗生成 ─┴─→ 图片合成</div><div id='flow' class='flow'></div><div id='detail' class='panel'></div></main>
<script>const M=""" + data + """;const order=M.architecture.nodes;const flow=document.querySelector('#flow'),detail=document.querySelector('#detail');
function img(stage,name){let p=M.images[stage]&&M.images[stage][name];return p?`<div class='card'><a href='${p}' target='_blank'><img loading='lazy' src='${p}'></a><b>${name}</b><div class='links'>${(M.shapes[stage+'.'+name]||[]).join(' × ')}</div></div>`:''}
function show(id){const n=order.find(x=>x.id===id)||order[0];document.querySelectorAll('.node').forEach(x=>x.classList.toggle('active',x.dataset.id===id));let g=M.images[id]||{};let names=Object.keys(g);detail.innerHTML=`<h2>${n.title}</h2><div class='muted'>${n.purpose}</div><div class='links'>输入特征：${n.uses.join('、')}　输出：${n.produces.join('、')}</div><div class='links' style='margin-top:10px'><b>内部算子：</b>${(n.ops||[]).map((x,i)=>(i+1)+'. '+x).join('　')}</div><div class='gallery' style='margin-top:14px'>${names.map(x=>img(id,x)).join('')}</div>`}
order.forEach((n,i)=>{let el=document.createElement('div');el.className='node';el.dataset.id=n.id;el.innerHTML=`<b>${n.title}</b><div class='muted'>${n.purpose}</div>`;el.onclick=()=>show(n.id);flow.appendChild(el);if(i<order.length-1){let a=document.createElement('span');a.className='arrow';a.textContent='→';flow.appendChild(a)}});show('features');</script></body></html>"""
