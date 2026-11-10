# 后端结构

离线核心流程和 Web 摄像头 API 已经实现。离线入口是项目根目录的 `scripts/run_core_demo.py`；摄像头入口是 `app/api/main.py`，通过 FastAPI WebSocket 复用同一套检测、素描、跟踪和合成对象。

| 位置 | 实现与职责 |
| --- | --- |
| `app/api/` | FastAPI 生命周期、二进制 WebSocket、冻结拍照 |
| `app/detectors/` | 已实现 YOLO5Face 加载、letterbox、NMS、五点坐标还原 |
| `app/sketchers/` | 已实现 Informative Drawings Lineart fine；U²-Net 保留为后续候选 |
`app/tracking/` | 已实现几何关联、trackId、丢失状态与帽子绑定 |
`app/geometry/` | 已实现扩大人脸裁剪；跟踪状态按 ID 维护并与合成同帧 |
| `app/compositing/` | 已实现淡彩帽子、旋转缩放、整帧场景和逐人卡片 |


## 启动

在项目根目录执行：

```powershell
conda activate d2l
python -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000
```

WebSocket 地址为 `ws://127.0.0.1:8000/ws/process`。客户端发送 JPEG 二进制帧，服务端返回“4 字节 JSON 头长 + JSON 头 + JPEG 场景图”的响应；发送 `{"kind":"capture"}` 会返回同一帧的逐人卡片。
输入统一为同一原始帧及不可变的配置快照。检测结果保留 `bbox`、`score`、`landmarks5`；跟踪阶段才加入 `trackId`。人物状态按 ID 独立维护，不能用检测列表的顺序绑定帽子。

模型只在服务启动时加载。第一阶段通过 PyTorch `eval()` / `inference_mode()` 完成推理，设备显式选择。YOLOv5-Face 完整 checkpoint 仅允许在核实来源后显式开启可信加载；Lineart 权重使用 `weights_only=True`。帽子配置、素描图、变换和最终成片必须对应同一 `frameId`。

上游多个项目都有顶层 `model`、`models`、`utils` 模块，不能把所有目录直接塞进 `sys.path`。适配阶段应隔离其命名空间或整理带来源声明的最小模型定义，防止 YOLO 的 `models` 和其他生成器模块互相覆盖。
