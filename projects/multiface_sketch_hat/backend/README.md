# 后端结构

当前仅预留模块位置，尚无可启动的 API 服务。

| 位置 | 后续职责 |
| --- | --- |
| `app/api/` | FastAPI 生命周期、二进制 WebSocket、冻结拍照 |
| `app/detectors/` | YOLO5Face 加载、letterbox、NMS、五点坐标还原 |
| `app/sketchers/` | Informative Drawings / U²-Net Portrait 的独立适配器 |
| `app/tracking/` | 几何关联、trackId、丢失状态与帽子绑定 |
| `app/geometry/` | 裁剪变换 T、关键点映射、时间平滑与占据遮罩 |
| `app/compositing/` | 帽口锚点、前后层合成、同帧多人画布与个人卡片 |

输入统一为同一原始帧及不可变的配置快照。检测结果保留 `bbox`、`score`、`landmarks5`；跟踪阶段才加入 `trackId`。人物状态按 ID 独立维护，不能用检测列表的顺序绑定帽子。

模型只在服务启动时加载。第一版通过 PyTorch `eval()` / `inference_mode()` 完成推理，设备显式选择。帽子配置、素描图、变换和最终成片必须对应同一 `frameId`。

上游多个项目都有顶层 `model`、`models`、`utils` 模块，不能把所有目录直接塞进 `sys.path`。适配阶段应隔离其命名空间或整理带来源声明的最小模型定义，防止 YOLO 的 `models` 和其他生成器模块互相覆盖。
