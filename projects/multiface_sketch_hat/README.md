# 多人素描与虚拟戴帽

深度学习应用作业：浏览器获取摄像头画面，使用 YOLOv5s-Face 检测多人，为各人生成素描，再按头部位置、尺度和倾斜佩戴二维帽子。

**当前阶段：离线核心算法与 Web 摄像头闭环均已打通，后续可继续做性能与视觉细化。**

## 路线

```mermaid
flowchart LR
    A[浏览器摄像头] --> B[YOLOv5s-Face 人脸框与五点]
    B --> C[关联 ID 与平滑裁剪]
    C --> D[预训练素描生成器]
    C --> E[帽口锚点与独立人像遮罩]
    D --> F[分层帽子合成]
    E --> F
    F --> G[多人画面与个人卡片]
```

主线为 React + TypeScript + Vite、FastAPI 本机服务、PyTorch 推理。当前已确定 Informative Drawings 的 `anime_style`（Lineart fine）作为主素描生成器；Canny/XDoG 保留在原实验包中作为后续对照，当前 API 使用 Lineart fine；U²-Net Portrait 暂作可选遮罩候选。沿用原技术路线，不训练或微调，不把普通 YOLO COCO 权重当成人脸检测器。

从 [环境与模型准备](docs/环境与模型准备.md) 开始安装；[方案评审](docs/方案评审.md) 记录可行性与实施边界。核心算法可用下面的命令运行。原始设计位于仓库的 `references/多人素描与虚拟戴帽技术路线.md`，未修改。

```powershell
conda activate d2l
python scripts/run_core_demo.py --scene-size 512 --cards
```

输出保存在被忽略的 `outputs/core_demo/`：检测框、整帧线稿、淡彩帽子场景和逐人卡片。示例输入为公开的 `data/input/grace_hopper.jpg`。

## 启动 Web 摄像头

先在一个终端启动后端（首次启动会加载本地模型）：

```powershell
conda activate d2l
$env:MULTIFACE_DEVICE = "cuda"
python -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000
```

再在另一个终端启动前端：

```powershell
npm.cmd --prefix frontend run dev
```

打开 Vite 输出的本地地址（通常为 `http://127.0.0.1:5173`），允许浏览器访问摄像头即可看到整帧多人素描。点击“拍摄逐人卡片”会基于同一帧生成各人头像卡片。前端以约 90 ms 的节奏尝试发送 JPEG 帧，并用 in-flight 闸门避免堆积；后端返回带 `frameId`、处理耗时、检测框、五点坐标和 `trackId` 的二进制响应。GPU 可用时保持 640 像素宽度和 Lineart 全质量更新。

## 目录

```text
backend/                 Python 服务及算法模块
  app/                   第一阶段检测、生成、几何和合成；第二阶段 API/跟踪
frontend/                React / TypeScript / Vite 最小页面
assets/hats/             可提交的帽子素材、锚点与来源说明
config/models.json       官方模型来源与预期文件位置
scripts/                 环境、模型文件准备检查
docs/                    方案评审与安装说明
tests/                   后续几何、协议与同帧合成验证
data/input/              本地示例图和视频，整体忽略
model/                   下载的权重与压缩包，整体忽略
third_party/             官方模型源码，整体忽略
outputs/                 对照图、导出成片，整体忽略
.cache/                  环境快照和测试截图，整体忽略
```

默认用 `conda activate d2l`。`requirements.txt` 不主动替换现有 torch/torchvision；GPU 选项单独列在安装说明。模型在后端启动时加载，浏览器只负责摄像头取帧、场景显示和拍照卡片展示。

## 实施顺序

1. 已完成模型准备与离线核心流程：YOLOv5-Face、Lineart fine、淡彩帽子和逐人卡片。
2. 用公开示例图和有来源的多人图检查检测、裁剪、线稿与帽子层次。
3. 接入摄像头，完成逐人 ID、几何平滑、单帧在途和冻结拍照。
4. 将整帧场景作为 Web 主视图，拍摄时生成同一帧的逐人卡片。
5. 已根据 GPU 测速结果启用高频摄像头发帧；CPU 回退时可用 `MULTIFACE_SKETCH_INTERVAL_MS` 缓存素描结果。

演示输入默认使用已有公开示例 `../cnn_face_cartoon/data/input/grace_hopper.jpg` 的本地副本。多人测试后续使用有来源的公开图片或生成素材；复制同一个头像只能检查多框与调度，不能证明不同人物的关联稳定性。摄像头帧与测试截图不进入文档资源目录。
