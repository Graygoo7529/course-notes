# UNet 语义分割引导的人脸漫画

《深度学习应用》作业：“规范头像 → UNet 解析 → 分区处理 → photo2cartoon → 输出整理”，不训练或微调。

## 环境与检查

```powershell
cd projects/unet_face_cartoon
uv sync --locked
uv run python scripts/check_env.py --require-cuda --output outputs/environment.json
uv run python -m ty check
uv run python -m unittest discover -s tests -v
```

Windows x64、Python 3.12，官方 PyTorch 2.10.0 / torchvision 0.25.0 CUDA 12.8 wheel。无需系统 Python、Conda 或另装 CUDA Toolkit。环境在本项目 `.venv`，锁文件纳入 Git，模型、照片、结果不纳入 Git。

## 模型准备

```powershell
uv run python scripts/download_models.py unet
uv run python scripts/download_models.py photo2cartoon-bundle --extract-generator
uv run python scripts/check_models.py unet --fetch-source --device cuda --output outputs/unet_check.json
uv run python scripts/check_models.py photo2cartoon --fetch-source --device cuda --output outputs/cartoon_check.json
```

下载先写 `.part`，校验后提升为正式文件；官方 ZIP 约 216 MiB，只提取指定 checkpoint。`--check` 只读本地文件；`--fetch-source` 显式下载固定 commit 的网络结构，检查哈希后再加载。可选 ONNX 使用 `download_models.py photo2cartoon-onnx`，当前不安装 ONNX Runtime。

## 目录

- `src/unet_face_cartoon/`：类别编号与图像契约，后续推理管线。
- `scripts/`：环境、下载和模型检查。
- `tests/`：不依赖真人照片的测试。
- `data/`、`models/`、`outputs/`、`.cache/`：本机数据和缓存，忽略。

资料：[方案评审](docs/方案评审.md)、[模型准备](docs/模型准备.md)、[环境记录](docs/环境记录.md)。
