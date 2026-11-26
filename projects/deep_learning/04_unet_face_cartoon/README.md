# UNet 语义分割引导的人脸漫画

《深度学习应用》作业：“规范头像 → UNet 解析 → 分区处理 → photo2cartoon → 输出整理”，不训练或微调。

**已实现静态单人照片的完整管线、可关闭的皮肤简化、可选发色控制和离线四图报告。**
参数通过命令行输入，HTML 用于查看与保存结果，不是浏览器内实时推理应用。

## 本机入口

项目位于 `D:\project\MyCourse\projects\deep_learning\04_unet_face_cartoon`。模型、环境和图片为本机忽略文件，不随 Git 自动传输。

双击 `outputs/assignment-final/index.html` 查看三张公开人像的默认对照；`outputs/hair-final/index.html` 查看发色示例；`outputs/no-face/index.html` 记录无人脸失败分支。文件需在实际运行机器上存在，不随 Git 自动传输。

## 运行

```powershell
# 在项目目录，换成自己的清晰单人照片；输出必须是新的空目录
uv run python run.py --input 'data/input/portrait.jpg' --output outputs/my-portrait

# 关闭分区控制：两张对照图完全相同，只运行一次生成器
uv run python run.py --input 'data/input/portrait.jpg' --skin-strength 0 --output outputs/off

# 可选发色：保留输入亮度，再由生成器决定最终颜色
uv run python run.py --input 'data/input/portrait.jpg' --skin-strength 0.5 --hair-color '#804080' --hair-strength 0.35 --output outputs/tinted

# 复现公开样例
uv run python scripts/download_samples.py
uv run python run.py --demo --device cuda --skin-strength 0.5 --output outputs/demo-new
```

`--input` 支持多个文件；每张必须只有一张有效人脸。没有人脸或有多张时写入失败报告并返回非零退出码，不生成替代滤镜图。`--device auto` 默认有 CUDA 用 CUDA，否则用 CPU；需要强制检查可传 `--device cuda`。首次需要下载网络定义时显式加 `--fetch-source`，普通推理不联网。

输出包含 `index.html`、`run.json`，以及每张照片的四图对照、原图、规范头像、原始标签、标签叠加、处理权重、原始/清理后前景、两份生成输入与漫画。几何正逆矩阵、关键点、模型校验值、参数、耗时和差异统计一并保存。输出目录非空时拒绝覆盖。

## 环境与检查

```powershell
cd projects/deep_learning/04_unet_face_cartoon
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
uv run python scripts/download_models.py sfd
uv run python scripts/download_models.py fan
uv run python scripts/check_models.py unet --fetch-source --device cuda --output outputs/unet_check.json
uv run python scripts/check_models.py photo2cartoon --fetch-source --device cuda --output outputs/cartoon_check.json
```

下载先写 `.part`，校验后提升为正式文件；官方 ZIP 约 216 MiB，只提取指定 checkpoint。`--check` 只读本地文件；`--fetch-source` 显式下载固定 commit 的网络结构，检查哈希后再加载。可选 ONNX 使用 `download_models.py photo2cartoon-onnx`，当前不安装 ONNX Runtime。

## 方法与局限

- 按 EXIF 校正方向，以最长边 1600 的工作图检测和对齐，保留原图到 512 头像的可逆坐标记录。
- UNet 输出原始 19 类标签；皮肤只处理类别 1，五官和鼻子不平滑。权重向区域内部衰减，避免柔化后溢出到眼睛。
- 头像前景先取 1～15，再保留主要皮肤所在连通域。实测这能去除衣领上误判成帽子/头发的孤立块；独立饰品可能被一起去掉，原始掩膜仍保存。
- 原图和处理图共享同一前景，置白后降到 256，分别进入生成器。零强度严格恢复基线；默认不把输入掩膜贴到生成后的脸上。
- 生成器实际只有 256×256；部分照片会明显变白、年轻化或改变五官。清晰照片上的轻度皮肤处理差异很小，不能将差异统计当成质量提升。
- 当前没有训练、多人编辑、摄像头、生成后五官重定位或网页上传服务。

完整方法见 [实现说明](docs/实现说明.md)，样例来源见 [图片来源](docs/图片来源.md)，实测结果见 [本次结果](docs/本次结果.md)。

## 目录

- `src/unet_face_cartoon/`：对齐几何、模型适配、区域处理、管线和报告。
- `scripts/`：环境、下载和模型检查。
- `tests/`：不依赖真人照片的测试。
- `data/`、`models/`、`outputs/`、`.cache/`：本机数据和缓存，忽略。

资料：[技术路线](docs/技术路线.md)、[方案评审](docs/方案评审.md)、[模型准备](docs/模型准备.md)、[环境记录](docs/环境记录.md)。
