# 实时漫画相机

课程作业第二阶段：把 `../cnn_face_cartoon` 的 CNN 结构引导算法接入实时摄像头。应用使用 **Tauri 2 + React + TypeScript + Vite**；默认 **MobileNetV3-Large + OpenVINO CPU**，可在启动前切换 PyTorch CPU 作为对照。模型和 MediaPipe 在会话内常驻，不训练、不微调。

## 已实现

- 原图／漫画预览、摄像头权限与错误提示、前后摄像头请求、镜像、暂停、停止、重启。
- 笔画密度与自然／粉彩色板；移动宽度下使用原图／漫画切换布局。
- 256 工作画布实时栅格线条；冻结原始帧后重新进行 384 工作画布的完整样条精修，保存 PNG。
- 单帧在途、忙时跳过采样；丢弃旧会话、旧配置与过时结果，防止延迟积压。
- 本机随机端口、令牌与来源检查的二进制 WebSocket；Tauri 管理 Python worker 和下载目录保存。
- OpenVINO 两档特征模型与 manifest；Python/PyTorch/OpenVINO 对照；可移植 C++ 单图探针。
- 运行信息面板：真实后端、漫画更新率、算法耗时、提交到显示延迟、跳过帧、关键点状态。

Android 本地相机应用尚未实现。目前完成了模型导出、手机宽度界面及原生探针准备；本机尚无可用 Android SDK/NDK、ARM64 OpenVINO 运行库和已连接设备。不能把手机宽度界面或 Windows C++ 测试视为手机推理成功。

## 启动

本机依赖和两个 OpenVINO 模型已经准备好，可直接运行：

```powershell
conda activate d2l
Set-Location 'B:\LearnSpace\Course Notes\projects\realtime_cartoon_camera'
$env:CARTOON_PYTHON = (Get-Command python).Source
$env:Path = 'C:\Users\Aogo\.cargo\bin;' + $env:Path
npm.cmd run tauri dev
```

浏览器开发模式使用 `npm.cmd run dev`，打开 <http://127.0.0.1:1420>。该模式同样连接本机 Python/OpenVINO；前端没有伪装成 CNN 的 Canvas 回退。两种模式任选其一，避免占用相同端口。浏览器模式依赖 Vite 开发插件启动 worker，不能直接双击 `dist/index.html` 使用推理。

Windows 调试可执行文件：`apps/desktop/src-tauri/target/debug/realtime-cartoon-camera.exe`。这是依赖当前仓库、模型和 d2l 环境的课程原型，尚未封装为可在其他电脑独立安装的完整发行包。

## 文档

- [安装与运行](docs/安装与运行.md)：依赖、导出、启动、检查。
- [架构说明](docs/架构说明.md)：算法、通信、调度与生命周期。
- [验证记录](docs/验证记录.md)：实际测试、截图和局限。
- [OpenVINO 与 C++ 探针](native/openvino/README.md)。
- [Android 下一阶段](native/android/README.md)。
- [推荐技术路线](../../references/实时漫画相机应用技术路线.md)。

`data/`、`models/`、`.cache/`、`node_modules/`、构建产物均忽略。已有课程笔记和上阶段算法文件保持原样，通过路径导入复用。
