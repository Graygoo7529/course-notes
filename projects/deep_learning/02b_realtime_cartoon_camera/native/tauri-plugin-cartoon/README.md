# 手机本地漫画插件

Rust 命令转发到 Kotlin 的串行执行器：`initialize`、`process_frame`、`reset`、`stop`、`save_png`。Android 使用 ONNX Runtime、OpenCV 和 MediaPipe，模型来源为项目 `models/android`，由 Gradle 直接加入 assets。

`CartoonMathTest` 用 PyTorch 导出的固定输入/输出核对单位化特征差，并检查归一化的填边排除与细化连通性。它不需要连接手机；完整摄像头/原生库运行仍需真机验证。

构建与安装见 [Android 文档](../android/README.md)。
