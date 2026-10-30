# Android 运行时准备

## 当前状态

已完成 MobileNetV3 的 256／384 IR、输入 manifest、响应式界面，以及可以交叉编译的 C++ 单图探针。Windows C++ 与 Python 输出已核对；Android SDK/NDK、ARM64 OpenVINO 原生库和已连接手机尚不可用，因此没有 APK、CameraX 或 JNI 实测结果。

依据路线先完成手机单图闸口，再接相机。本项目没有把 Python sidecar 或电脑远程接口当作手机本地推理。

## 运行时闸口

按 [OpenVINO 2026.4.1 Android 构建文档](https://github.com/openvinotoolkit/openvino/blob/2026.4.1/docs/dev/build_android.md)准备工具链。该版本文档使用 NDK r28c；ARM64 需要对应架构的源码构建，不能复制 Windows wheel 或 Android x86-64 库。启用 TBB 时还要构建相同 NDK 的 OneTBB。

1. 用 `adb shell getprop ro.product.cpu.abi` 和 `adb shell getprop ro.build.version.sdk` 确认真实 ABI/API。
2. 固定 OpenVINO 2026.4.1，按官方文档构建 CPU Runtime，记录构建选项与依赖版本。
3. 在仓库运行 `python python/native_probe.py prepare --size 256`，产生相同归一化输入和参照输出。
4. 使用同一工具链编译 `native/openvino` 下的探针。CMake 需提供 `CMAKE_TOOLCHAIN_FILE`、`ANDROID_ABI=arm64-v8a`、适配真实设备的 `ANDROID_PLATFORM`、`ANDROID_STL=c++_shared` 和目标平台 `OpenVINO_DIR`。
5. 将探针、模型 XML/BIN、`input.f32`、OpenVINO CPU 插件和依赖 `.so` 推送到设备专用临时目录；设置该目录的 `LD_LIBRARY_PATH`，运行探针。
6. 拉回 `actual.0.f32`、`actual.1.f32`、`actual.2.f32`，执行 `python python/native_probe.py compare --size 256`，检查形状与差异。ARM64 数值路径可能不同，若容差失败先看实际差异，不直接放宽断言。

## 后续原生相机

通过闸口后，移植相邻特征差、分位数归一化、S/Q、Lab 色板和栅格筛线，再接 CameraX `KEEP_ONLY_LATEST` 与 Tauri Kotlin 插件/JNI。帧处理需考虑 YUV plane stride、pixel stride、旋转和镜像，并及时关闭 ImageProxy。原始帧留在原生侧，React 只接收结果和状态。

手机实时版只用 MobileNetV3。SciPy 样条不直接移植；初版先实现栅格预览，手机精修需要单独实现与核对。参考 [CameraX 分析接口](https://developer.android.com/media/camera/camerax/analyze) 与 [Tauri 移动插件](https://v2.tauri.app/develop/plugins/develop-mobile/)。
