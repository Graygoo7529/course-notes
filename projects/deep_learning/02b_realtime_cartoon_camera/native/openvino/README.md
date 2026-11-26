# OpenVINO 模型与原生探针

## 特征模型

使用上一项目固定 revision 的 `mobilenetv3_large_100.ra_in1k`。导出三层特征，不导出分类 logits。输入为 RGB、NCHW、先除 255 再按 manifest 的 mean/std 归一化；权重存储 FP32。

```powershell
conda activate d2l
python python/export_openvino.py --size 256
python python/export_openvino.py --size 384
```

| 画布 | stride 4 | stride 8 | stride 16 |
| --- | --- | --- | --- |
| 256 | 1×24×64×64 | 1×40×32×32 | 1×112×16×16 |
| 384 | 1×24×96×96 | 1×40×48×48 | 1×112×24×24 |

转换产物与 manifest 在被忽略的 `models/`。S/Q、几何和渲染仍在宿主端。当前桌面 OpenVINO worker 在 CPU 上显式使用 f32 精度提示和 4 个推理线程。

## C++ 单图探针

`probe.cpp` 使用 OpenVINO C++ API 加载模型、读取预归一化 float32 输入、预热、重复推理并输出三个张量。它没有 Python、图像解码或 UI 依赖，可用于 Android 运行时的第一道检查；不宣称已实现手机漫画渲染。

本机已执行以下流程，三层张量与 Python OpenVINO 输出逐值一致：

```powershell
$cmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ov = 'C:\Anaconda3\envs\d2l\Lib\site-packages\openvino'
& $cmake -S native/openvino -B .cache/native-build -G 'Visual Studio 17 2022' -A x64 "-DOpenVINO_DIR=$ov/cmake"
& $cmake --build .cache/native-build --config Release
python python/native_probe.py prepare --size 256
$env:Path = "$ov\libs;" + $env:Path
& .cache/native-build/Release/cartoon_openvino_probe.exe models/mobilenetv3_features_256.xml .cache/native_probe_256/input.f32 .cache/native_probe_256/actual 20
python python/native_probe.py compare --size 256
```

切换到 384 时替换模型路径、探针目录，并用 `prepare/compare --size 384`。原生可执行文件的输入字节数必须与模型完全一致；比较脚本检查三层 shape 和逐元素容差，而不是只看均值。

Android 需要交叉编译此探针并配套 ARM64 OpenVINO 库，见 [Android 文档](../android/README.md)。
