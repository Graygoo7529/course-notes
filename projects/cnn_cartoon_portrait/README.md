# 自拍人像漫画化

输入一张自拍，交互标记人物，用固定卷积与图像处理输出透明背景漫画。当前默认流程为：

**逐通道轻度去噪与方向响应 → 结构融合 → 保边色块与线条双分支 → RGBA 合成。**

核心计算保持原图尺寸，在 CPU 上运行，无需训练集、预训练模型或显卡。GrabCut 根据当前照片估计颜色模型，需要框选或前景/背景笔划。第一版仍可通过 `--pipeline baseline` 运行，便于比较。

## 快速开始

~~~powershell
conda activate d2l
cd 'B:\LearnSpace\Course Notes\projects\cnn_cartoon_portrait'
python run.py --demo
~~~

这会处理内置合成图，并生成新结果目录。本机依赖位于项目 `.deps`，入口会自动加载；新环境先安装：

~~~powershell
python -m pip install --target .deps -r requirements-dev.txt
~~~

依赖版本固定在 `requirements.txt`，开发依赖包含 ty。当前使用 Python 3.13；个人照片、生成结果及本地依赖均被 Git 忽略。

## 处理自拍

~~~powershell
python run.py --input "A:\AogoDesk\Pictures\一寸照片\一寸照片.jpg" --interactive
~~~

小图预览最多放大 4 倍，并受显示尺寸限制。例如 $150\times197$ 可显示为 $600\times788$。照片用双线性插值，标签用最近邻；处理、笔划和导出均使用原图坐标。`--preview-scale 1` 可关闭预览放大。EXIF 方向在标记前校正。

| 操作 | 用途 |
| --- | --- |
| `R`，鼠标拖动 | 框住完整人物，重新框选会重置笔划 |
| `F` / `B`，鼠标涂画 | 标记确定前景 / 确定背景 |
| `[` / `]` | 调整画笔半径，单位为原图像素 |
| `Enter` | 执行分割；补画后需要再次计算 |
| `Space` | 接受最近一次计算且未再修改的遮罩 |
| `C` / `Esc` | 清空标记 / 取消 |

矩形之外默认是确定背景，框选要包含完整头顶、耳朵和希望保留的肩部，可以接触图像边缘。也可先 `C` 清空，再分别添加 `F/B` 笔划，不必先框矩形。说明栏和留白不接受新的标记。

## 复用与对照

确认遮罩后，可以跳过交互和 GrabCut，单独比较漫画效果：

~~~powershell
python run.py --input outputs/20261006-231358-781351/input.png --mask outputs/20261006-231358-781351/mask.png
~~~

已有遮罩只保留了哪些像素，后续风格处理就只能使用这些像素；旧结果中缺少头顶和肩部，需要重新标记原图来修正。

`--rect X Y W H` 可用原图坐标初始化；`--labels` 复用 GrabCut 的四类标签；`--mask` 复用二值前景，三种方式互斥。`--output` 指定新结果目录，程序拒绝覆盖已有目录。

~~~powershell
python run.py --input data/input/selfie.jpg --mask outputs/your-run/mask.png --pipeline baseline
python run.py --input data/input/selfie.jpg --mask outputs/your-run/mask.png --no-thin
~~~

第一条运行原有高斯、硬明度量化和灰度 Sobel；第二条仅关闭新版的线条细化。对照时应固定照片、遮罩与处理尺寸，并记录阈值变化。

## 参数

以下为结构路线的默认值；第一版保留自己的参数默认值。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--sigma` | 0.6 | 提取结构前的轻度高斯，单位原图像素；0 关闭 |
| `--operator` | sobel | 可改为 scharr；均按单位坡度归一化 |
| `--channel-weights R G B` | 各 1/3 | 非负通道融合权重，和须为 1 |
| `--luma-smoothness` / `--chroma-smoothness` | 8 / 1.5 | 基础明度 / 色度的保边平滑强度 |
| `--levels` / `--amount` | 6 / 0.35 | 明度档位数 / 分层强度 |
| `--transition` | 0.25 | 连续过渡半宽相对档距，范围 $(0,0.5]$ |
| `--detail` | 0.35 | 保留的局部明度起伏，0 全部舍弃，1 全部回填 |
| `--threshold` / `--high-threshold` | 0.025 / 0.06 | 连通筛选的低 / 高阈值，要求 $0<低<高$ |
| `--softness` / `--ink-gamma` | 0.12 / 0.9 | 着墨强度曲线的宽度 / 指数 |
| `--strength` | 0.55 | 最终描线深浅；0 不叠加线条 |
| `--outline-width` | 0 | 可选内轮廓宽度，0 关闭 |
| `--iterations` | 5 | GrabCut 迭代次数，复用遮罩时不参与计算 |

颜色边界权重另有 `--tau-lightness`、`--tau-chroma`、`--tau-gradient`，默认 8、12、0.12，分别对应 Lab 明度差、色度差、归一化方向变化。增大尺度会减弱该项对平滑的阻断；不要把不同单位的参数直接比较。

脸部黑线过多时，先提高筛选阈值；只是线条过深时，降低 `strength` 或增大 `softness`。明度分界过强时，降低 `amount` 或增大 `transition`。先固定遮罩，一次改变一个主要因素。`python run.py --help` 显示全部参数。

## 输出与特征

每次运行在 `outputs` 下创建新目录。结果以非预乘 RGBA 保存，透明度只在背景合成时应用一次。

| 文件 | 内容 |
| --- | --- |
| `cartoon.png` | 最终透明漫画 |
| `overview.png` | 原图、遮罩、轻度去噪、色块、线条、合成六步对照 |
| `features_overview.png` | RGB 三通道、灰度路线、幅值加权、结构矩阵融合的共同尺度对照 |
| `preview_light.png` / `preview_dark.png` | 浅、深背景下的结果 |
| `input.png` / `mask.png` | 规范化 RGB 输入副本 / 二值遮罩 |
| `guide.png` / `base.png` | 轻度去噪引导图；后者保留为旧接口兼容名称 |
| `colors.png` / `lines.png` | 色块 / 白底黑线预览，线条 PNG 为 $1-E$ |
| `base_lightness.png` / `detail_lightness.png` / `mapped_lightness.png` | 基础明度 / 有符号细节显示 / 映射并回填后的明度 |
| `response_*.png` / `coherence.png` | 各路线结构响应 / 方向一致性 |
| `boundary_weights.png` | 邻接权重的平均值预览；暗处较少平滑 |
| `lines_thinned.png` / `lines_retained.png` / `outline.png` | 细化响应 / 连通筛选位置 / 可选内轮廓 |
| `features.npz` | 浮点方向响应、结构分量、角度、有效性、两向边界权重、明度分解和线条等原始数组 |
| `selection_labels.png` | 交互或矩形初始化标签 0/1/2/3，复用 mask 时不生成 |
| `parameters.json` | 实际配置、来源、尺度、依赖版本、处理耗时及求解残差 |

有符号梯度与细节应从 `features.npz` 读取；PNG 使用可视化映射，不能当作原始数值。六张响应图共用同一显示尺度，细节图中灰表示零，两者尺度均记入参数文件。背景是无效区，以保存的 mask 判断。

第一版对照只导出原有六步结果，不包含新版特征文件。默认参数在一张 $150\times197$ 照片上进行了观察与调整，尚不能代表多图效果；放大的线条预览也不会恢复原图缺失的细节。

## 函数管道

[workflow.py](src/cartoon_portrait/workflow.py) 中的 `stylize` 是无文件副作用的计算入口，其内部保持：

~~~python
features = extract_channel_features(image, mask, config.features)
structure = fuse_features(features, config.fusion)
colors, color_debug = make_colors(image, mask, structure, config.colors)
lines, line_debug = make_lines(structure, mask, config.lines)
rgba = compose_rgba(colors, lines, mask.astype("float32"), config.strength)
~~~

| 模块 | 职责与课程联系 |
| --- | --- |
| [features.py](src/cartoon_portrait/features.py) | 逐通道高斯与 Sobel/Scharr；非线性特征后 PW 融合，求结构强度与方向 |
| [colors.py](src/cartoon_portrait/colors.py) | 颜色/梯度边界权重、WLS、软明度映射和 RGB 重建 |
| [lines.py](src/cartoon_portrait/lines.py) | 方向细化、插值、连通筛选与着墨 |
| [pipeline.py](src/cartoon_portrait/pipeline.py) | 共享图像接口、遮罩高斯、GrabCut，以及第一版对照 |
| [selection.py](src/cartoon_portrait/selection.py) | 纯预览/坐标函数与桌面事件层 |
| [cli.py](src/cartoon_portrait/cli.py)、[artifacts.py](src/cartoon_portrait/artifacts.py) | 配置、文件入口与中间结果导出 |

配置和结果使用具名数据类，计算函数不修改输入。保边平滑用 Jacobi 预条件共轭梯度，四邻域差分隐式执行稀疏矩阵，无新增依赖；迭代使用 float64，检查真实残差后输出 float32。超出迭代上限会明确报错。

高斯的横纵分解属于空间可分离，RGB 独立处理属于 DW，非线性结构特征的通道汇总对应 PW。WLS、连通筛选、颜色变换与 RGBA 合成各有自己的数学作用，整个系统不是一个训练好的 CNN。

默认假定输入为 sRGB，不执行 ICC 色彩管理。大图仍保持原分辨率，耗时和内存随像素数增加；当前 alpha 为二值，精细发丝与半透明边缘尚未处理。

## 验证与记录

~~~powershell
conda activate d2l
python -m unittest discover -s tests -v
$env:PYTHONPATH = "$PWD\.deps;$PWD\src"
python -m ty check
~~~

验证覆盖颜色边界抵消、相同通道退化、方向旋转、常量与背景隔离、WLS 与独立线性系统解的比较、未收敛报错、明度恒等重建、软映射、连通筛选、单行/单列输入，以及整数/非整数预览坐标和模拟窗口事件。实际照片结果见实施记录，模拟交互检查不等同于人工鼠标体验验证。

- [研究方案与设计](研究方案与设计.md)：任务定义、第一版与方法背景。
- [下一版设计](下一版设计.md)：本次实现采用的结构、公式和接口约定。
- [实施记录](实施记录.md)：运行证据、效果观察与限制。
