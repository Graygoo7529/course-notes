# 自拍人像漫画化

输入一张自拍，交互标记人物，用固定卷积和颜色处理输出透明背景的漫画人像。第一版已经实现：**GrabCut 分割 → 前景高斯平滑 → 色块、线条双分支 → RGBA 合成**，无需训练数据或预训练模型。

核心计算保持原图尺寸，在 CPU 上运行，不依赖显卡。这里直接应用的是卷积算子及图像处理知识，还不是一个 CNN。GrabCut 会在当前图片上估计颜色模型；它不是语义人像识别器，需要框选或笔划提示。

## 快速开始

在项目目录运行：

~~~powershell
conda activate d2l
cd 'B:\LearnSpace\Course Notes\projects\cnn_cartoon_portrait'
python run.py --demo
~~~

该命令使用内置的**合成测试图**跑通流程，并打印结果目录。当前机器的依赖已安装在本项目 `.deps`，`run.py` 会自动加载；无需重复安装。复制项目到新环境时，先执行：

~~~powershell
python -m pip install --target .deps -r requirements-dev.txt
~~~

项目使用 Python 3.13，依赖版本固定在 `requirements.txt`，开发依赖额外包含 ty。依赖、个人照片和生成结果均被 Git 忽略。

## 处理自拍

将照片放入 `data/input`，例如 `selfie.jpg`，然后运行：

~~~powershell
python run.py --input data/input/selfie.jpg --interactive
~~~

窗口只缩小显示预览，标记会映射回原图坐标。处理结果保留原始尺寸；相机 EXIF 方向在框选前已校正。

| 操作 | 用途 |
| --- | --- |
| `R`，鼠标拖动 | 框住完整人物；重新框选会重置已有笔划 |
| `F`，鼠标涂画 | 标记确定前景，例如脸、头发、衣服 |
| `B`，鼠标涂画 | 标记确定背景，例如两臂之间的空隙 |
| `[` / `]` | 减小 / 增大画笔 |
| `Enter` | 运行分割；补画后再次按下可修正 |
| `Space` | 接受当前分割并生成漫画 |
| `C` | 清空所有标记 |
| `Esc` | 取消 |

先检查头发和肩部是否完整，再接受遮罩。矩形之外默认是确定背景，因此框要包含完整人物；**矩形可以接触图像边缘**。若不适合用矩形，按 `C` 后分别涂少量 `F` 和 `B`：此时未标记区域为可能背景，不会强制删除四周像素。

需要无窗口运行时，可以给出 `--rect X Y W H`，其中数值是校正方向后的原图坐标；或传入已保存的 `--labels` / `--mask`。`python run.py --help` 可查看完整参数。

## 查看结果

每次运行在 `outputs` 下新建一个目录，也可以使用 `--output outputs/experiment-01` 指定。程序拒绝覆盖已有结果目录。

| 文件 | 用途 |
| --- | --- |
| `overview.png` | 原图、遮罩、平滑、色块、描线、合成的六步对照 |
| `cartoon.png` | 最终透明背景 RGBA 图片 |
| `preview_light.png`、`preview_dark.png` | 在白色与深色背景上检查人物边缘 |
| `input.png` | 已校正方向的 RGB 输入副本，供复用遮罩时对齐坐标 |
| `mask.png` | 二值人物遮罩：白色保留，黑色去除 |
| `base.png`、`colors.png` | 平滑与量化结果，人物外的零值只是占位 |
| `lines.png` | 白底黑线预览；文件像素对应 `1-E`，不是函数返回的描线强度 `E` |
| `selection_labels.png` | GrabCut 初始化标签 0/1/2/3，看起来近乎黑色，不是遮罩预览 |
| `parameters.json` | 参数、分割方式、尺寸、来源和依赖版本 |

已有二值遮罩时不再导出初始化标签。最终 PNG 使用普通**非预乘 RGBA**：透明度在查看器合成背景时生效，不能提前再次乘入 RGB。

合成测试图的结果已保存在 [步骤总览](outputs/synthetic-baseline/overview.png)。它验证的是数据流和边界处理，不代表真实自拍的漫画质量。

## 复用与调参

确认人物遮罩后，可使用上次的规范化输入与 `mask.png` 调整风格，跳过 GrabCut：

~~~powershell
python run.py --input outputs/synthetic-baseline/input.png --mask outputs/synthetic-baseline/mask.png --levels 5 --amount 0.75 --threshold 0.045 --output outputs/style-02
~~~

实际自拍同理，替换为自己的运行目录即可。若要重新执行相同的分割初始化，把 `--mask .../mask.png` 换成 `--labels .../selection_labels.png`。

| 参数 | 默认值 | 调节意图 |
| --- | --- | --- |
| `--sigma` | 1.5 | 增大可抑制纹理，也可能模糊五官；单位为原图像素，0 关闭平滑 |
| `--levels` | 6 | 降低可减少明度档数，产生更明显的色块 |
| `--amount` | 0.65 | 增大更接近量化后的明度，0 保留原明度 |
| `--threshold` | 0.035 | 提高可减少弱纹理形成的杂线 |
| `--softness` | 0.08 | 增大可放缓线条强度的过渡；必须大于 0 |
| `--strength` | 0.8 | 增大可加深描线，0 不叠加描线 |
| `--iterations` | 5 | GrabCut 迭代次数；使用 `--mask` 时不参与计算 |

建议先固定遮罩，再一次调整一个参数。若脸部黑线太多，先提高 `threshold`；若颜色分层太生硬，先降低 `amount`。高分辨率照片中的同一像素半径，对应更小的视觉范围，默认参数需要随图片调整。

## 代码与课程

[核心函数](src/cartoon_portrait/pipeline.py)对应设计文档中的七个接口：

~~~python
image = read_image(input_path)
mask = segment_person(image, selection)
base = smooth_foreground(image, mask, sigma=cfg.sigma)
colors = quantize_colors(base, levels=cfg.levels, amount=cfg.amount)
lines = extract_lines(base, mask, threshold=cfg.threshold, softness=cfg.softness)
rgba = compose_rgba(colors, lines, mask.astype("float32"), strength=cfg.strength)
save_png(rgba, output_path)
~~~

`smooth_foreground` 用横纵可分离高斯核，分别卷积前景颜色和遮罩后归一化；RGB 各通道独立处理。`extract_lines` 先用固定权重组合灰度，再计算两个方向的 Sobel 响应。**线条来自平滑图，而不是量化图**，防止新增的明度档位边界被误描成线。

GrabCut、Lab 量化和 RGBA 合成不属于空间卷积。第一版没有上采样，也没有双边滤波、XDoG 或精细发丝抠图。二值 alpha 可能存在锯齿，高斯平滑可能削弱内部边界；这些是下一轮在自拍上检查和改进的重点。

所有颜色计算默认输入是 sRGB，不执行 ICC 色彩管理。灰度输入转为 RGB，带透明通道的输入先铺白底。大图会增加 CPU 计算时间和内存需求；当前没有自动缩小工作尺寸。

## 验证与记录

~~~powershell
conda activate d2l
python -m unittest discover -s tests -v
$env:PYTHONPATH = "$PWD\.deps;$PWD\src"
python -m ty check
~~~

已通过 14 项检查及类型检查，包括前景常量保持、背景颜色隔离、遮罩边缘不产生假线、Sobel 阶跃响应、Lab 量化、非预乘透明度、EXIF 方向、中文路径和结果复用。交互坐标与接受流程通过模拟事件检查。首张实际照片已完成结果分析，小图标记、平滑尺度与明度分层仍需改进，详见实施记录。

- [研究方案与设计](研究方案与设计.md)：概念、公式、课程映射与升级方向。
- [下一版设计](下一版设计.md)：逐通道结构提取与融合、色块和线条双分支、明度映射与小图交互；包含函数契约，尚未实现。
- [实施记录](实施记录.md)：本轮实现范围、验证结果与待观察问题。
