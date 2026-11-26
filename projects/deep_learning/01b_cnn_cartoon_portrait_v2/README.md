# CNN 人像漫画化 v2

v2 是一次干净的、可解释的函数式实现。旧的 `cnn_cartoon_portrait` 保留为 v1 对照，不依赖训练数据，也不把每一种实验算子都塞进生产流程。

## 五段管道

1. **特征提取**：FeatureBank Lite 对 RGB 使用 `σ=(0.6,1.2,2.4)` 三个带遮罩的高斯尺度；逐通道 Sobel 卷积得到 `gx_R/G/B`、`gy_R/G/B`，平滑结构张量后得到每个尺度的 `edge / theta / coherence`，再用 `σ√λ_max` 做尺度归一化；同时保留有符号 DoG、Lab 色度边缘和邻域颜色屏障。观察室会解释每张图是什么、解决什么问题、交给哪个阶段使用。
2. **线条生成**：把遮罩边界、细尺度结构和正向 DoG 暗部拆成外轮廓、内部结构、暗部强调三类候选；用 `theta/coherence` 做方向非极大值抑制和可靠性筛选，再以滞后阈值、拓扑细化、短间隙连接形成连续骨架。`retained` 和 `bridged` 是主要笔画证据，路径追踪只负责把它们转成抗锯齿曲线；最后按来源分配线宽，并在开放端逐渐收笔。
3. **区域生成**：在 Lab 空间以固定随机种子的少量原型量化颜色，再利用 `color_barrier` 把小碎片合并到相邻色域；`--palette-style` 可选 `natural / warm / pastel / noir`，只调整原型色的色相、饱和度和明度。
4. **明暗生成**：每个区域先取中位亮度形成 `base`，再提取 `detail` 与受控 `shadow`；阴影只改变 L 通道，保留漫画色彩。
5. **图片合成**：在放大画布上用抗锯齿线条绘制，区域底色、阴影和线条合成为 RGBA 输出。这里的放大是插值和高分辨率绘制，转置卷积与 PixelShuffle 在无训练参数的任务中没有必要。

## 运行

需要一张照片和同尺寸前景遮罩（白色为人物）：

```powershell
python run.py --input photo.jpg --mask mask.png --output outputs/v2-selfie --view=true
```

需要传输或在另一台电脑上查看时，加上 `--export=true`：

```powershell
python run.py --input photo.jpg --mask mask.png --output outputs/v2-selfie --export=true
```

命令会生成 `outputs/v2-selfie/observer_bundle.zip`。压缩包内含 `index.html`、全部中间特征图、最终漫画、参数、清单和特征数据；解压后直接打开 `index.html` 即可，图片使用相对路径，不依赖原电脑的绝对路径。已有结果也可以重新打包：

```powershell
python inspect_run.py --run outputs/v2-selfie --export=true --view=true
```

观察室本身不需要再打开原始照片的本地绝对路径：页面显示的是压缩包内的 `input.png`、`mask.png` 和 `images/` 下的副本。若浏览器对 `file://` 页面中的脚本有限制，可以用上面的 `--view=true` 通过本地 HTTP 服务打开，显示内容相同。

也可先生成，再单独打开观察室：

```powershell
python inspect_run.py --run outputs/v2-selfie --view=true
```

观察室先显示五个主函数和结果图；点击节点后显示该函数的内部算子图。所有中间图都在 `outputs/v2-selfie/images`，可直接放大查看。
