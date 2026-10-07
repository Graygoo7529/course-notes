# CNN 人像漫画化 v2

v2 是一次干净的、可解释的函数式实现。旧的 `cnn_cartoon_portrait` 保留为 v1 对照，不依赖训练数据，也不把每一种实验算子都塞进生产流程。

## 五段管道

1. **特征提取**：对 RGB 做一次空间可分离高斯平滑；逐通道 Sobel 卷积得到 `gx_R/G/B`、`gy_R/G/B`，用结构张量融合出可视的 `edge / theta / coherence`；用亮度 DoG 得到 `dark`，用 Lab 邻域差得到 `color_barrier`。
2. **线条生成**：沿 `theta` 做非极大值抑制，融合 `edge` 与 `dark`，滞后阈值保证连续性，再做骨架化和方向一致的短间隙连接。
3. **区域生成**：在 Lab 空间以固定随机种子的少量原型量化颜色，再利用 `color_barrier` 把小碎片合并到相邻色域；`--palette-style` 可选 `natural / warm / pastel / noir`，只调整原型色的色相、饱和度和明度。
4. **明暗生成**：每个区域先取中位亮度形成 `base`，再提取 `detail` 与受控 `shadow`；阴影只改变 L 通道，保留漫画色彩。
5. **图片合成**：在放大画布上用抗锯齿线条绘制，区域底色、阴影和线条合成为 RGBA 输出。这里的放大是插值和高分辨率绘制，转置卷积与 PixelShuffle 在无训练参数的任务中没有必要。

## 运行

需要一张照片和同尺寸前景遮罩（白色为人物）：

```powershell
python run.py --input photo.jpg --mask mask.png --output outputs/v2-selfie --view=true
```

也可先生成，再单独打开观察室：

```powershell
python inspect_run.py --run outputs/v2-selfie --view=true
```

观察室先显示五个主函数和结果图；点击节点后显示该函数的内部算子图。所有中间图都在 `outputs/v2-selfie/images`，可直接放大查看。
