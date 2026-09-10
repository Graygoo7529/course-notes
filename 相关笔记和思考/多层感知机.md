> 对 《动手学深度学习》 的学习笔记
> 主要内容为知识的概述、习题、自己的思考
> 关注原理、减少对于代码实现的考虑

参考：[多层感知机 — 动手学深度学习](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/index.html)

## 多层感知机基础
### 隐藏层
回顾仿射变换： $\hat{y} = w_{1}x_{1} + ... + w_{d} x_{d} +b.$
线性回归通过单个仿射变换将输入直接映射到输出，这意味着每个特征对模型输出的影响是线性的、可加的。在一些问题中，可以通过特征预处理，使变换后的特征与输出之间更接近线性关系；但对图像、语音、文本等复杂数据，这种线性假设过于受限。

例如在猫狗图像分类中，某个位置像素强度的增加，并不总是稳定地提高或降低图像属于某一类别的可能性。单个像素的意义依赖周围像素的上下文，因此仅依赖原始像素的线性组合难以刻画图像中的复杂结构。

为了表示更一般的函数关系，可以在网络中加入隐藏层以克服线性模型的限制。**多层感知机**（multilayer perceptron，MLP）将多个全连接层堆叠起来，前面的层学习特征表示，最后一层在该表示上进行线性预测。

假设矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 表示 $n$ 个样本的小批量，且每个样本具有 $d$ 维度原始输入特征，对于具有 $h$ 个隐藏单元的单隐藏层多层感知机，将隐藏表示（hidden representations）或隐藏层变量（hidden variable）定义为 $\mathbf{H} \in \mathbb{R}^{n \times h}$ 为隐藏层的输出。隐藏层和输出层采用全连接，因此有隐藏层权重 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ 、隐藏层偏置 $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$、输出层权重 $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ 和输出层偏置 $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$，由此定义输出 $\mathbf{O} \in \mathbb{R}^{n \times q}$ 为：
$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$
然而，上述模型不足以表述非线性关系，因为只需合并隐藏层，便可将模型退化为一个具有新的权重和偏置的单层线性模型。为了发挥多层架构的潜力，需要在隐藏层仿射变换之后对每个隐藏单元应用非线性的**激活函数**（activation function），使用激活函数后的单隐藏层感知机定义为：
$$
\begin{aligned}
    \mathbf{H} & = \sigma (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$
**多层感知机是通用近似器**：通过隐藏单元和输入之间的相互作用，多层感知机具有近似拟合连续函数的能力；即使是网络只有一个隐藏层，给定足够的隐藏单元和正确的权重， 仍然能对函数建模(尽管实际很困难)，而使用更深的网络，可以更容易地逼近。

### 激活函数
#### ReLU 激活函数
给定元素 $x$，ReLU 被定义为该元素与 0 的最大值： 
$$
\operatorname{ReLU}(x) = \max(x,0).
$$
ReLU 仅保留正元素并丢弃负元素，激活函数是分段线性的；当输入为负，其导数为 0，当输入为正，其导数为 1，当输入为 0，ReLU 不可导（可默认使用左侧的导数）。因此，ReLU 的求导表现是，要么让参数通过，要么让参数消失（该单元的梯度被截断）。

```tikz
\begin{document}
\begin{tikzpicture}[domain=-4:4]
  \draw[very thin,color=gray!40] (-4.1,-0.5) grid (4.1,4.1);
  \draw[->] (-4.2,0) -- (4.3,0) node[right] {$x$};
  \draw[->] (0,-0.7) -- (0,4.3) node[above] {$f(x)$};

  % ReLU
  \draw[color=red, thick, domain=-4:0] plot (\x,{0});
  \draw[color=red, thick, domain=0:4] plot (\x,{\x})
    node[right] {$\mathrm{ReLU}(x)=\max(0,x)$};

  % derivative of ReLU
  \draw[color=blue, thick, domain=-4:0] plot (\x,{0});
  \draw[color=blue, thick, domain=0:4] plot (\x,{1})
    node[right] {$\mathrm{ReLU}'(x)$};

  % note at x=0
  \draw[dashed,color=blue] (0,0) -- (0,1);
  \node[below right] at (0,0) {$0$};
\end{tikzpicture}
\end{document}


```

训练神经网络依赖反向传播（参数更新使用梯度：损失函数对于参数的导数）。反向传播本质上是在多层之间不断相乘各层导数（依据梯度计算的链式法则）。若激活函数导数长期小于 1，深层网络中的梯度会逐层衰减，导致前面层更新缓慢，即**梯度消失**。ReLU 在正半轴导数为 1，能让激活神经元的梯度更直接地传播，从而缓解该问题；但若神经元长期处于负区间，其梯度为 0，可能一直得不到更新。

ReLU 函数的变体：例如，参数化 ReLU 添加了一个线性项，因此即使参数是负的，某些信息仍然可以通过：

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

对于一个只由仿射变换 + ReLU / pReLU 组成的 MLP，整体函数是一个连续的分段线性函数。这是因为，（1）仿射变换是连续分段线性，（2）ReLU / pReLU 也是连续分段线性的，（3）连续分段线性函数的复合仍是连续分段线性的。换言之，在、固定激活区域内，整个网络退化为局部仿射函数；不同区域对应不同的激活模式，因此有不同的仿射表达式。


#### sigmoid 激活函数
对于一个定义域在 $\mathbb{R}$ 中的输入，sigmoid 函数将输入变换为区间(0, 1)上的输出。sigmoid 源于对生物神经元“激发”或“不激发”进行建模，它是一个平滑的、可微的阈值单元近似。
$$
\begin{align} 
&\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.\\

&\frac{d}{dx} \operatorname{sigmoid}(x)  \\
&= \frac{\exp(-x)}{(1 + \exp(-x))^2}  \\
&= \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).
\end{align}$$

sigmoid 导数如公式和图像所示。当输入为 0，sigmoid函数的导数达到最大值 0.25；当输入在任一方向上越远离 0 点时，导数越接近 0。

```tikz
\begin{document}
\begin{tikzpicture}[domain=-6:6, xscale=0.75, yscale=3.0]
  \draw[very thin,color=gray!45] (-6.1,-0.1) grid (6.1,1.15);
  \draw[->] (-6.3,0) -- (6.4,0) node[right] {$x$};
  \draw[->] (0,-0.15) -- (0,1.25) node[above] {$f(x)$};

  % reference lines
  \draw[dashed,color=gray] (-6,1) -- (6,1);
  \draw[dashed,color=gray] (-6,0.5) -- (6,0.5);
  \draw[dashed,color=gray] (-6,0.25) -- (6,0.25);

  % sigmoid
  \draw[color=red, thick, domain=-6:6, samples=100]
    plot (\x,{1/(1+exp(0-\x))});

  % derivative of sigmoid
  \draw[color=blue, thick, domain=-6:6, samples=100]
    plot (\x,{(1/(1+exp(0-\x)))*(1-(1/(1+exp(0-\x))))});

  % labels
  \node[color=red,right] at (3.4,0.88) {$\mathrm{sigmoid}(x)$};
  \node[color=blue,right] at (1.0,0.25) {$\mathrm{sigmoid}'(x)$};

  % key values
  \node[left] at (0,1) {$1$};
  \node[left] at (0,0.5) {$0.5$};
  \node[left] at (0,0.25) {$0.25$};
\end{tikzpicture}
\end{document}


```

#### tanh 激活函数
tanh(双曲正切)函数将输入变换为区间(-1, 1)上的输出，函数的形状类似于sigmoid函数，但 tanh 函数关于坐标系原点中心对称；当输入在 0 附近时，tanh 函数接近线性变换。
$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$
$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$
tanh 导数如公式和图像所示。当输入为 0，tanh 函数的导数达到最大值 1；当输入在任一方向上越远离 0 点时，导数越接近 0。

```tikz
\begin{document}
\begin{tikzpicture}[domain=-4:4, xscale=0.9, yscale=2.0]
  \draw[very thin,color=gray!45] (-4.1,-1.15) grid (4.1,1.15);
  \draw[->] (-4.3,0) -- (4.4,0) node[right] {$x$};
  \draw[->] (0,-1.25) -- (0,1.3) node[above] {$f(x)$};

  \draw[dashed,color=gray] (-4,1) -- (4,1);
  \draw[dashed,color=gray] (-4,-1) -- (4,-1);

  % tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
  \draw[color=red, thick, domain=-4:4, samples=100]
    plot (\x,{(1-exp(0-2*\x))/(1+exp(0-2*\x))});

  % tanh'(x) = 1 - tanh(x)^2
  \draw[color=blue, thick, domain=-4:4, samples=100]
    plot (\x,{1-((1-exp(0-2*\x))/(1+exp(0-2*\x)))*((1-exp(0-2*\x))/(1+exp(0-2*\x)))});

  \node[color=red,right] at (2.3,0.95) {$\mathrm{tanh}(x)$};
  \node[color=blue,right] at (1.1,0.45) {$\mathrm{tanh}'(x)$};

  \node[left] at (0,1) {$1$};
  \node[left] at (0,-1) {$-1$};
  \node[below right] at (0,0) {$0$};
\end{tikzpicture}
\end{document}

```

tanh 函数和 sigmoid 存在关联 $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x).$
$$
\operatorname{tanh}(x)+1
=
\frac{e^{2x}-1}{e^{2x}+1}+1
=
\frac{2e^{2x}}{e^{2x}+1}

$$

$$
\operatorname{sigmoid}(2x)
=
\frac{1}{1+e^{-2x}}
=
\frac{e^{2x}}{e^{2x}+1}

$$
### 多层感知机超参数搜索与联合优化
多层感知机的超参数有：学习率、权重衰减、隐藏层数、每层的隐藏单元数等，构建多个超参数的搜索与联合优化策略具有挑战性。

这是因为超参数之间并不是相互独立的。例如，学习率影响参数更新幅度；批量大小影响梯度估计的噪声；隐藏层数和隐藏单元数决定模型容量；训练轮数过少可能欠拟合，训练轮数过多又可能过拟合。改变其中一个超参数，往往会改变其他超参数的合适取值。

若使用网格搜索，假设每个超参数只尝试 5 个候选值，若有 4 个超参数，则组合数为：
$$
5^4 = 625.
$$
随着超参数数量增加，搜索空间会快速膨胀。因此，简单枚举所有组合通常代价很高。

实际中更常用的策略是**随机搜索**。相比固定网格，随机搜索可以在连续空间中探索更多不同取值，尤其适合学习率、权重衰减等跨数量级变化的超参数。例如：
$$
\eta \sim \operatorname{loguniform}(10^{-4}, 10^{-1}),
$$

$$
\lambda \sim \operatorname{loguniform}(10^{-6}, 10^{-2}).
$$
其中 $\eta$ 表示学习率，$\lambda$ 表示权重衰减系数。

为了降低搜索成本，可以使用“先粗筛，再精训”的思想：
1. 随机采样多组超参数
2. 每组只训练少量轮数，快速淘汰表现较差的配置
3. 保留验证集表现较好的配置，分配更多训练轮数
4. 对最终候选配置进行完整训练

这种思想对应于 Successive Halving、Hyperband、ASHA 等多保真超参数搜索方法。其核心是：不要让明显较差的配置消耗完整训练资源，而是将更多计算分配给更有希望的配置。

对于多层感知机，可以采用如下搜索顺序：
- 先搜索学习率和权重衰减，因为它们直接影响优化过程
- 再搜索隐藏层数和隐藏单元数，因为它们决定模型容量
- 若出现过拟合，再考虑 dropout、权重衰减和早停
- 最后固定较优配置，在验证集上选择表现最好的模型

一个简单的搜索空间可以写成：
$$
\eta \in [10^{-4}, 10^{-1}],
$$

$$
\lambda \in [10^{-6}, 10^{-2}],
$$

$$
h \in \{128, 256, 512, 1024\},
$$

$$
L \in \{1,2,3\}.
$$
其中 $h$ 表示每层隐藏单元数，$L$ 表示隐藏层数。

> 总结：多个超参数搜索的难点在于组合空间大、单次训练代价高、超参数之间存在相互作用。实践中通常先用随机搜索和少量训练轮数进行粗筛，再对较优配置增加训练资源；对于更大规模任务，可以使用 Hyperband、贝叶斯优化或基于早停的自动搜索方法。

## 模型复杂性
### 术语定义
过拟合（overfitting）：模型在训练数据上拟合的比在潜在分布中更接近
正则化（regularization）：对抗过拟合的技术
训练误差（training error）：模型在训练数据集上计算得到的误差
泛化误差（generalization error）：模型应用在从原始样本的分布中抽取的无限多数据样本时，模型误差的期望
独立同分布假设（i.i.d. assumption）：假设训练数据和测试数据都是从相同的分布中独立提取的；抽取的样本之间没有相关性
早停（early stopping）：较少训练迭代周期，避免模型过度训练
$K$ 折交叉验证：原始训练数据被分成 $K$ 个不重叠的子集，每次在  $K-1$ 个子集上进行训练，并在剩余的子集上验证，用于在数据稀缺时构成一个合适的验证集用于进行模型选择


### 模型复杂性和泛化
模型复杂性与样本数量的关系：通常情况下，当有更**复杂**的模型和更少的样本时，训练误差会下降，但泛化误差会增大

一些直观的影响模型泛化的因素有：（1）可调整参数的数量（自由度），当自由度很大时，模型容易过拟合；（2）参数采用的值，当权重的取值范围大时，模型可能容易过拟合；（3）训练样本的数量，当模型表达能力强而样本数量少时，模型容易过拟合

```tikz
\begin{document}
\begin{tikzpicture}[domain=0:9, xscale=1.0, yscale=1.0]
  % axes
  \draw[->, thick] (0,0) -- (9.2,0) node[below left] {model-complex};
  \draw[->, thick] (0,0) -- (0,5.2) node[left] {loss};

  % curves
  \draw[color=black, thick, domain=0.7:8.7, samples=120]
    plot (\x,{4.4*exp(0-0.42*\x)+0.25});

  \draw[color=black, thick, domain=0.7:8.7, samples=120]
    plot (\x,{1.85+0.055*(\x-4.4)*(\x-4.4)+1.5*exp(0-0.9*\x)});

  % best vertical line
  \draw[dashed, thick] (4.3,0) -- (4.3,4.9);
  \node[above] at (4.3,4.95) {best};

  % top arrows and labels
  \draw[->, thick] (2.3,4.85) -- (1.0,4.85);
  \node at (1.6,4.35) {underfitting};

  \draw[->, thick] (5.8,4.85) -- (7.1,4.85);
  \node at (6.5,4.35) {overfitting};

  % curve labels
  \node[right] at (7.0,2.6) {generalization-error};
  \node[right] at (7.0,1.0) {training-error};
\end{tikzpicture}
\end{document}

```


#### 一个多项式例子用于说明模型复杂性
目标：给定由单个特征 $\mathbf{x}$ 和对应实数标签 $y$ 组成的训练数据，试图找到一个下面的 $d$ 阶多项式来估计标签 $y$. 其中，特征由 $\mathbf{x}$ 的幂给出，使用 $\frac{x^i}{i!}$ 可以避免很大的特别大的指数值。
$$\hat{y}= \sum_{i=0}^d \frac{x^i}{i!} w_i$$

从特征空间的角度看，多项式回归并不是直接在原始的一维输入 $x$ 上做线性拟合，而是先把 $x$ 映射到一个由非线性相关维度组成的新特征空间：
$$
\phi(x)=\left(1, x, \frac{x^2}{2!}, \dots, \frac{x^d}{d!}\right).
$$
于是模型可以写成 $\hat{y}=\mathbf{w}^{\top}\phi(x)$。它对原始输入 $x$ 是非线性的，但对参数 $\mathbf{w}$ 仍然是线性的。多项式阶数越高，特征空间维度越高，模型可使用的方向越多，能表达的函数集合也越大。因此在固定训练数据上，高阶模型的最小训练误差不会高于低阶模型；更准确地说，若有 $n$ 个不同的输入点，通常存在一个最高 $n-1$ 阶的多项式穿过所有训练样本。

这个例子说明：模型太灵活，而数据太少时，模型会把噪声也当成规律学进去。若真实关系并不需要高阶特征，那么高阶项本应学到接近 0 的权重；但训练数据有限且标签含有随机噪声时，模型无法可靠判断某些波动是真实规律还是偶然噪声，新增的高阶特征维度就可能被用来贴合训练样本中的偶然变化。因此，训练损失可以有效降低，但测试损失仍然很高，最终造成过拟合。

## 权重衰减
权重衰减也被称为 $L_{2}$ 正则化，通过度量（？如何精确地进行度量）函数与零的距离来衡量函数的复杂度。

一种度量方案是，通过对权重向量的范数来度量其复杂性，例如 $\|\mathbf{w}^2\|$，要使得复杂性小，常用方法是将其范数作为惩罚项加到最小化损失中（即：训练目标为最小化预测损失和惩罚项之和）。考虑损失函数 $L$，其中， $\mathbf{x}^{(i)}$ 是样本 $i$ 的特征，$\mathbf{y}^{(i)}$ 是样本 $i$ 的标签，引入 $L_2$ 惩罚项和正则化常数超参数 $\lambda$：
$$
L(\mathbf{w},b) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2} 
\left(
\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}
\right)
$$

$$
L(\mathbf{w},b)+ \frac{\lambda}{2}\|\mathbf{w}\|^{2}.
$$
对于$\lambda > 0$，训练目标将限制 $\| \mathbf{w} \|$ 的大小。使用平方范数（而非欧几里得距离）是为了便于计算：惩罚项各分量导数的和等于和的导数；使用  $L_{2}$  范数（岭回归）而不是 $L_{1}$ 范数（套索回归），是因为它对权重向量的大分量施加了巨大的惩罚，使算法偏向于在大量特征上均匀分布权重的模型； $L_{1}$ 相比之下容易使权重集中于一小部分特征上（特征选择）。

$L_{2}$ 正则化回归的小批量梯度下降更新如下：
$$
\begin{aligned}
\mathbf{w} & \leftarrow 
\left( 1 - \eta \lambda \right) \mathbf{w} 
- \frac{\eta}{|\mathcal{B}|} \sum_{{i} \in \mathcal{B}} 
\mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} +b - y^{(i)} \right).

\end{aligned}

$$
是否对偏置 $b^2$ 进行惩罚在不同场景下有所区别，一般地，不会在网络输出层的偏置项进行正则化。

### L1 与 L2 正则化的效果区别
岭回归通常使用平方 $L_2$ 范数，套索回归使用 $L_1$ 范数，对应的正则化项分别为：
$$
R_{L_2}(\mathbf{w})
=
\frac{\lambda}{2}\sum_j w_j^2,
\qquad
R_{L_1}(\mathbf{w})
=
\lambda\sum_j |w_j|.
$$
当 $w_j\neq0$ 时，两种惩罚项关于权重的导数为：
$$
\frac{\partial R_{L_2}}{\partial w_j}=\lambda w_j,
\qquad
\frac{\partial R_{L_1}}{\partial w_j}=\lambda\operatorname{sign}(w_j).
$$
$L_2$ 的惩罚力度随 $|w_j|$ 增大，因此会更强地压制大权重；但当权重接近零时，惩罚梯度也接近零，所以通常只会将权重缩小，而不会使其精确等于零。$L_1$ 的惩罚梯度大小始终为 $\lambda$，即使权重已经很小，仍然受到固定大小的收缩，因此更容易被推到零。

$L_1$ 在 $w_j=0$ 处不可导，其次梯度为：
$$
\partial |w_j|=[-1,1].
$$
因此，只要损失函数在该位置的梯度满足：
$$
\left|\frac{\partial L}{\partial w_j}\right|\leq\lambda,
$$
就可以令 $w_j=0$ 并满足最优条件。相比之下，$L_2$ 要在 $w_j=0$ 处满足最优条件，通常要求 $\frac{\partial L}{\partial w_j}=0$，这一条件更加严格。因此，$L_1$ 容易得到包含大量零权重的稀疏模型，可用于特征选择；$L_2$ 通常得到许多较小但非零的权重。

还可以从相关特征之间的权重分配理解这种区别。假设两个作用相近的特征只需要满足 $w_1+w_2=c$，且 $w_1,w_2\geq0$。对于 $L_2$：
$$
c^2+0^2=c^2,
\qquad
\left(\frac{c}{2}\right)^2+\left(\frac{c}{2}\right)^2=\frac{c^2}{2}.
$$
均匀分配权重具有更小的平方和，因此 $L_2$ 明确偏向于让相关特征共同承担权重。对于 $L_1$：
$$
|w_1|+|w_2|=c,
$$
不同的非负分配方式具有相同惩罚，$L_1$ 不会因为均匀分配而降低代价；结合其在零点处的尖点，最优解更容易落在坐标轴上，只保留少数特征。

> 总结：$L_2$ 的梯度随权重大小变化，因此强烈压制大权重，却很少产生精确的零，并且倾向于在相关特征之间分散权重；$L_1$ 在零点附近仍保持固定的收缩力度，而且零点具有一个次梯度区间，因此可以把弱特征的权重直接压到零，形成稀疏解。

### 从向量的 L2 范数推广到矩阵的 Frobenius 范数

对于列向量 $\mathbf{w}\in\mathbb{R}^d$，$L_2$ 范数平方可以写成向量内积：
$$
\|\mathbf{w}\|_2^2
=
\sum_{i=1}^d w_i^2
=
\mathbf{w}^\top\mathbf{w}.
$$
由于 $\mathbf{w}^\top\mathbf{w}$ 的形状为 $1\times1$，所以结果是一个标量。对于矩阵 $\mathbf{W}\in\mathbb{R}^{m\times n}$，Frobenius 范数定义为矩阵所有元素平方和的平方根，即将矩阵展平为向量后应用 $L_2$ 范数；Frobenius 范数的平方就是矩阵所有元素的平方和：
$$
\|\mathbf{W}\|_F
=
\sqrt{\sum_{i=1}^m\sum_{j=1}^n W_{ij}^2}.
$$

$$
\|\mathbf{W}\|_F^2
=
\sum_{i=1}^m\sum_{j=1}^n W_{ij}^2.
$$
考察矩阵乘积 $\mathbf{W}^\top\mathbf{W}$，形状为 $n\times n$，第 $j$ 个对角元素恰好是 $\mathbf{W}$ 第 $j$ 列元素的平方和；对该矩阵取迹，就能进一步将所有列的平方和汇总为一个标量：
$$
\begin{aligned}
\operatorname{tr}(\mathbf{W}^\top\mathbf{W})
&=
\sum_{j=1}^n(\mathbf{W}^\top\mathbf{W})_{jj}\\
&=
\sum_{j=1}^n\sum_{i=1}^m W_{ij}^2\\
&=
\|\mathbf{W}\|_F^2.
\end{aligned}
$$
由此可见，Frobenius 范数的平方可以通过矩阵乘法和迹运算计算，若权重是矩阵，矩阵形式的 $L_2$ 正则项可以写成：
$$
\frac{\lambda}{2}\|\mathbf{W}\|_F^2
=
\frac{\lambda}{2}\operatorname{tr}(\mathbf{W}^\top\mathbf{W}).
$$
该正则项关于 $\mathbf{W}$ 的梯度为：
$$
\nabla_{\mathbf{W}}
\frac{1}{2}\|\mathbf{W}\|_F^2
=
\mathbf{W}.
$$
因此，矩阵权重同样具有权重衰减更新：
$$
\begin{aligned}
\mathbf W
&\leftarrow
\mathbf W-\eta\nabla_{\mathbf W}J\\
&=
\mathbf W-\eta
\left(
\nabla_{\mathbf W}L+\lambda\mathbf W
\right)\\
&=
(1-\eta\lambda)\mathbf W
-
\eta\nabla_{\mathbf W}L.
\end{aligned}
$$
### 正则化的贝叶斯解释
设训练数据为 $\mathcal{D}=(\mathbf{X},\mathbf{y})$，模型参数为 $\mathbf{w}$。从概率角度看，似然 $P(\mathcal{D}\mid\mathbf{w})$ 表示：给定参数 $\mathbf{w}$ 后，观察到当前训练数据的可能性。

#### 最大似然估计
最大似然估计（maximum likelihood estimation，MLE）选择最能解释训练数据的参数：
$$
\mathbf{w}_{\mathrm{MLE}}
=
\operatorname*{argmax}_{\mathbf{w}}
P(\mathcal{D}\mid\mathbf{w}).
$$
为了便于计算，通常将最大化似然转化为最小化负对数似然：
$$
\mathbf{w}_{\mathrm{MLE}}
=
\operatorname*{argmin}_{\mathbf{w}}
\left[-\log P(\mathcal{D}\mid\mathbf{w})\right].
$$
因此，许多常见损失函数都可以理解为负对数似然。例如，在线性回归中假设标签噪声服从正态分布，最小化负对数似然就等价于最小化平方误差。

最大似然估计只考虑参数对训练数据的拟合程度。当数据量较少或模型表达能力很强时，某些复杂参数可能很好地解释训练数据中的偶然噪声，从而造成过拟合。

#### 最大后验估计
贝叶斯方法在观察数据之前，先用参数的先验分布 $P(\mathbf{w})$ 表示对参数取值的已有假设。观察训练数据后，根据贝叶斯公式得到后验分布：
$$
P(\mathbf{w}\mid\mathcal{D})
=
\frac{P(\mathcal{D}\mid\mathbf{w})P(\mathbf{w})}{P(\mathcal{D})}
\propto
P(\mathcal{D}\mid\mathbf{w})P(\mathbf{w}).
$$
最大后验估计（maximum a posteriori estimation，MAP）选择后验概率最大的参数：
$$
\begin{aligned}
\mathbf{w}_{\mathrm{MAP}}
&=
\operatorname*{argmax}_{\mathbf{w}}
P(\mathbf{w}\mid\mathcal{D})\\
&=
\operatorname*{argmin}_{\mathbf{w}}
\left[
-\log P(\mathcal{D}\mid\mathbf{w})
-\log P(\mathbf{w})
\right].
\end{aligned}
$$
其中，$P(\mathcal{D})$ 不依赖于 $\mathbf{w}$，所以不会影响最优参数。上式正好具有“预测损失 + 正则化项”的形式：
$$
\underbrace{-\log P(\mathcal{D}\mid\mathbf{w})}_{\text{预测损失}}
+
\underbrace{-\log P(\mathbf{w})}_{\text{正则化项}}.
$$
这说明正则化项可以解释为参数先验的负对数。正则化不是只要求模型拟合训练数据，还要求参数符合预先设定合理取值范围。

#### $L_2$ 正则化与高斯先验
若假设权重服从零均值、各向同性的高斯先验：
$$
P(\mathbf{w})
\propto
\exp\left(-\frac{\lambda}{2}\|\mathbf{w}\|_2^2\right),
$$
则其负对数为：
$$
-\log P(\mathbf{w})
=
\frac{\lambda}{2}\|\mathbf{w}\|_2^2+C.
$$
去掉与参数无关的常数 $C$，就得到 $L_2$ 正则项。因此，$L_2$ 正则化相当于加入“权重通常应当分布在零附近”的先验假设。$\lambda$ 越大，高斯先验越集中于零附近，对大权重的限制越强。

#### $L_1$ 正则化与拉普拉斯先验
若假设各个权重相互独立并服从零均值拉普拉斯先验：
$$
P(\mathbf{w})
\propto
\exp\left(-\lambda\|\mathbf{w}\|_1\right),
$$
则：
$$
-\log P(\mathbf{w})
=
\lambda\|\mathbf{w}\|_1+C.
$$
这对应 $L_1$ 正则化。拉普拉斯分布在零点处具有尖峰，因此其最大后验解更容易将较弱的权重压到零，产生稀疏模型。

需要注意，先验分布参数与正则化系数之间的精确关系还取决于损失采用求和还是平均，以及似然中的噪声方差。例如在线性回归中，若噪声方差为 $\sigma^2$、高斯先验方差为 $\tau^2$，将目标函数缩放为普通平方误差后，正则化强度满足 $\lambda=\sigma^2/\tau^2$。

> 总结：最大似然估计只选择最能解释训练数据的参数；最大后验估计还会考虑参数先验。取负对数后，似然变成预测损失，先验变成正则化项。因此，无正则化训练通常对应最大似然估计，带正则化训练可以解释为最大后验估计；$L_2$ 正则化对应高斯先验，$L_1$ 正则化对应拉普拉斯先验。高斯先验认为权重应当“普遍较小”，但不一定为零；拉普拉斯先验认为许多权重应当恰好为零，少数权重可以相对较大。

## 暂退法
《 Dropout: a simple way to prevent neural networks from overfitting》 暂退法的想法被提出：在训练过程中，在计算后续层之前像网络中的每一层注入噪声；其理论角度是函数的平滑性：函数不应该对其输入的微小变化敏感；当训练一个多层网络时，注入噪声能在输入/输出映射上增强网络拟合函数平滑性。

论文方法通过无偏向方式注入噪声：将均值为零分布 $\epsilon  \sim \mathcal{N} (0, \sigma^2)$ 的高斯噪声添加到线性模型的输入 $\mathbf{x}$ 中，产生扰动 $\mathbf{x}' = \mathbf{x} + \epsilon$，预期是 $E[\mathbf{x}'] = \mathbf{x}$. 

在标准暂退法中，随机丢弃一部分隐藏单元（制造噪声），同时放大未被丢弃的单元，使激活值（激活函数之后、暂退法之前的隐藏层中间变量）期望保持不变。设丢弃概率为 $p$，暂退后的激活值 $h'$ 可以写为： 
$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 概率为 } p \\
    \frac{h}{1-p} & \text{ 其他情况}
\end{cases}
\end{aligned}
$$
用 $h'$ 以暂退概率替换 $h$，其期望值保持不变 $E[h'] = h$. 这里的“消除每一层的偏差”指的是放大未被丢弃的单元以消除暂退造成的期望缩小。


### 实现中使用的暂退法
仅在训练期间使用暂退法，将暂退法应用到隐藏层，以 $p$ 的概率将隐藏单元置为零；这会使得神经网络的输出计算不再依赖被删除的隐藏层单元，并且它们各自的梯度在执行方向传播时也会消失。

### 暂退概率与层的位置
交换两个隐藏层的暂退概率并不等价。记 $D_p$ 为暂退概率为 $p$ 的暂退操作，则两层网络中一般有：
$$
\begin{aligned}
 D_{p_2}\left(\operatorname{ReLU}\left(\mathbf{W}_2D_{p_1}(\mathbf{h}_1)+\mathbf{b}_2\right)\right)
\neq \\
 D_{p_1}\left(\operatorname{ReLU}\left(\mathbf{W}_2D_{p_2}(\mathbf{h}_1)+\mathbf{b}_2\right)\right).
 \end{aligned}
$$
假设原暂退概率为 $p_1=0.2$、$p_2=0.5$，第一层保留约 $80\%$ 的激活值，第二层保留约 $50\%$ 的激活值。第一层学习到的基础特征因此较稳定，而第二层受到较强的随机扰动，不能过度依赖某些高层特征。交换后，第一层只保留约 $50\%$ 的激活值，较强的噪声会经过权重变换和激活函数继续传播；第二层只丢弃约 $20\%$ 的激活值，对高层特征的约束减弱。这样可能使基础表示更不稳定、训练更困难，同时保留更多高层特征之间的依赖关系。

### 暂退法与激活值方差
暂退法的随机性来自一个**暂退掩码**。对固定的激活值 $h$，令：
$$
m=
\begin{cases}
0, & \text{概率为 }p,\\
1, & \text{概率为 }1-p,
\end{cases}
$$
则暂退后的激活值为：
$$
h'=\frac{m}{1-p}h.
$$
这里的 $m$ 决定当前前向传播是否保留该单元。由于 $E[m]=1-p$，有：
$$
E[h'\mid h]=h.
$$
也就是说，暂退法不会改变固定激活值的平均大小，但会使它在不同掩码下随机变化。固定 $h$、只改变掩码时，这种随机变化的方差为：
$$
\operatorname{Var}(h'\mid h)
=
\frac{p}{1-p}h^2.
$$
当不使用暂退法时 $p=0$，掩码恒为 $1$，固定 $h$ 时不会产生这部分方差。

实际训练中，不同样本的 $h$ 本身也不同，网络参数还会随训练更新。若暂退掩码与 $h$ 独立，则一批样本中观测到的总方差由两部分组成：原始激活值在样本之间的差异，以及暂退掩码带来的额外差异。由全方差公式：
$$
\operatorname{Var}(h')
=
\operatorname{Var}(h)
+
\frac{p}{1-p}E[h^2].
$$
在每个训练时刻 $t$，对第 $\ell$ 个隐藏层当前输出的 $N$ 个观测值计算**经验方差**：
$$
s_\ell^2(t)
=
\frac{1}{N}\sum_{i=1}^{N}
\left(h_{\ell,i}(t)-\bar{h}_\ell(t)\right)^2.
$$
不使用暂退法时，曲线主要反映样本差异和参数更新；使用暂退法时，曲线还包含掩码噪声。

较大的暂退概率使额外方差增大，模型更难依赖某个固定神经元或少数特征的组合。不同神经元需要在彼此缺失时仍能完成任务，这种相互替代的约束可以减弱**共适应**，即神经元过度依赖某些固定搭配，从而减轻过拟合。若暂退概率过大，噪声也会破坏有效信号，使优化变慢并导致欠拟合。


### 暂退法与权重衰退同时使用

暂退法和权重衰退都能限制模型复杂度，但作用对象不同：暂退法随机删除激活值，限制模型对固定特征组合的依赖；权重衰退直接惩罚较大的权重，使模型倾向于使用更平滑的参数。前者改变训练时的网络结构，后者改变参数更新，因此二者在目标上有重叠，在机制上互补。

线性模型对输入使用暂退法时，期望平方损失可以写成：
$$
\begin{align}


\mathbb{E}_{\mathbf{m}}
\left[
\frac{1}{2}
\left(y-\mathbf{w}^{\top}
\frac{\mathbf{m}\odot\mathbf{x}}{1-p}
\right)^2
\right]
= \\

\frac{1}{2}(y-\mathbf{w}^{\top}\mathbf{x})^2
+
\frac{p}{2(1-p)}
\sum_jx_j^2w_j^2.

\end{align}
$$
第二项具有 $L_2$ 正则化的形式，说明暂退法在该情形下会产生一种与输入特征尺度有关的权重惩罚。但在包含带有 ReLU 等非线性激活函数的深层网络中，暂退法通常不等价于固定的 $L_2$ 正则项。

因此，二者在作用机制上是互补的：暂退法要求表示在部分神经元缺失时仍然有效，权重衰退则要求这种表示不依赖过大的参数。暂退法产生的随机子网络可能促使模型增大某些权重来补偿被丢弃的激活值，权重衰退可以抑制这种补偿；反过来，仅使用权重衰退并不能阻止模型始终依赖同一组神经元。

### 把暂退法应用到权重矩阵而不是激活值

如果随机置零的对象从激活值改为权重矩阵中的元素，这种方法通常称为 **DropConnect**。设权重矩阵为 $\mathbf{W}$，引入与其同形状的伯努利掩码 $\mathbf{M}$：
$$
M_{ij}=
\begin{cases}
0, & \text{概率为 }p,\\
1, & \text{概率为 }1-p.
\end{cases}
$$
训练时使用随机权重：
$$
\mathbf{W}'=\frac{\mathbf{M}\odot\mathbf{W}}{1-p}.
$$
由于 $E[\mathbf{M}]=1-p$，有：
$$
E[\mathbf{W}'\mid\mathbf{W}]=\mathbf{W}.
$$
因此，对固定输入 $\mathbf{x}$，线性层的预激活值满足：
$$
E[\mathbf{x}\mathbf{W}'+\mathbf{b}\mid\mathbf{x},\mathbf{W}]
=
\mathbf{x}\mathbf{W}+\mathbf{b}.
$$
预测阶段不再随机屏蔽权重，直接使用原始权重 $\mathbf{W}$。

与普通暂退法相比，普通暂退法随机删除隐藏单元的输出，相当于同时删除该单元连接到下一层的所有边；DropConnect 则逐条随机删除权重，同一隐藏单元仍可能通过其他连接传递信息。

因此，普通暂退法产生由神经元子集组成的随机网络，主要限制模型对完整特征的依赖；DropConnect 产生由连接子集组成的随机网络，主要限制模型对特定连接的依赖。后者的扰动更细粒度，但需要生成和处理与权重矩阵同形状的掩码，计算和存储开销通常更大。

对中间变量（第 $k$ 个神经元的预激活值） $z_k=\sum_jx_jW_{jk}$，DropConnect 的权重掩码带来的条件方差为：
$$
\operatorname{Var}(z'_k\mid\mathbf{x},\mathbf{W})
=
\frac{p}{1-p}
\sum_jx_j^2W_{jk}^2.
$$
所以 DropConnect 同样能注入无偏噪声、限制连接之间的过度依赖。与普通暂退法一样，置零只发生在训练期间，预测时使用完整的权重矩阵，因此它主要是一种正则化方法，而不是直接得到永久稀疏模型的剪枝方法。


## 前向传播与反向传播
以带权重衰减的单隐藏层多层感知机为例，说明反向传播的细节。梯度自动计算，即自动微分，简化了深度学习算法实现。

### 前向传播
**前向传播**：按顺序从输入层到输出层，计算和存储神经网络中每层的结果。假设输入样本为 $\mathbf{x} \in \mathbb{R}^d$，且隐藏层不包含偏置项；设 $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ 是隐藏层的权重参数，将中间变量 $\mathbf{z} \in \mathbb{R}^h$ 提供激活函数 $\phi$ 后，得到长度为 $h$ 的隐藏激活向量 $\mathbf{h} \in \mathbb{R}^h$；设输出层的权重参数为 $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$，得到输出层变量 $\mathbf{o} \in \mathbb{R}^q$.
$$
\begin{align}
\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x} \\ 
\mathbf{h} = \phi(\mathbf{z}) \\
\mathbf{o} = \mathbf{W}^{(2)}\mathbf{h}.
\end{align}
$$
假设 $l$ 为损失函数， $y$ 为样本标签，计算单个数据样本损失项:
$$
L = l(\mathbf{o},y)
$$
权重矩阵使用 Frobenius 范数，给定超参数 $\lambda$ 正则化项（$L2$）为：
$$
s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_{F}^2 + \|\mathbf{W}^{(2)}\|_{F}^2)
$$
称模型在给定数据样本上的正则化损失为 $J$ 目标函数：
$$
J=L+s.
$$
### 反向传播
**反向传播**：计算神经网络参数梯度的方法。该方法根据微积分链式规则，按照相反的顺序从输出层到输入层遍历网络，在此过程中存储计算参数梯度所需的中间变量（偏导数）。

使用 $prod$ 运算符在执行必要操作（换位和交换输入位置）后将其参数相乘，对于向量即矩阵乘法。假设有函数 $Y = f(x)$ 和函数 $Z = g(Y)$，函数的输入和输出可以任意形状的张量。计算 $Z$ 关于 $X$ 的导数：
$$
\frac{ \partial \mathsf{Z} }{\partial \mathsf{X}}
= prod(
\frac{ \partial \mathsf{Z} }{\partial \mathsf{Y}},
\frac{ \partial \mathsf{Y} }{\partial \mathsf{X}}
) .
$$
考虑单隐藏层的网络参数 $\mathbf{W}^{(1)}$ 和 $\mathbf{W}^{(2)}$，反向传播需要计算梯度 $\frac{ \partial J}{\partial \mathbf{W}^{(1)}}$ 和  $\frac{ \partial J}{\partial \mathbf{W}^{(2)}}$，为此应用链式法则，依次计算每个中间变量和参数的梯度，计算顺序与前向传播执行顺序相反，首先，计算目标函数 $J=L+s$ 相对于损失项 $L$ 和正则项 $s$ 的梯度：
$$
\frac{\partial J}{\partial L} = 1, 
\frac{\partial J}{\partial s} = 1.
$$
然后，计算目标函数 $J$ 关于输出层变量 $\mathbf{o}$ 的梯度，有：
$$
\begin{align}
\frac{ \partial J}{ \partial \mathbf{o}} &= prod(\frac{ \partial J}{ \partial L}, \frac{ \partial L}{ \partial \mathbf{o}}) \\
&=\frac{ \partial L}{ \partial \mathbf{o}} \in \mathbb{R}^q.
\end{align}
$$
接下来计算损失项 $L$ 关于输出层参数 $\mathbf{W}^{(2)}$ 的梯度：
$$
\frac{\partial L}{\partial \mathbf{W}^{(2)}} 
= prod(\frac{ \partial L}{ \partial \mathbf{o}}, \frac{ \partial \mathbf{o}}{ \partial \mathbf{W}^{(2)}}) 
= \frac{ \partial L}{ \partial \mathbf{o}} \mathbf{h}^ \top
$$
接下来计算正则化项 $s$ 关于两个参数的梯度：
$$
\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}, 
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.
$$
现在即可计算目标函数 $J$ 关于输出层参数 $\mathbf{W}^{(2)}$ 的梯度：
$$
\begin{align}
\frac{ \partial J}{ \partial \mathbf{W}^{(2)}}
&= prod(\frac{ \partial J}{ \partial L}, \frac{ \partial L}{ \partial \mathbf{W}^{(2)}})   
+ prod(\frac{ \partial J}{ \partial s}, \frac{ \partial s}{ \partial \mathbf{W}^{(2)}})\\
&=\frac{ \partial L}{ \partial \mathbf{o}} \mathbf{h}^ \top
+ \lambda \mathbf{W}^{(2)} 
= \frac{ \partial J}{ \partial \mathbf{o}} \mathbf{h}^ \top
+ \lambda \mathbf{W}^{(2)} .
\end{align}
$$
为了继续求取关于隐藏层参数 $\mathbf{W}^{(1)}$ 的梯度，继续沿着输出层到隐藏层反向传播，先求取关于隐藏层输出 $\mathbf{h}$ 的梯度：
$$
\begin{align}
\frac{ \partial J}{ \partial \mathbf{h}}
&= prod(\frac{ \partial J}{ \partial \mathbf{o}}, \frac{ \partial \mathbf{o}}{ \partial \mathbf{h}}) 
={\mathbf{W}^{(2)}}^\top \frac{ \partial J}{ \partial \mathbf{o}} .
\end{align}
$$
正向传播时 $\mathbf{W}^{(1)} \mathbf{x} \to \mathbf{z} \to \mathbf{h}$，其中从预激活的中间变量 $\mathbf{z}$ 到隐藏层变量 $\mathbf{h}$ 的激活函数 $\phi$ 是按元素计算，用 $\odot$ 便是按元素乘法运算符，计算关于中间变量 $\mathbf{z}$ 的梯度：
$$
\frac{ \partial J}{ \partial \mathbf{z}} 
= prod(\frac{ \partial J}{ \partial \mathbf{h}}, \frac{ \partial \mathbf{h}}{ \partial \mathbf{z}}) 
= \frac{ \partial J}{ \partial \mathbf{h}} \odot \phi'(\mathbf{z}) \in \mathbb{R} ^ h .
$$
最后，可以求出目标函数 $J$ 关于隐藏层参数 $\mathbf{W}^{(1)}$ 的梯度  $\frac{ \partial J}{\partial \mathbf{W}^{(1)}} \in \mathbb{R}^{h \times d}$ , 根据链式法则计算出：
$$
\begin{align}
\frac{ \partial J}{\partial \mathbf{W}^{(1)}} 
&= prod(\frac{ \partial J}{ \partial \mathbf{z}}, \frac{ \partial \mathbf{z}}{ \partial \mathbf{W}^{(1)}}) 
+ prod(\frac{ \partial J}{ \partial s}, \frac{ \partial s}{ \partial \mathbf{W}^{(1)}}) \\
&= \frac{ \partial J}{ \partial \mathbf{z}} \mathbf{x} ^ {\top} + \lambda \mathbf{W}^{(1)} .
\end{align}
$$
以上案例展示了单隐藏层多层感知机反向传播的计算过程，反向传播从损失关于输出 $\mathbf{o}$ 的梯度开始，沿前向传播的相反方向应用链式法则。对于输出层，将输出误差与隐藏层变量 $\mathbf{h}$ 结合，得到 $\frac{\partial J}{\partial\mathbf{W}^{(2)}}$；对于隐藏层，先将梯度从 $\mathbf{o}$ 传回 $\mathbf{h}$，再乘以激活函数导数得到 $\frac{\partial J}{\partial\mathbf{z}}$，最后与输入 $\mathbf{x}$ 结合，得到 $\frac{\partial J}{\partial\mathbf{W}^{(1)}}$。优化算法据此更新 $\mathbf{W}^{(1)}$ 和 $\mathbf{W}^{(2)}$，使预测损失下降；其中叠加的 $\lambda\mathbf{W}^{(1)}$ 和 $\lambda\mathbf{W}^{(2)}$ 同时使权重向零收缩。

### 训练神经网络
在训练神经网络时，前向传播和反向传播相互依赖。一方吗，前向传播期间计算正则项取决于模型参数的当前值，它们是由优化算法根据最近迭代的反向传播给出的；另一方面，反向传播期间参数的梯度计算，取决于当前前向传播给出的隐藏层变量 $\mathbf{h}$ 的当前值。

因此，在初始化模型参数后，交替使用前向传播和反向传播，利用反向传播给出的梯度来更新模型参数。反向传播时，重复利用前向传播中存储的中间值，以避免重复计算。换言之，我们需要保存中间值直到反向传播完成，这也是训练比预测需要更多内存的原因之一。

### 矩阵变量的梯度形状
若标量函数 $f$ 的输入是矩阵 $\mathbf{X}\in\mathbb{R}^{n\times m}$，矩阵中的每个元素 $X_{ij}$ 都会影响函数值，因此需要分别计算 $f$ 关于每个元素的偏导数。将这些偏导数放回对应位置，就得到：
$$
\nabla_{\mathbf{X}}f
=
\frac{\partial f}{\partial\mathbf{X}}
=
\begin{bmatrix}
\frac{\partial f}{\partial X_{11}} & \cdots & \frac{\partial f}{\partial X_{1m}}\\
\vdots & \ddots & \vdots\\
\frac{\partial f}{\partial X_{n1}} & \cdots & \frac{\partial f}{\partial X_{nm}}
\end{bmatrix}
\in\mathbb{R}^{n\times m}.
$$
可以把梯度看作贴在原矩阵上的一张“变化率表”：$X_{ij}$ 所在的位置，对应记录 $\frac{\partial f}{\partial X_{ij}}$。例如输入是 $2\times3$ 矩阵，就有 $6$ 个元素和 $6$ 个对应的偏导数，因此梯度仍是 $2\times3$ 矩阵。

梯度与输入保持相同形状，才能逐元素更新矩阵：
$$
\mathbf{X}
\leftarrow
\mathbf{X}-\eta\nabla_{\mathbf{X}}f.
$$
同理，若某层权重 $\mathbf{W}^{(1)}\in\mathbb{R}^{h\times d}$，则其梯度也满足：
$$
\frac{\partial J}{\partial\mathbf{W}^{(1)}}
\in
\mathbb{R}^{h\times d}.
$$
梯度中的每个位置说明：轻微改变对应权重时，目标函数 $J$ 会沿哪个方向、以多快的速度变化。

### 二阶导数与计算图
一阶导数说明目标函数沿各个参数方向的变化速度，二阶导数进一步说明这些变化速度本身如何改变。设模型参数为 $\boldsymbol{\theta}$，第一次反向传播得到梯度：
$$
\mathbf{g}
=
\nabla_{\boldsymbol{\theta}}J.
$$
若要继续计算二阶导数，就需要再对梯度 $\mathbf{g}$ 求导，$\mathbf{H}$ 称为 Hessian 矩阵。
$$
\mathbf{H}
=
\nabla_{\boldsymbol{\theta}}^2J.
$$
普通反向传播只需得到梯度的数值，完成后可以释放计算图。计算二阶导数时，第一次反向传播中的每一步也必须被记录下来，因为第二次求导还要沿着这些步骤继续计算。例如：
$$
J(w)=w^3,
\qquad
\frac{dJ}{dw}=3w^2,
\qquad
\frac{d^2J}{dw^2}=6w.
$$
若第一次求导后只留下 $3w^2$ 的数值，而没有保留它由 $w$ 计算得到的过程，就无法继续求出 $6w$。因此，计算二阶导数会使计算图包含第一次反向传播的运算，需要保存更多中间结果，计算时间和显存占用都会增加。

若模型有 $P$ 个参数，梯度包含 $P$ 个元素，而完整 Hessian 包含 $P^2$ 个元素：
$$
\mathbf{g}\in\mathbb{R}^P,
\qquad
\mathbf{H}\in\mathbb{R}^{P\times P}.
$$
当参数很多时，显式计算和保存完整 Hessian 通常不可行。因此，实际中常只计算 Hessian 与某个向量 $\mathbf{v}$ 的乘积：
$$
\mathbf{H}\mathbf{v}
=
\nabla_{\boldsymbol{\theta}}
\left(
\nabla_{\boldsymbol{\theta}}J^\top\mathbf{v}
\right),
$$
它能够提供某个方向上的曲率信息，而不需要构造完整 Hessian。深度学习通常使用一阶梯度方法；需要曲率信息时，则采用 Hessian 向量积或近似二阶方法来控制计算成本。

### 训练方法与内存
训练时，GPU 需要保存模型参数、梯度、优化器状态，以及反向传播所需的各层激活值。模型大小主要决定参数占用，批量大小主要决定一次前向传播产生多少激活值。因此，显存不足可能来自两种情况：同时处理的样本太多，或模型本身太大。

对于训练中的内存占用，小批量训练通过减少同时处理的样本来降低激活显存，并会通过梯度噪声影响训练结果；梯度累积用更多计算步骤近似较大批量；模型并行拆分模型本身，用于解决单个 GPU 无法容纳完整模型的问题。

#### 小批量训练
小批量 $\mathcal{B}$ 使用部分样本估计完整训练集的梯度：
$$
\mathbf{g}_{\mathcal{B}}
=
\frac{1}{|\mathcal{B}|}
\sum_{i\in\mathcal{B}}\nabla l_i(\mathbf{w}).
$$
随机抽样时，该估计平均而言与完整梯度一致；若样本梯度近似独立，其方差大致满足：
$$
\operatorname{Var}(\mathbf{g}_{\mathcal{B}})
\propto
\frac{1}{|\mathcal{B}|}.
$$
较小批量占用的激活显存更少，每轮更新次数更多，但梯度波动更明显；较大批量的梯度更稳定、并行效率通常更高，但占用更多显存，每轮更新次数更少。由于梯度噪声和更新次数不同，参数会沿不同路径移动，因此批量大小会改变最终训练结果。改变批量大小时，通常还需要相应调整学习率和训练步数。

#### 梯度累积
若希望保持较大的批量效果，但显存只能容纳较小批量，可以连续处理 $K$ 个微批量并累积梯度，最后更新一次参数。若每个微批量包含 $b$ 个样本，则有效批量大小为：
$$
B_{\mathrm{eff}}=Kb.
$$
梯度累积减少了单次前向传播保存的激活值，用更多计算步骤换取更低的显存占用；但它不能解决模型参数本身放不下的问题。在有效批量相同、梯度正确归一化且累积期间不更新参数的条件下，梯度累积近似等价于直接使用大批量。实际差别主要来自批量归一化、随机操作、浮点误差，以及梯度裁剪、权重衰减和优化器更新的执行位置。

#### 模型并行
如果批量大小已经降为 $1$，模型仍无法放入单个 GPU，就需要使用模型并行，将计算图的不同部分放到不同 GPU。例如将前几层放在 GPU 1，后几层放在 GPU 2：
$$
\mathbf{x}
\xrightarrow{\text{GPU 1}}
\mathbf{h}
\xrightarrow{\text{传输}}
\mathbf{o}
\xrightarrow{\text{GPU 2}}
J.
$$
前向传播时，GPU 1 将 $\mathbf{h}$ 发送给 GPU 2；反向传播时，GPU 2 再将 $\frac{\partial J}{\partial\mathbf{h}}$ 传回 GPU 1。模型并行能够利用多个 GPU 的总显存，但会增加设备之间的通信和等待。若各部分计算量不同，还可能出现部分 GPU 空闲的问题。

模型并行可以与微批量结合形成流水线，使不同 GPU 同时处理不同微批量的不同阶段，减少设备等待。还可以使用较低精度，或在反向传播时重新计算部分激活值，进一步节省显存。

## 数值稳定性
深层网络的梯度需要经过**许多层导数连续相乘**，因此其大小可能随网络深度指数级缩小或放大。基于反向传播的梯度计算，前面层的梯度必须经过后面所有层才能传回来。网络越深，需要连续相乘的矩阵越多。

多个很小的正概率相乘时，一个常见的技巧是切换到对数空间， 即将数值表示的压力从尾数转移到指数。但梯度传播不是简单的正数相乘，而是矩阵和向量相乘。梯度包含正负、大小、方向，以及面临实际梯度更新参数等问题，更为复杂。数值不稳定，不仅仅是计算机能否表示极大或极小数的问题，还可能会**影响参数更新，无法学习有效特征**；或者导致一次更新就可能使参数发生巨大变化，**造成损失剧烈波动，甚至无法收敛**。

矩阵连乘比标量连乘更复杂。一个矩阵可能：在某些方向上放大向量；在另一些方向上缩小向量。严格来说，判断梯度大小如何变化时，矩阵的**奇异值**通常比**特征值**更直接；特征值只描述特征向量方向上的变化；而奇异值直接描述矩阵对向量长度的最大和最小缩放程度，更适合分析梯度的放大与缩小。

### 梯度消失与梯度爆炸
梯度消失：参数更新过小，在更新时几乎不会移动；sigmoid 韩式是导致梯度消失的常见原因，在 sigmoid 函数输入很大或很小时，它的梯度会很接近于零；在网络有很多层时，很可能在某一层切断梯度。

梯度爆炸：参数更新过大，破坏模型的稳定收敛；以生成 100 个 $\sigma^2 = 1$ 的高斯随机矩阵为例，将它们与初始矩阵相乘，使矩阵乘积发生爆炸；这种情况常常由于网络初始化导致，最终无法能够使梯度下降优化器收敛。

### 梯度分析中的奇异值与特征值
普通特征值要求矩阵是方阵，因为 $\mathbf{M}\mathbf{v}$ 和 $\lambda\mathbf{v}$ 必须具有相同形状。若矩阵是实对称矩阵，则所有特征值都是实数，并且存在一组相互正交的特征向量。

特征值与特征向量能够描述矩阵变换在特征向量方向上的作用：
$$
\mathbf{M}\mathbf{v}=\lambda\mathbf{v}.
$$
向量 $\mathbf{v}$ 经过矩阵变换后方向不变，长度变为原来的 $|\lambda|$ 倍。但梯度通常指向任意方向，不一定恰好是矩阵的某个特征向量；对于一般的非对称矩阵，特征向量也不一定相互垂直，因此特征值不能直接说明任意梯度的长度变化。

奇异值与奇异向量描述矩阵在一组相互正交的方向上缩放，对于第 $i$ 对奇异向量：
$$
\mathbf M\mathbf v_i=\sigma_i\mathbf u_i.
$$
右奇异向量  $v_{i}$ 表示变换前的方向，左奇异向量 $u_{i}$ 表示变换后的方向， 奇异值 $\sigma_i$ 表示长度缩放倍数。

右奇异向量构成输入空间的一组标准正交基，可以把任意向量表示为相互正交的右奇异向量线性组合：
$$
\mathbf{x}
=
\sum_i c_i\mathbf{v}_i
\qquad
c_i=\mathbf{v}_i^\top\mathbf{x}.
$$
该向量在经过矩阵变换后，位于左奇异向量构成的输出空间，可以使用左奇异向量线性表示：
$$
\mathbf{M}\mathbf{x}
=
\sum_i c_i\sigma_i\mathbf{u}_i,
$$

$$
\|\mathbf{M}\mathbf{x}\|_2^2
=
\sum_i c_i^2\sigma_i^2.
$$
任意向量都可以分解到右奇异向量方向上，因此奇异值能够用于分析向量经过矩阵后的长度变化。奇异值适用于任意形状的矩阵。若 $\mathbf{M}\in\mathbb{R}^{m\times n}$，其奇异值是 $\mathbf{M}^{\top}\mathbf{M}$ 的特征值的平方根：
$$
\sigma_i
=
\sqrt{\lambda_i(\mathbf{M}^{\top}\mathbf{M})}.
$$
由于 $\mathbf{M}^{\top}\mathbf{M}$ 是对称半正定矩阵，奇异值总是非负实数。矩阵不需要是方阵、可逆矩阵或满秩矩阵，因此神经网络中的非方阵权重也可以使用奇异值分析。

设最大和最小奇异值分别为 $\sigma_{\max}$ 和 $\sigma_{\min}$。这里将 $\sigma_{\min}(\mathbf M)$ 定义为 $\sqrt{\lambda_{\min}(\mathbf M^\top\mathbf M)}$，并包含零奇异值；若 $\mathbf M$ 存在零空间，则 $\sigma_{\min}(\mathbf M)=0$. 对于任意输入向量 $\mathbf{x}$，有：
$$
\sigma_{\min}(\mathbf{M})\|\mathbf{x}\|_2
\leq
\|\mathbf{M}\mathbf{x}\|_2
\leq
\sigma_{\max}(\mathbf{M})\|\mathbf{x}\|_2.
$$
因此，$\sigma_{\max}$ 给出向量长度最多会被放大多少，$\sigma_{\min}$ 给出向量长度最少能保留多少；某个具体向量的实际缩放程度，则由它在各个右奇异向量方向上的分量 $c_i$ 共同决定。

相比于特征值与特征向量，奇异向量描述一个输入方向经过缩放后映射到另一个输出方向，而特征向量描述矩阵作用后保持不变的方向。因此，奇异值更适合分析一般矩阵变换对向量长度的影响。

特征值与特征向量描述矩阵作用后方向保持不变的特殊方向，并保留缩放的大小和符号，因此适合分析同一个方阵被反复应用的结果：$\mathbf M^k\mathbf v=\lambda^k\mathbf v$. 对于 Hessian 这类对称矩阵，特征值还表示目标函数沿对应特征向量方向的曲率。

### Hessian 矩阵的特征值与曲率
下面说明为什么 Hessian 矩阵的特征值能够回答“损失函数沿哪些方向向上或向下弯曲”，定义 Hessian 是目标函数关于参数的二阶导数组成的矩阵：
$$
\mathbf H=\nabla_{\boldsymbol{\theta}}^2J.
$$
设参数沿方向 $\mathbf v$ 移动一小段： $\boldsymbol{\theta}'=\boldsymbol{\theta}+t\mathbf v.$
则目标函数的二阶近似为：
$$
J(\boldsymbol{\theta}+t\mathbf v)
\approx
J(\boldsymbol{\theta})
+
t\nabla J^\top\mathbf v
+
\frac{t^2}{2}\mathbf v^\top\mathbf H\mathbf v.
$$
若 $\mathbf v$是 Hessian 的单位特征向量：
$$
\mathbf H\mathbf v=\lambda\mathbf v,
\qquad
\|\mathbf v\|=1,
$$
则：
$$
\mathbf v^\top\mathbf H\mathbf v=\lambda.
$$
所以 $\lambda$ 表示目标函数沿 $\mathbf v$ 方向的曲率：$\lambda>0$ 曲线向上弯、$\lambda<0$ 曲线向下弯、$\lambda\approx0$ 该方向比较平坦。在梯度为零的位置：若 Hessian 所有特征值都为正，则是严格局部极小值；若所有特征值都为负，则是严格局部极大值；若同时存在正、负特征值，则是鞍点；若存在零特征值，仅根据 Hessian 通常无法判断。


## 网络对称性与参数初始化
神经网络中一个设计问题是参数化的排列对称性：如果多个隐藏单元从完全相同的参数出发，它们会进行相同的计算、获得相同的梯度，并在训练中始终保持相同。这样，多个神经元实际上只相当于一个神经元。对称性的本质是使多个可学习模块始终执行相同功能，从而浪费模型容量。

小批量随机梯度下降不会打破这种对称性，因为即使使用不同的小批量，两个单元在同一个小批量上看到的样本仍然相同。暂退法产生的随机噪声能够打破训练过程中的完全同步状态，但仍需使用合理的随机初始化来减轻网络结构上的排列对称性。

### Xavier 初始化
Xavier 初始化的作用是：（1）减轻同步状态；（2）控制前向传播中激活值和反向传播中梯度的大小，提升数值稳定性。

假设没有使用非线性的全连接层输出 $o_{i} = \sum_{j=1}^{n_{in}} w_{ij} x_{j} .$
权重 $w_{ij}$ 从同一分布中独立抽取，假设该分布具有零均值和方差 $\sigma^2$，这不意味着必须是高斯分布，只是均值和方差需要存在；假设输入层 $x_{j}$ 也具有零均值和方差 $\gamma^2$，并且它们独立于 $w_{ij}$ 且彼此独立；下面据此计算 $o_{i}$ 的均值和方差：
$$
\begin{align}
E[o_{i}] & = \sum_{j=1}^{n_{\mathrm{in}}} E[w_{ij} x_{j}] \\
& = \sum_{j=1}^{n_{\mathrm{in}}} E[w_{ij} ] E[x_{j}] \\
& = 0. \\ 
\end{align}
$$

$$
\begin{align}
Var[o_{i}] &= E[o_{i}^2] - (E[o_{i}])^2  \\
&= \sum_{j=1}^{n_{\mathrm{in}}}E[w_{ij}^2x_{j}^2] - 0 \\
&= \sum_{j=1}^{n_{\mathrm{in}}}E[w_{ij}^2]E[x_{j}^2] \\
&= n_{\mathrm{in}} \sigma^2 \gamma^2 .

\end{align}
$$
保持方差不变的一种方式是设置 $n_{\mathrm{in}} \sigma^2 = 1 .$ 现在考虑反向传播过程，类似地可以得出需要 $n_{\mathrm{out}} \sigma^2 = 1$ 其中 $n_{\mathrm{out}}$ 是该层的输出的数量，设输出端传回的梯度为$\delta_i=\frac{\partial J}{\partial o_i}$, 推导如下：

输入 $x_j$ 接收到的梯度需要汇总所有输出单元传回的结果：
$$
\frac{\partial J}{\partial x_j}
=
\sum_{i=1}^{n_{\mathrm{out}}}
w_{ij}\delta_i.
$$
假设 $\delta_i$ 相互独立，具有零均值和方差 $\tau^2$，并且独立于权重 $w_{ij}$，则反向梯度的均值为：
$$
\begin{align}
E\left[\frac{\partial J}{\partial x_j}\right]
&=
\sum_{i=1}^{n_{\mathrm{out}}}E[w_{ij}\delta_i]\\
&=
\sum_{i=1}^{n_{\mathrm{out}}}E[w_{ij}]E[\delta_i]\\
&=0.
\end{align}
$$
其方差为：
$$
\begin{align}
\operatorname{Var}\left[\frac{\partial J}{\partial x_j}\right]
&=
E\left[\left(\frac{\partial J}{\partial x_j}\right)^2\right]\\
&=
\sum_{i=1}^{n_{\mathrm{out}}}E[w_{ij}^2\delta_i^2]\\
&=
\sum_{i=1}^{n_{\mathrm{out}}}E[w_{ij}^2]E[\delta_i^2]\\
&=
n_{\mathrm{out}}\sigma^2\tau^2.
\end{align}
$$
为了使梯度经过该层反向传播后仍保持方差 $\tau^2$，需要 $n_{\mathrm{out}}\sigma^2=1.$ 结合上一个结论可知，前向传播要求 $n_{\mathrm{in}}\sigma^2=1$，反向传播要求 $n_{\mathrm{out}}\sigma^2=1$. $n_{\mathrm{in}}\neq n_{\mathrm{out}}$ 时，两者无法同时严格满足。

Xavier 初始化在二者之间取折中：
$$
\sigma^2
=
\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}.
$$
该选择使前向激活值和反向梯度的方差都不会随着网络层数快速地放大或缩小。一般地，Xavier 初始化从均值为零，方差为 $\sigma^2=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$ 的高斯分布中采样权重。我们也可以将其改为选择从均匀分布中抽取权重时的方差。 注意均匀分布$U(-a, a)$的方差为$\frac{a^2}{3}$，将$\frac{a^2}{3}$代入到$\sigma^2$的条件中，将得到初始化值域:

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

尽管在上述数学推导中使用“不存在非线性”的假设，但 Xavier 初始化方法在实践中被证明是有效的。此外，Xavier初始化表明，对于每一层，输出的方差不受输入数量的影响，任何梯度的方差不受输出数量的影响。

#### 为什么要保持前向与反向传播的方差

深层网络的前向传播和反向传播都包含许多次矩阵乘法。即使每一层只使数值尺度发生轻微变化，这些变化也会随着网络深度不断累积，最终造成激活值或梯度的消失、爆炸。

对于前向传播，前面已经得到单层输出的方差：
$$
\operatorname{Var}(o_i)
=
n_{\mathrm{in}}\sigma^2\operatorname{Var}(x_j).
$$
因此，每经过一个线性层，激活值的方差大约乘以：
$$
c_{\mathrm{forward}}
=
n_{\mathrm{in}}\sigma^2.
$$
若连续 $L$ 层的缩放系数都近似为 $c_{\mathrm{forward}}$，则有：
$$
\operatorname{Var}(\mathbf h_L)
\approx
c_{\mathrm{forward}}^L
\operatorname{Var}(\mathbf x).
$$
当 $c_{\mathrm{forward}}<1$ 时，激活值的方差会随深度指数级缩小，不同输入产生的表示可能逐渐趋于相同，后面的层难以获得有效信息；当 $c_{\mathrm{forward}}>1$ 时，激活值的方差会不断增大，可能导致数值溢出，或者使 sigmoid、tanh 等激活函数进入饱和区。只有当 $c_{\mathrm{forward}}\approx1$ 时，各层激活值才能保持相近的数值尺度，使输入信息更容易传递到深层。

对于反向传播，输入端接收到的梯度方差为：
$$
\operatorname{Var}\left[\frac{\partial J}{\partial x_j}\right]
=
n_{\mathrm{out}}\sigma^2\operatorname{Var}(\delta_i).
$$
因此，梯度每反向经过一个线性层，其方差大约乘以：
$$
c_{\mathrm{backward}}
=
n_{\mathrm{out}}\sigma^2.
$$
若 $c_{\mathrm{backward}}<1$，梯度会逐层衰减，靠近输入的层几乎无法获得有效的更新信号；若 $c_{\mathrm{backward}}>1$，梯度会逐层放大，使参数更新剧烈波动，甚至导致训练发散。因此，保持反向梯度的方差，是为了让网络中的不同层都能获得尺度合适、相对稳定的学习信号。

前向传播和反向传播分别希望满足：
$$
n_{\mathrm{in}}\sigma^2\approx1,
\qquad
n_{\mathrm{out}}\sigma^2\approx1.
$$
当 $n_{\mathrm{in}}\neq n_{\mathrm{out}}$ 时，两个条件不能同时严格成立。Xavier 初始化令平均连接数对应的缩放系数为 $1$：
$$
\frac{n_{\mathrm{in}}+n_{\mathrm{out}}}{2}\sigma^2=1,
$$
从而得到：
$$
\sigma^2
=
\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}.
$$
这相当于在前向信号和反向梯度之间取得折中。

### 其它权重初始化方法

Xavier 初始化希望同时兼顾前向激活值和反向梯度的大小，但不同激活函数会以不同方式改变输入，因此还需要根据激活函数和网络结构调整初始权重。

#### He 初始化
He 初始化主要用于 ReLU 及其变体。设输入 $z$ 关于零点近似对称，经过 ReLU 后，约有一半的值被置为零，因此输出的二阶矩约为：
$$
E[\operatorname{ReLU}(z)^2]
\approx
\frac{1}{2}E[z^2].
$$
为了补偿这一缩小作用，He 初始化令权重均值为零、方差为：
$$
\sigma^2=\frac{2}{n_{\mathrm{in}}}.
$$
这里保持的是激活值的二阶矩，即数值的整体大小；由于 ReLU 的输出均值通常不为零，它并不等同于严格保持方差。对于负半轴斜率为 $a$ 的 Leaky ReLU，正、负两部分共同产生的缩放系数约为 $\frac{1+a^2}{2}$，因此：
$$
\sigma^2
=
\frac{2}{(1+a^2)n_{\mathrm{in}}}.
$$
若使用 $n_{\mathrm{in}}$，主要保持前向传播中的信号大小；若改用 $n_{\mathrm{out}}$，则主要保持反向传播中的梯度大小。

#### LeCun 初始化
LeCun 初始化只根据输入数量设置权重方差：
$$
\sigma^2=\frac{1}{n_{\mathrm{in}}}.
$$
它常与 SELU 激活函数配合，使各层激活值逐渐接近零均值和单位方差。不过，这种自归一化效果还依赖输入经过标准化、网络以全连接层为主，并使用与 SELU 配套的设置，不能只靠初始化单独保证。

#### 正交初始化

Xavier 和 He 初始化主要控制权重的方差，使激活值与梯度的**平均尺度**保持稳定；但即使矩阵元素的整体方差合适，权重矩阵仍可能沿某些方向强烈放大向量、沿另一些方向显著压缩向量。正交初始化进一步从奇异值的角度控制这种方向差异。

正交初始化先构造正交矩阵 $\mathbf Q$，再令：
$$
\mathbf W=g\mathbf Q,
$$
其中 $g$ 控制所有方向的整体缩放。对于方阵，若 $\mathbf Q^\top\mathbf Q=\mathbf I$，则：
$$
\begin{aligned}
\|\mathbf W\mathbf x\|_2^2
&=
(g\mathbf Q\mathbf x)^\top(g\mathbf Q\mathbf x)\\
&=
g^2\mathbf x^\top\mathbf Q^\top\mathbf Q\mathbf x\\
&=
g^2\|\mathbf x\|_2^2.
\end{aligned}
$$
因此：
$$
\|\mathbf W\mathbf x\|_2
=
|g|\|\mathbf x\|_2.
$$
正交矩阵 $\mathbf Q$ 可以理解为只对输入进行旋转或翻转，不改变向量长度；乘上 $g$ 后，所有方向都被统一缩放 $|g|$ 倍。从奇异值的角度看，$\mathbf Q$ 的所有奇异值为 $\sigma_i(\mathbf Q)=\sqrt{\lambda_i(\mathbf Q^\top\mathbf Q)}=\sqrt{\lambda_i(\mathbf I)}=1$.

考虑：
$$
\begin{aligned}
\mathbf W^\top\mathbf W
&=
(g\mathbf Q)^\top(g\mathbf Q)\\
&=
g^2\mathbf Q^\top\mathbf Q\\
&=
g^2\mathbf I.
\end{aligned}
$$

矩阵 $g^2\mathbf I$ 的所有特征值都等于 $g^2$，所以 $\mathbf W$ 的所有奇异值为 $\sigma_i(\mathbf W)=\sqrt{\lambda_i(\mathbf W^\top\mathbf W)}=\sqrt{g^2}=|g|$. 从几何上看，奇异值表示矩阵沿各个正交方向缩放向量长度的倍数。$\mathbf Q$ 只旋转或翻转向量，所以每个方向的缩放倍数都是 $1$；$\mathbf W=g\mathbf Q$ 在此基础上又将所有方向统一缩放 $|g|$ 倍，这使单位球经过线性变换后仍是统一缩放的球，而不会因为不同方向的缩放程度差异过大而变成长而扁的椭球。

若深层线性网络中的每层都是方阵，并且都使用 $g=1$ 的正交权重，则正交矩阵的乘积仍然是正交矩阵：
$$
\mathbf h_L
=
\mathbf Q_L\mathbf Q_{L-1}\cdots\mathbf Q_1\mathbf x,
\qquad
\|\mathbf h_L\|_2=\|\mathbf x\|_2.
$$
反向传播时会乘以 $\mathbf Q^\top$，它同样保持向量长度，因此梯度也不会仅仅因为线性层的连续相乘而消失或爆炸。这种性质对深层网络有帮助，对反复使用同一个权重矩阵的循环网络尤其有意义。

对于非方阵 $\mathbf Q\in\mathbb R^{m\times n}$，需要根据输入和输出维度选择让行或列正交：

- 当 $m\geq n$ 时，可以使列正交，即 $\mathbf Q^\top\mathbf Q=\mathbf I_n$。此时所有输入方向的长度仍可得到保持。
- 当 $m<n$ 时，只能使行正交，即 $\mathbf Q\mathbf Q^\top=\mathbf I_m$。由于输出维度小于输入维度，矩阵必然存在零空间，不可能保留所有输入方向的信息，但其非零奇异值仍可保持一致。

这里所说的“零奇异值”，本质上是指矩阵存在零空间：零空间 $\operatorname{Null}(\mathbf Q)=\{\mathbf x\in\mathbb R^n\mid\mathbf Q\mathbf x=\mathbf0\}$ 包含所有经过 $\mathbf Q$ 后被完全压缩为零的输入方向。根据秩-零化度定理，$\operatorname{rank}(\mathbf Q)+\dim\operatorname{Null}(\mathbf Q)=n$；行正交意味着 $\operatorname{rank}(\mathbf Q)=m$，因此 $\dim\operatorname{Null}(\mathbf Q)=n-m>0$，即恰好存在 $n-m$ 个线性无关的输入方向会被完全丢弃。从奇异值角度看，若将输入空间的全部 $n$ 个方向计入，这 $n-m$ 个零空间方向对应 $\mathbf Q^\top\mathbf Q$ 的 $n-m$ 个零特征值，即扩展意义下的 $n-m$ 个零奇异值；奇异值为零表示：矩阵会把某个非零输入方向完全压缩成零向量，该方向上的信息被彻底丢失。

#### LSUV 初始化

正交初始化的缺点是：正交性只约束线性变换。加入激活函数后，网络在当前位置的局部变换为：
$$
\mathbf J
=
\operatorname{diag}\left(\phi'(\mathbf z)\right)\mathbf W.
$$
ReLU 会使部分导数变成 $0$，sigmoid 和 tanh 的导数也可能明显小于 $1$，因此整个雅可比矩阵不再正交，信号和梯度的长度也不再严格保持。正交初始化只能提供一个方向缩放较均匀的起点，不能单独消除数据分布、偏置和非线性激活造成的尺度变化。

LSUV（Layer-Sequential Unit-Variance）初始化在正交初始化的基础上，使用一个真实的小批量对各层输出进行逐层校正。其基本过程是：

1. 先使用正交初始化生成各层权重。
2. 将一个真实小批量输入网络，按前向顺序计算当前层实际输出。
3. 若该层输出的经验方差为 $v$，则近似缩放权重：

$$
\mathbf W\leftarrow\frac{\mathbf W}{\sqrt v},
$$

4. 对该层重复前向计算和缩放，直到输出方差接近 $1$，再继续校正下一层。

如果该层完全是线性的，将权重除以 $\sqrt v$ 可以直接把输出方差缩放到接近 $1$；存在非线性激活函数时，方差不一定按相同比例变化，因此通常需要重复校正。LSUV 相当于根据真实数据为每层自动调整整体缩放因子 $g$，从而把输入分布和激活函数的实际影响考虑进来。


#### 特定网络结构的初始化
残差网络满足：
$$
\mathbf h_{l+1}
=
\mathbf h_l+\mathbf F_l(\mathbf h_l).
$$
若许多残差分支从一开始就产生较大的输出，它们会随着深度不断累加。因此，可以缩小残差分支初始权重，或者将分支最后一层初始化为零，使网络初始时接近恒等映射 $\mathbf h_{l+1}\approx\mathbf h_l$。Fixup 初始化进一步根据网络深度缩放残差分支，用于减轻未使用归一化层时梯度不稳定。

课本提到的 Delta-Orthogonal 初始化是卷积网络专用的正交初始化。它使整个卷积变换尽量保持输入长度，让网络输入到输出的雅可比矩阵的奇异值集中在 $1$ 附近，从而使信号和梯度经过很多层后仍能保留合适的大小。在特定激活函数和参数设置下，这种方法可以支持极深卷积网络的训练。

不同初始化方法的共同目标都是打破隐藏单元之间的同步状态，并使前向信号和反向梯度保持合适大小。通常，tanh 等激活函数使用 Xavier 初始化，ReLU 及其变体使用 He 初始化，SELU 使用 LeCun 初始化；正交、LSUV 和残差网络专用初始化则进一步利用了矩阵性质、真实数据或网络结构。偏置通常可以初始化为零，因为随机权重已经能够打破隐藏单元之间的对称性。

### 可交换模块的对称性
“可交换模块对称性”广泛存在于其他网络中：

- 卷积网络中的多个卷积核如果处于完全对称的参数和连接状态，可能始终学习相同特征。
- 多头注意力中的多个头如果参数和后续连接也完全对称，可能产生相同的注意力模式。
- 混合专家模型中的多个专家如果初始化、路由和输入都完全相同，可能始终成为相同的专家。
- 多分支网络中的同构分支如果以完全相同的方式初始化和聚合，也可能退化为同一个分支的重复。

除此之外，还有不是排列造成参数对称性。例如 ReLU 满足正齐次性：

$$
\operatorname{ReLU}(cz)
=
c\operatorname{ReLU}(z),
\qquad c>0.
$$

因此：

$$
a\operatorname{ReLU}(\mathbf w^\top\mathbf x)
=
\frac{a}{c}
\operatorname{ReLU}\left((c\mathbf w)^\top\mathbf x\right).
$$

将输入权重放大 $c$ 倍、输出权重缩小为原来的 $1/c$，网络函数完全不变。这是一种连续的缩放对称性。它通常不会使两个神经元始终同步，但会导致同一个函数对应无穷多组参数，形成平坦方向并影响优化条件。权重衰减、归一化或约束参数尺度可以减轻其影响。

图神经网络还可能表现出数据结构带来的对称性。例如在一个环形图中，如果所有节点具有相同初始特征，那么结构上对称的节点经过相同的消息传递后仍会获得相同表示。如果任务要求区分这些节点，就必须加入位置编码、锚点特征或其他外部信息。不过，如果任务本来就要求节点置换不变性，那么这种对称性是有意设计的。

### 矩阵乘积的谱界与梯度稳定性

深层网络的反向传播包含许多层雅可比矩阵的乘积。教材习题要求查找两个矩阵乘积的特征值界，其目的在于说明：每一层轻微的放大或压缩都可能在连乘中不断累积，最终造成梯度爆炸或梯度消失。不过，对于一般神经网络中的非对称、非正规矩阵，特征值只描述特征向量方向上的作用，梯度长度的变化应主要使用奇异值分析。

#### 一般矩阵的特征值界

设两个同阶方阵为 $\mathbf A$ 和 $\mathbf B$，并记 $\mathbf P=\mathbf A\mathbf B$。若 $\lambda_i(\mathbf P)$ 是 $\mathbf P$ 的一个特征值，对应单位特征向量为 $\mathbf v_i$，则：
$$
\mathbf P\mathbf v_i
=
\lambda_i(\mathbf P)\mathbf v_i,
$$
所以：
$$
|\lambda_i(\mathbf P)|
=
\|\mathbf P\mathbf v_i\|_2.
$$
根据最大、最小奇异值对任意单位向量的缩放界，可以得到：
$$
\sigma_{\min}(\mathbf P)
\leq
|\lambda_i(\mathbf P)|
\leq
\sigma_{\max}(\mathbf P).
$$
另一方面，矩阵乘积的奇异值满足：
$$
\sigma_{\max}(\mathbf A\mathbf B)
\leq
\sigma_{\max}(\mathbf A)\sigma_{\max}(\mathbf B),
$$
$$
\sigma_{\min}(\mathbf A\mathbf B)
\geq
\sigma_{\min}(\mathbf A)\sigma_{\min}(\mathbf B).
$$
因此，$\mathbf A\mathbf B$ 的任意特征值都满足：
$$
\sigma_{\min}(\mathbf A)\sigma_{\min}(\mathbf B)
\leq
|\lambda_i(\mathbf A\mathbf B)|
\leq
\sigma_{\max}(\mathbf A)\sigma_{\max}(\mathbf B).
$$
若矩阵奇异，最小奇异值为 $0$，左侧只能给出平凡下界。这些乘积奇异值不等式可参考 [Stanford EE263 课程资料，第 14.14 题](https://web.stanford.edu/class/archive/ee/ee263/ee263.1082/notes/ee263coursereader.pdf)；谱范数等于最大奇异值以及 $\|\mathbf A\mathbf B\|_2\leq\|\mathbf A\|_2\|\mathbf B\|_2$ 的推导可参考 [MIT 18.06 课程笔记](https://web.mit.edu/18.06/www/Fall19/Lecture%20notes.pdf)。

谱范数 $\|\mathbf A\|_2$ 表示矩阵对任意非零向量的 $L_2$ 长度所能产生的最大放大倍数：$\|\mathbf A\|_2=\max_{\mathbf x\neq\mathbf0}\frac{\|\mathbf A\mathbf x\|_2}{\|\mathbf x\|_2}$。若 $\mathbf A=\mathbf U\mathbf\Sigma\mathbf V^\top$ 是奇异值分解，则左右的正交变换不改变向量长度，矩阵的缩放完全由 $\mathbf\Sigma$ 的对角元素决定；任意输入向量经过变换后的长度至多被最大奇异值 $\sigma_{\max}(\mathbf A)$ 放大，并且当输入取对应的右奇异向量时恰好达到这一上界。具体推导为：由于正交矩阵不改变 $L_2$ 范数，令 ( $\mathbf z=\mathbf V^\top\mathbf x$)，则：
$$
\|\mathbf A\mathbf x\|_2^2
=
\|\mathbf\Sigma\mathbf z\|_2^2
=
\sum_i \sigma_i^2 z_i^2
\leq
\sigma_1^2\sum_i z_i^2
=
\sigma_1^2\|\mathbf x\|_2^2.
$$

因此有 $\|\mathbf A\|_2=\sigma_{\max}(\mathbf A)=\sqrt{\lambda_{\max}(\mathbf A^\top\mathbf A)}$。直观上，单位球经 $\mathbf A$ 变为椭球，谱范数就是其最长半轴的长度。它通常不等于最大特征值模：例如 $\begin{bmatrix}0&K\\0&0\end{bmatrix}$ 的特征值都为 $0$，但最大奇异值和谱范数都是 $K$；这说明特征值只描述保持方向不变的特殊方向，谱范数则描述所有方向中的最严重长度放大。

非正规矩阵可能在并非自身特征向量的方向上产生很大的瞬时放大；各因子的特征值没有记录这种作用，而奇异值能够给出任意方向上长度变化的界。

#### 对称正定矩阵的特例

如果 $\mathbf A$ 和 $\mathbf B$ 都是对称正定矩阵，则 $\mathbf A\mathbf B$ 与对称正定矩阵 $\mathbf A^{1/2}\mathbf B\mathbf A^{1/2}$ 相似，因此其特征值都是正实数，并满足：
$$
\lambda_{\min}(\mathbf A)\lambda_{\min}(\mathbf B)
\leq
\lambda_i(\mathbf A\mathbf B)
\leq
\lambda_{\max}(\mathbf A)\lambda_{\max}(\mathbf B).
$$
该结论可以通过 Rayleigh 商理解：$\mathbf B$ 对任意方向的缩放位于其最小和最大特征值之间，$\mathbf A$ 又提供一层相同形式的界。但神经网络的层雅可比矩阵通常并非对称正定矩阵，因此不能普遍使用这个更强的特征值界。

#### 深层网络中的梯度界

设梯度在反向传播中依次经过 $L$ 个雅可比矩阵：
$$
\mathbf g_0
=
\mathbf M_1\mathbf M_2\cdots\mathbf M_L\mathbf g_L.
$$
反复使用乘积奇异值界，可以得到：
$$
\left(
\prod_{\ell=1}^{L}\sigma_{\min}(\mathbf M_\ell)
\right)
\|\mathbf g_L\|_2
\leq
\|\mathbf g_0\|_2
\leq
\left(
\prod_{\ell=1}^{L}\sigma_{\max}(\mathbf M_\ell)
\right)
\|\mathbf g_L\|_2.
$$
该式给出以下启示：

- 若各层的最大奇异值长期小于 $1$，梯度上界会随深度指数级衰减。
- 若许多层的最大奇异值大于 $1$，某些方向的梯度可能被指数级放大。
- 若某层的最小奇异值很小或为 $0$，至少存在一些方向被严重压缩或完全丢弃。
- 若各层的奇异值都接近 $1$，不同方向上的梯度更容易保持合适的尺度。

即使每层的奇异值只略微偏离 $1$，例如都位于 $[1-\epsilon,1+\epsilon]$，经过 $L$ 层后，最坏情况下的整体缩放范围仍可能达到：
$$
(1-\epsilon)^L
\quad\text{到}\quad
(1+\epsilon)^L.
$$
因此，网络越深，每层很小的尺度偏差越可能在连乘中变成明显的梯度消失或爆炸。若各层都是可逆方阵，乘积的条件数还满足：
$$
\kappa_2(\mathbf M_1\cdots\mathbf M_L)
\leq
\prod_{\ell=1}^{L}\kappa_2(\mathbf M_\ell),
$$
说明每层不同方向之间的不均匀缩放也可能随深度累积。不过，这些乘积界描述的是最坏情况，实际缩放还取决于相邻层奇异方向之间的对齐关系和当前梯度的方向。

#### 对参数初始化的启示

Xavier 和 He 初始化控制的是独立性等假设下激活值与梯度的**平均方差**，可以避免整体数值尺度快速变化，但平均方差合适并不保证所有方向都具有合适的缩放。一个随机权重矩阵可能平均方差正常，同时具有很大的最大奇异值和很小的最小奇异值。

正交初始化进一步控制方向差异。对于增益 $g=1$ 的方阵正交权重：
$$
\sigma_{\min}(\mathbf W)
=
\sigma_{\max}(\mathbf W)
=
1,
$$
所以纯线性层既不会放大也不会压缩任意输入方向。然而，加入激活函数后，每层的实际雅可比矩阵变为：
$$
\mathbf M_\ell
=
\operatorname{diag}\left(\phi'(\mathbf z_\ell)\right)\mathbf W_\ell.
$$
sigmoid 的导数最大只有 $1/4$，会持续压缩梯度；ReLU 则会将部分导数置为 $0$，使最坏情况下的最小奇异值可能为 $0$。因此，仅让权重矩阵正交并不足以保证整个非线性网络的梯度稳定，还需要同时考虑激活函数、权重增益、归一化和残差结构。

> 总结：两个一般矩阵的乘积可以使用最大和最小奇异值给出可靠的特征值模界，但不能只根据各矩阵自身的特征值判断乘积行为。对于梯度传播，更重要的是让各层乃至整个网络雅可比矩阵的奇异值不过度偏离 $1$，使不同方向的梯度既不过度放大，也不过度压缩；这是比仅控制权重方差更严格的稳定性目标。

### 按层自适应速率缩放与发散修正

如果训练过程中已经知道某些层的梯度或参数更新正在发散，可以在每次更新参数之前，根据该层当前的尺度动态调整学习率。

标准随机梯度下降对所有层使用相同的学习率。设第 $\ell$ 层的权重和梯度分别为 $\mathbf w_t^{(\ell)}$ 和 $\mathbf g_t^{(\ell)}$，则更新为：
$$
\mathbf w_{t+1}^{(\ell)}
=
\mathbf w_t^{(\ell)}
-
\gamma_t\mathbf g_t^{(\ell)}.
$$
此时该层参数的相对变化幅度约为：
$$
\frac{\|\Delta\mathbf w_t^{(\ell)}\|_2}
{\|\mathbf w_t^{(\ell)}\|_2}
=
\gamma_t
\frac{\|\mathbf g_t^{(\ell)}\|_2}
{\|\mathbf w_t^{(\ell)}\|_2}.
$$
不同层的权重范数与梯度范数之比可能相差多个数量级，因此同一个全局学习率对某些层可能过大，使一次更新达到甚至超过权重本身的大小；对另一些层又可能过小，使参数几乎不发生变化。

LARS（Layer-wise Adaptive Rate Scaling）为每一层计算局部缩放系数。考虑权重衰减系数 $\beta$ 时，可以写成：
$$
r_t^{(\ell)}
=
\eta
\frac{\|\mathbf w_t^{(\ell)}\|_2}
{\|\mathbf g_t^{(\ell)}\|_2
+\beta\|\mathbf w_t^{(\ell)}\|_2},
$$
其中 $\eta<1$ 是信任系数，控制单次更新相对于当前权重的大小；$\gamma_t$ 仍是由全局学习率调度器给出的学习率。忽略动量时，参数更新为：
$$
\mathbf w_{t+1}^{(\ell)}
=
\mathbf w_t^{(\ell)}
-
\gamma_t r_t^{(\ell)}
\left(
\mathbf g_t^{(\ell)}
+\beta\mathbf w_t^{(\ell)}
\right).
$$

当 $\|\mathbf w_t^{(\ell)}\|_2$ 和 $\beta$ 固定时，梯度范数 $\|\mathbf g_t^{(\ell)}\|_2$ 越大，分母越大，局部缩放系数 $r_t^{(\ell)}$ 越小；梯度范数越小，$r_t^{(\ell)}$ 越大。因此，LARS 会缩小梯度较大层的有效学习率 $\gamma_t r_t^{(\ell)}$，并增大梯度较小层的有效学习率，以抵消不同层之间梯度整体尺度的差异。

令本次参数变化为 $\Delta\mathbf w_t^{(\ell)}=\mathbf w_{t+1}^{(\ell)}-\mathbf w_t^{(\ell)}$。将局部缩放系数代入更新式，可得：
$$
\begin{aligned}
\frac{\|\Delta\mathbf w_t^{(\ell)}\|_2}
{\|\mathbf w_t^{(\ell)}\|_2}
&=
\gamma_t\eta
\frac{
\|\mathbf g_t^{(\ell)}+\beta\mathbf w_t^{(\ell)}\|_2
}
{
\|\mathbf g_t^{(\ell)}\|_2
+\beta\|\mathbf w_t^{(\ell)}\|_2
}\\
&\leq
\gamma_t\eta,
\end{aligned}
$$
其中不等式来自三角不等式 $\|\mathbf g_t^{(\ell)}+\beta\mathbf w_t^{(\ell)}\|_2\leq\|\mathbf g_t^{(\ell)}\|_2+\beta\|\mathbf w_t^{(\ell)}\|_2$。因此，在忽略动量时，无论某层的梯度范数多大，该层单次更新相对于当前权重的比例都满足：
$$
\frac{\|\Delta\mathbf w_t^{(\ell)}\|_2}
{\|\mathbf w_t^{(\ell)}\|_2}
\leq
\gamma_t\eta.
$$
这说明 LARS 控制的是参数的相对更新量，而不是梯度的绝对大小：较大的有限梯度会自动配合较小的局部缩放系数，从而避免参数仅因梯度尺度过大而一步移动过远。具体方法可参考 [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)。

LARS 最初用于稳定大批量训练：增大批量通常需要提高全局学习率，但不同层能够承受的更新幅度不同，因此需要按层调整有效学习率。使用动量时，LARS 直接限制本轮梯度与权重衰减项经过缩放后的相对大小；参数的实际更新还包含历史动量，因此其相对更新量不一定仍以 $\gamma_t\eta$ 为上界。
