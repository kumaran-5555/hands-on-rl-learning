# E.1.2 贝尔曼方程的矩阵形式

> **前置知识**：[E.1.1 向量与矩阵](./linear-algebra-basics)——你需要知道向量、矩阵和矩阵乘法。同时建议你已经读过第 3 章的[贝尔曼方程](../chapter03_mdp/bellman-equation)，知道单个状态的贝尔曼方程长什么样。

---

## 这一篇要做什么

第 3 章里，你学会了写单个状态的贝尔曼方程：

$$
V^\pi(s) = \sum_{a} \pi(a|s)\left[R(s,a) + \gamma\sum_{s'} P(s'|s,a)V^\pi(s')\right].
$$

这行公式处理一个状态没问题。但如果环境有 1000 个状态，就要写 1000 个这样的方程。能不能把它们**压缩成一个**？

本篇的目标就是得到这个**完成体公式**——不管状态数多少，贝尔曼方程始终是一行矩阵方程：

$$
\boxed{\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}}
$$

其中 $\boldsymbol{v}$ 是所有状态的价值向量，$\boldsymbol{r}$ 是所有状态的即时奖励，$P$ 是状态转移矩阵。移项后还有闭式解：

$$
\boxed{\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}}
$$

下面从你已知的单个方程出发，一步步推导出这两个公式。推导分三步：先把价值写成向量，再把转移写成矩阵，最后拼成方程组并求解。

---

## 第一步：把所有状态的价值排成向量

沿用附录导读中的两状态例子：

- 在 $s_1$，奖励是 $2$，下一步一定到 $s_2$。
- 在 $s_2$，奖励是 $1$，下一步一定到 $s_1$。
- 折扣因子 $\gamma = 0.5$。

两个状态的价值分别是 $v_1$ 和 $v_2$。上一篇学过向量——把所有状态的价值排成一列：

$$
\boldsymbol{v} =
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix},
\qquad
\boldsymbol{r} =
\begin{bmatrix}
2 \\
1
\end{bmatrix}.
$$

$\boldsymbol{v}$ 是"我们想求的"，$\boldsymbol{r}$ 是"已知的即时奖励"。两个状态的时候，向量只有两个分量；如果有 $n$ 个状态，向量就有 $n$ 个分量，但写法不变。

---

## 第二步：把转移关系写成矩阵

从 $s_1$ 出发一定到 $s_2$，从 $s_2$ 出发一定到 $s_1$。上一篇学过转移矩阵：

$$
P =
\begin{array}{c@{\;}c}
& \begin{array}{cc}
\to s_1 & \to s_2
\end{array} \\
\begin{array}{c}
s_1 \\
s_2
\end{array}
&
\left[
\begin{array}{cc}
0 & 1 \\
1 & 0
\end{array}
\right]
\end{array}.
$$

回忆矩阵乘向量的含义：$P\boldsymbol{v}$ 的第 $i$ 行是"从状态 $s_i$ 出发，下一步期望到达的价值"。上一篇已经验算过：

$$
P\boldsymbol{v} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
v_2 \\
v_1
\end{bmatrix}.
$$

这正是在说：从 $s_1$ 出发下一步到 $s_2$，所以未来价值是 $v_2$；从 $s_2$ 出发下一步到 $s_1$，所以未来价值是 $v_1$。矩阵乘向量自动完成了"概率加权求和"。

---

## 第三步：拼成方程组

现在有了三样东西：价值向量 $\boldsymbol{v}$、奖励向量 $\boldsymbol{r}$、转移矩阵 $P$。把单个状态的贝尔曼方程写成手算形式：

$$
\begin{aligned}
v_1 &= 2 + 0.5v_2, \\
v_2 &= 1 + 0.5v_1.
\end{aligned}
$$

用矩阵语言重写右边：即时奖励是 $\boldsymbol{r}$，折扣后的未来价值是 $\gamma P\boldsymbol{v}$。拼在一起：

$$
\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}.
$$

验证右边第一行：

$$
2 + 0.5 \times (0 \cdot v_1 + 1 \cdot v_2) = 2 + 0.5v_2.
$$

右边第二行：

$$
1 + 0.5 \times (1 \cdot v_1 + 0 \cdot v_2) = 1 + 0.5v_1.
$$

和手写方程完全一致。矩阵形式没有引入新东西，只是把大量相似方程压缩成一个。

### 这个压缩为什么有效？

关键在 $P\boldsymbol{v}$ 这一步。$P$ 的每一行恰好是一组转移概率（行和为 $1$），矩阵乘法恰好是"概率 $\times$ 价值"的加权平均。贝尔曼方程说"价值 = 即时奖励 + 折扣后的未来价值"，而矩阵方程说完全相同的事——只是对**所有状态同时**说。

| 符号                     | 含义                         | 维度          |
| ------------------------ | ---------------------------- | ------------- |
| **v**       | 所有状态的价值（我们想求的） | n × 1  |
| **r**       | 所有状态的即时奖励           | n × 1  |
| γP**v** | 折扣后的概率加权未来价值     | n × 1  |

三个量的维度都是 $n \times 1$，等号两边形状一致，公式才有意义。这就是第 3 章里 DP（动态规划）方法背后做的事——反复应用 $v_{k+1} = r + \gamma Pv_k$，直到收敛。

---

## 闭式解：能不能一步到位？

既然 $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ 是一个线性方程组，一个自然的想法是：**能不能直接解出 $\boldsymbol{v}$？** 就像解 $2x = 6$ 得到 $x = 3$ 一样，移项、求逆，一步到位。

从 $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ 出发，把含有 $\boldsymbol{v}$ 的项移到左边：

$$
\boldsymbol{v} - \gamma P\boldsymbol{v} = \boldsymbol{r}.
$$

提取公因子（$I$ 是单位矩阵——对角线为 $1$，其余为 $0$）：

$$
(I - \gamma P)\boldsymbol{v} = \boldsymbol{r}.
$$

如果 $I - \gamma P$ 可逆，就有闭式解：

$$
\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}.
$$

带入两状态的具体数字：

$$
I - \gamma P =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
- 0.5
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}.
$$

解方程组：

$$
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
2 \\
1
\end{bmatrix}
\quad\Longrightarrow\quad
v_1 = 3.33,\quad v_2 = 2.67.
$$

### 几何直觉：交点就是不动点

这个方程组可以画成两条直线：

- 第一个方程 $v_1 - 0.5v_2 = 2$
- 第二个方程 $-0.5v_1 + v_2 = 1$

在 $(v_1, v_2)$ 平面上，两条直线的交点就是 $(3.33, 2.67)$。这个交点也叫**不动点**——把 $(3.33, 2.67)$ 代入贝尔曼更新 $v_{new} = r + 0.5Pv$，得到的仍然是 $(3.33, 2.67)$。价值不再变化，说明我们找到了真实价值。

---

## 从两个状态到 $n$ 个状态

上面的闭式解 $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$ 是用两个状态推导的。状态更多时公式还成立吗？答案是肯定的——从 2 个状态到 1000 个状态，方程形式完全不变：

$$
\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma P_\pi \boldsymbol{v}_\pi.
$$

其中：

- $\boldsymbol{v}_\pi \in \mathbb{R}^n$：策略 $\pi$ 下所有状态的价值
- $\boldsymbol{r}_\pi \in\mathbb{R}^n$：每个状态的期望即时奖励
- $P_\pi \in\mathbb{R}^{n\times n}$：策略诱导的转移矩阵（$P_\pi[i,j] = \sum_a \pi(a\mid s_i) p(s_j\mid s_i, a)$）

三个状态时，$P$ 变成 $3\times3$，$\boldsymbol{v}$ 和 $\boldsymbol{r}$ 变成 $3\times1$，方程 $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ 仍然成立。这就是矩阵表示的威力：**不管状态数多少，方程形式不变**。

### $I - \gamma P$ 什么时候可逆？

求解的关键是 $I - \gamma P$ 必须可逆。直觉上，这要求贝尔曼更新不能"发散"。当 $0 < \gamma < 1$ 且 $P$ 是合法转移矩阵（每行概率和为 $1$）时，$I - \gamma P$ 几乎总是可逆的。

更严格地说，$\gamma P$ 的谱半径（最大特征值的绝对值）满足 $\rho(\gamma P) \leq \gamma < 1$，所以 $I - \gamma P$ 的特征值都远离 $0$，矩阵一定可逆。E.1.4 会详细解释为什么。

---

## 为什么实际训练中不直接求逆？

推导到这里，闭式解看起来很完美：写出矩阵形式，求逆，得到答案。但这条路在实际中走不通，原因有三个：

1. **规模太大。** 如果状态数 $n=10^6$，矩阵 $I-\gamma P$ 是 $10^6 \times 10^6$，求逆的计算量是 $O(n^3)$，几乎不可能。
2. **矩阵未必显式存在。** 在很多实际问题中，我们不知道 $P$ 的具体数值，只能通过采样观察状态转移。第 3 章讲的 MC 和 TD 方法就是在不知道 $P$ 的情况下工作的。
3. **状态可能是连续的。** 如果状态是图像或文本，就不存在有限大小的矩阵——根本建不出 $P$。

实际算法用迭代方法逼近这个解：

- **值迭代**：反复做 $v_{k+1} = r + \gamma P v_k$，直到收敛。
- **策略评估**：在策略迭代中反复应用贝尔曼更新。
- **TD 学习**：用采样数据做增量更新。

这些方法本质上都是在用更可扩展的方式逼近 $(I-\gamma P)^{-1}\boldsymbol{r}$ 这个解，而不需要真的求逆。第 3 章里 DP、MC、TD 三代方法的演进，对应的就是"已知模型直接迭代 → 不知道模型用采样 → 采样也只需要一步"这条路径。

::: warning 常见误区
看到 $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$ 时，不要以为实际算法真的在计算矩阵逆。这个公式是理论闭式解，帮助理解"解存在且唯一"。实际算法是迭代的。
:::

---

## 第一道墙翻过了——但第二道墙出现了

这一篇把第 3 章学过的贝尔曼方程压缩成了矩阵形式 $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$，还得到了闭式解 $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$。不管有多少个状态，方程始终是一行。

但这里有一个默认假设被我们悄悄绕过了：**每个状态都有独立的 $v(s)$ 可以存下来**。回顾一下：

| 环境                         | 状态空间大小   | 能存下吗？ |
| ---------------------------- | -------------- | ---------- |
| 两状态小环境                 | 2           | 轻松       |
| GridWorld 10×10             | 100         | 没问题     |
| 围棋棋盘                     | ~10¹⁷⁰     | 不可能     |
| 连续状态（如机器人关节角度） | 无穷        | 不可能     |

状态一多，不仅矩阵求逆做不了，连价值表本身都存不下。闭式解虽然漂亮，但面对围棋的 $10^{170}$ 个状态，它就是一张写不完的纸。

**怎么办？** 答案是：不存每个状态的值，而是用一个函数来**近似**它。给状态提取特征，用特征和权重的点积计算价值——这就是下一篇的内容。

> **下一篇**：[E.1.3 点积、范数与函数近似](./linear-algebra-function-approx) —— 当状态太多存不下时，如何用特征向量和点积近似价值。
