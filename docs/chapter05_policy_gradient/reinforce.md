# 5.2 策略梯度定理与 REINFORCE

上一节说明了为什么需要 Policy-Based 方法：DQN 的 $\arg\max$ 在连续动作空间中走不通，直接学习策略 $\pi_\theta(a|s)$ 是更自然的路线。本节回答两个问题：用什么指标衡量"策略有多好"？怎么优化这个指标？

## 策略目标函数

第 3 章引入过[策略目标函数](../chapter03_mdp/policy-objective) $J(\theta)$——衡量"这个策略整体上有多好"。答案很自然：在所有可能的起点上，策略 $\pi_\theta$ 期望能累积的[折扣总奖励](../chapter03_mdp/mdp)。

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

| 符号                      | 角色     | 含义                                                 |
| ------------------------- | -------- | ---------------------------------------------------- |
| $\theta$                  | 策略参数 | 神经网络的权重——调它们就改变策略的行为               |
| $\pi_\theta$              | 策略函数 | 给定状态，输出每个动作的概率分布                     |
| $J(\theta)$               | 目标函数 | 策略的"成绩单"——参数为 $\theta$ 的策略平均能拿多少分 |
| $\mathbb{E}_{\pi\theta}$  | 期望     | 按策略 $\pi_\theta$ 行动很多很多次，取平均           |
| $\gamma^t r_t$            | 折扣奖励 | 第 $t$ 步的奖励，越远未来的奖励越"不值钱"            |

$J(\theta)$ 就是北极星——目标很简单：找到让 $J(\theta)$ 最大的参数 $\theta$。

### 具体数值：一个 3 步 episode 的 $J(\theta)$

假设一个 episode 只有 3 步，每步拿到奖励 $r_0=1$，$r_1=2$，$r_2=3$，折扣因子 $\gamma=0.9$。这条轨迹的折扣总回报为

$$
\begin{aligned}
\sum_{t=0}^{2} \gamma^t r_t &= \gamma^0 r_0 + \gamma^1 r_1 + \gamma^2 r_2 \\
&= 0.9^0 \times 1 + 0.9^1 \times 2 + 0.9^2 \times 3 \\
&= 1 \times 1 + 0.9 \times 2 + 0.81 \times 3 \\
&= 1 + 1.8 + 2.43 \\
&= 5.23.
\end{aligned}
$$

这是**一条**轨迹的回报。$J(\theta)$ 是所有可能轨迹回报的期望——按策略 $\pi_\theta$ 跑无数次，取平均。不同的策略 $\pi_\theta$ 产生不同的轨迹分布，因此 $J(\theta)$ 不同。假设策略甲倾向于选择高奖励路径，策略乙倾向于选择低奖励路径：

| 策略 | 可能轨迹的平均回报 | $J(\theta)$ |
| ---- | ------------------ | ----------- |
| 甲   | 平均拿 $5.23$ 左右 | 较大        |
| 乙   | 平均拿 $2.10$ 左右 | 较小        |

优化目标就是找到让 $J(\theta)$ 最大（即平均回报最高）的参数 $\theta$。

## 梯度上升

怎么让 $J(\theta)$ 变大？深度学习里最经典的招数：沿着梯度方向走。

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

| 符号                      | 角色     | 含义                                       |
| ------------------------- | -------- | ------------------------------------------ |
| $\nabla_\theta J(\theta)$ | 梯度     | "参数往哪个方向调，能让策略的成绩提升最多" |
| $\alpha$                  | 学习率   | "每一步走多大"——太大就震荡，太小就慢       |
| $+$                       | 梯度上升 | 注意是加号——我们要最大化，不是最小化       |

### 具体数值：一次参数更新

假设参数 $\theta = [0.5,\ 0.3,\ -0.1]$，梯度 $\nabla_\theta J(\theta) = [0.1,\ -0.2,\ 0.05]$，学习率 $\alpha = 0.01$。更新过程逐分量写出：

$$
\begin{aligned}
\theta_0 &\leftarrow 0.5 + 0.01 \times 0.1 = 0.5 + 0.001 = 0.501, \\
\theta_1 &\leftarrow 0.3 + 0.01 \times (-0.2) = 0.3 - 0.002 = 0.298, \\
\theta_2 &\leftarrow -0.1 + 0.01 \times 0.05 = -0.1 + 0.0005 = -0.0995.
\end{aligned}
$$

更新后的参数 $\theta = [0.501,\ 0.298,\ -0.0995]$。每个分量都朝着梯度指示的方向移动了一小步。$\alpha$ 控制步长：$\alpha=0.01$ 时每步只移动千分之几，但累积很多轮后效果显著。

但 $\nabla_\theta J(\theta)$ 怎么算？目标函数里有一个期望 $\mathbb{E}$——理论上要求把所有可能的轨迹都跑一遍然后取平均。现实中可能的轨迹数量是天文数字，不可能全部跑遍。就好比想知道全校学生的平均身高——不可能量遍每一个人，但可以随机抽 100 个人来估计。抽 100 个人算出来的样本均值，就是对真实均值的**抽样估计**。策略梯度用的也是同一个思路：跑几条轨迹，用这几条轨迹的梯度平均，来估计真实的 $\nabla_\theta J(\theta)$。

## 策略梯度定理

这就是策略梯度定理出场的地方。1992 年，Williams 在 REINFORCE 论文中证明：那个看似无法计算的梯度 $\nabla_\theta J(\theta)$，可以被转化为一个可以用采样来估计的形式 [^1]。后来 Sutton 等人在 2000 年进一步推广和系统化了这一结果 [^2]。

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

逐项认识：

| 符号                                        | 角色           | 含义                                                      |
| ------------------------------------------- | -------------- | --------------------------------------------------------- |
| $\nabla_\theta$                             | 求梯度         | "参数该往哪调"                                            |
| $\log \pi_\theta(a_t \| s_t)$               | 对数概率       | 在状态 $s_t$ 下，策略选择动作 $a_t$ 的对数概率            |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | 对数概率的梯度 | "参数怎么调能改变这个动作被选中的概率"                    |
| $G_t$                                       | 累积回报       | 从时刻 $t$ 到结束的总奖励——"做了这个动作后最终拿了多少分" |
| 外层 $\mathbb{E}$                           | 期望           | "跑很多次取平均"——用采样来近似                            |

翻译成一句话：**如果一个动作导致了好的结果（$G_t$ 大），就增加再做这个动作的概率；如果导致了坏的结果（$G_t$ 小），就降低它的概率。**

### 具体数值：赌博机场景的梯度计算

用赌博机场景走一遍。两臂赌博机：A 臂赢率 30%，B 臂赢率 70%。策略只有一个参数 $\theta$（不妨认为 $\theta$ 越大，选 B 的概率越高），当前 $\pi_\theta(B|s) = 0.7$，$\pi_\theta(A|s) = 0.3$。

**轨迹 1：选了 B，奖励 = 1.0**

第一步，算对数概率：

$$
\log \pi_\theta(B|s) = \log(0.7) = -0.357.
$$

第二步，算对数概率的梯度。假设当前参数下 $\nabla_\theta \log \pi_\theta(B|s) = 0.5$（这个数值取决于具体的参数化方式，这里取一个具体值）。

第三步，乘以回报。赌博机只有一步，所以 $G_0 = r_0 = 1.0$：

$$
\nabla_\theta \log \pi_\theta(B|s) \cdot G_0 = 0.5 \times 1.0 = 0.5.
$$

梯度为正，参数更新 $\theta \leftarrow \theta + \alpha \times 0.5$。参数增大，$\pi_\theta(B|s)$ 随之增大——策略学会了"选 B 结果好，以后多选 B"。

**轨迹 2：选了 A，奖励 = 0**

$$
\log \pi_\theta(A|s) = \log(0.3) = -1.204.
$$

假设 $\nabla_\theta \log \pi_\theta(A|s) = -0.5$（选 A 和选 B 的梯度方向相反）。乘以回报 $G_0 = 0$：

$$
\nabla_\theta \log \pi_\theta(A|s) \cdot G_0 = (-0.5) \times 0 = 0.
$$

梯度为零，参数不更新。选了 A 但奖励为 0，策略既不鼓励也不惩罚这个动作。

**轨迹 3：选了 A，奖励 = 0.5**

$$
\nabla_\theta \log \pi_\theta(A|s) \cdot G_0 = (-0.5) \times 0.5 = -0.25.
$$

梯度为负，参数更新 $\theta \leftarrow \theta + \alpha \times (-0.25)$。参数减小，$\pi_\theta(B|s)$ 随之减小、$\pi_\theta(A|s)$ 随之增大——策略学到了"A 这次拿到了一些奖励，可以稍微多选 A"。但因为 A 的平均奖励远低于 B，多次采样后 B 的正梯度会累积压过 A，最终策略收敛到偏向 B。

### 具体数值：3 步 episode 的梯度计算

考虑一个 CartPole 式的场景：3 步 episode，$\gamma=0.9$。

| 步骤 | 状态  | 动作       | 奖励      |
| ---- | ----- | ---------- | --------- |
| 0    | $s_0$ | $a_0$ = 右 | $r_0 = 1$ |
| 1    | $s_1$ | $a_1$ = 左 | $r_1 = 2$ |
| 2    | $s_2$ | $a_2$ = 右 | $r_2 = 3$ |

先计算每一步的累积回报 $G_t$。$G_2$ 只包含最后一步：

$$
G_2 = r_2 = 3.
$$

$G_1$ 从第 1 步累加到结束：

$$
G_1 = r_1 + \gamma r_2 = 2 + 0.9 \times 3 = 2 + 2.7 = 4.7.
$$

$G_0$ 从第 0 步累加到结束：

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23.
$$

假设每一步的对数概率梯度值为：

| 步骤 | $\log \pi_\theta(a_t \| s_t)$               | $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | $G_t$  |
| ---- | ------------------------------------------- | ------------------------------------------- | ------ |
| 0    | $\log \pi_\theta(\text{右}\mid s_0) = -0.4$ | $[0.3, -0.1]$                               | $5.23$ |
| 1    | $\log \pi_\theta(\text{左}\mid s_1) = -0.7$ | $[-0.2, 0.4]$                               | $4.7$  |
| 2    | $\log \pi_\theta(\text{右}\mid s_2) = -0.3$ | $[0.1, 0.2]$                                | $3$    |

每一步贡献的梯度项：

$$
\begin{aligned}
\text{步骤 0：} \quad \nabla_\theta \log \pi_\theta(a_0|s_0) \cdot G_0 &= [0.3, -0.1] \times 5.23 = [1.569, -0.523], \\
\text{步骤 1：} \quad \nabla_\theta \log \pi_\theta(a_1|s_1) \cdot G_1 &= [-0.2, 0.4] \times 4.7 = [-0.94, 1.88], \\
\text{步骤 2：} \quad \nabla_\theta \log \pi_\theta(a_2|s_2) \cdot G_2 &= [0.1, 0.2] \times 3 = [0.3, 0.6].
\end{aligned}
$$

这一条轨迹提供的梯度估计为三者之和：

$$
\begin{aligned}
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t &= [1.569, -0.523] + [-0.94, 1.88] + [0.3, 0.6] \\
&= [1.569 - 0.94 + 0.3,\ -0.523 + 1.88 + 0.6] \\
&= [0.929,\ 1.957].
\end{aligned}
$$

如果 $\alpha = 0.01$，参数更新为 $\theta \leftarrow \theta + 0.01 \times [0.929,\ 1.957]$。第二步的贡献最大（$G_1=4.7$ 且梯度分量 $0.4$ 较大），说明"在 $s_1$ 选左"这个决策对最终回报的贡献突出，参数会朝"增大在 $s_1$ 选左的概率"方向移动。

### 对数导数技巧

为什么不直接写成 $\nabla_\theta \pi_\theta(a_t|s_t) \cdot G_t$，非要多一个 $\log$？

这是一个数学技巧，叫做**对数导数技巧**（Log-Derivative Trick）。根据链式法则：

$$\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$$

这个"除以 $\pi$"的操作恰好抵消了期望计算中隐含的 $\pi$ 因子，让整个公式变得干净且可计算。从工程角度看，概率 $\pi$ 在 $(0, 1)$ 之间，直接对概率求梯度可能产生极小的数值，影响训练稳定性。$\log$ 把 $(0, 1)$ 映射到 $(-\infty, 0)$，梯度数值更稳定。

### 具体数值：对数导数技巧的数值效果

取一个具体例子：$\pi_\theta(a|s) = 0.7$，假设参数微小扰动后 $\pi_\theta(a|s)$ 变为 $0.71$，则

$$
\nabla_\theta \pi_\theta(a|s) \approx 0.71 - 0.7 = 0.01.
$$

直接用概率梯度：

$$
\nabla_\theta \pi_\theta(a|s) = 0.01.
$$

对数导数技巧：

$$
\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \frac{0.01}{0.7} = 0.0143.
$$

再考虑另一个动作 $\pi_\theta(a'|s) = 0.05$，同样扰动后变为 $0.06$：

$$
\nabla_\theta \pi_\theta(a'|s) = 0.01, \quad \nabla_\theta \log \pi_\theta(a'|s) = \frac{0.01}{0.05} = 0.2.
$$

两个动作的概率梯度相同（都是 $0.01$），但概率小的动作对数导数梯度是概率大的动作的 $0.2/0.0143 \approx 14$ 倍。这符合直觉：把一个 $5\%$ 概率的动作提升到 $6\%$，相对增幅是 $20\%$；把一个 $70\%$ 概率的动作提升到 $71\%$，相对增幅只有 $1.4\%$。除以 $\pi$ 相当于把绝对变化转化为相对变化，让不同概率量级的动作在梯度空间里有可比的尺度。

<details>
<summary>数学推导：从目标函数到策略梯度定理</summary>

目标函数的梯度需要对轨迹的概率分布求导：

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \sum_t r_t(\tau)$$

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是一条轨迹，$P(\tau; \theta)$ 是策略产生轨迹 $\tau$ 的概率。梯度只能作用于 $P(\tau; \theta)$（奖励不依赖参数）：

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \sum_t r_t(\tau)$$

关键一步：利用恒等式 $\nabla_\theta P = P \cdot \nabla_\theta \log P$：

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \left( \nabla_\theta \log P(\tau; \theta) \right) \sum_t r_t(\tau)$$

轨迹概率可以分解为：$P(\tau; \theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$。取对数后对 $\theta$ 求梯度，环境转移概率 $P(s'|s,a)$ 不依赖于 $\theta$，所以只剩策略部分：

$$\nabla_\theta \log P(\tau; \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

代回期望中，就得到了策略梯度定理。这个过程最妙的地方在于：环境动力学（状态转移概率）在求导时被消掉了。这意味着策略梯度**不需要知道环境的模型**——这是它比动态规划方法灵活得多的根本原因。

</details>

## REINFORCE 算法

策略梯度定理告诉了我们梯度的形式。**REINFORCE** 就是这个定理最朴素的实现——用[蒙特卡洛采样](../chapter03_mdp/dp-mc-td)来估计期望。算法流程：

1. 用当前策略 $\pi_\theta$ 跑完一个完整的 episode，记录每一步的状态、动作和奖励
2. 对每一步，计算从那一步到 episode 结束的累积回报 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
3. 用采样来估计梯度：$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$
4. 沿梯度方向更新参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J$

在 PyTorch 中，这可以写成一行：

```python
loss = -log_prob * G_t  # 负号因为 PyTorch 默认做梯度下降（最小化），而我们要梯度上升（最大化）
```

完整的多步版本：

```python
# REINFORCE 核心（多步版本）
for t in range(len(rewards)):
    G_t = sum(gamma ** k * rewards[t + k] for k in range(len(rewards) - t))
    loss += -log_probs[t] * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 具体数值：一个完整 episode 的 REINFORCE 更新

沿用前面的 3 步 episode，$\gamma=0.9$：

| 步骤 | 状态  | 动作 | 奖励    | $\log \pi_\theta(a_t \| s_t)$ |
| ---- | ----- | ---- | ------- | ----------------------------- |
| 0    | $s_0$ | 右   | $r_0=1$ | $-0.4$                        |
| 1    | $s_1$ | 左   | $r_1=2$ | $-0.7$                        |
| 2    | $s_2$ | 右   | $r_2=3$ | $-0.3$                        |

**第一步：计算每步的累积回报 $G_t$。**

$G_2$ 从第 2 步到结束（只有 1 步）：

$$
G_2 = r_2 = 3.
$$

$G_1$ 从第 1 步到结束（2 步）：

$$
G_1 = r_1 + \gamma r_2 = 2 + 0.9 \times 3 = 2 + 2.7 = 4.7.
$$

$G_0$ 从第 0 步到结束（3 步）：

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23.
$$

汇总：

| 步骤 | $r_t$ | $G_t$ |
| ---- | ----- | ----- |
| 0    | 1     | 5.23  |
| 1    | 2     | 4.7   |
| 2    | 3     | 3     |

**第二步：计算梯度估计。**

每步的梯度项为 $\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$。在 PyTorch 中，`log_prob` 就是 $\log \pi_\theta(a_t|s_t)$，自动微分会处理梯度。这里只看标量 loss 的构成：

$$
\begin{aligned}
\text{loss} &= -\sum_{t=0}^{2} \log \pi_\theta(a_t|s_t) \cdot G_t \\
&= -(\log \pi_\theta(\text{右}|s_0) \cdot G_0 + \log \pi_\theta(\text{左}|s_1) \cdot G_1 + \log \pi_\theta(\text{右}|s_2) \cdot G_2) \\
&= -((-0.4) \times 5.23 + (-0.7) \times 4.7 + (-0.3) \times 3) \\
&= -(−2.092 + (−3.29) + (−0.9)) \\
&= -(-6.282) \\
&= 6.282.
\end{aligned}
$$

**第三步：反向传播与参数更新。**

`loss.backward()` 计算的是 $\nabla_\theta \text{loss}$。因为 $\text{loss} = -\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$，所以

$$
\nabla_\theta \text{loss} = -\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t.
$$

`optimizer.step()` 执行 $\theta \leftarrow \theta - \alpha \cdot \nabla_\theta \text{loss}$（PyTorch 默认梯度下降），负负得正：

$$
\theta \leftarrow \theta - \alpha \cdot \left(-\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right) = \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t.
$$

这正是梯度上升。这一轮更新让参数朝着"增大高回报动作概率"的方向移动。

### 一个最简例子：赌博机

在深入 CartPole 之前，先用一个极简场景理解 `loss = -log_prob * reward` 在做什么。

想象一台赌博机，两个摇臂：A 赢率 30%，B 赢率 70%。策略网络只有一个 Softmax 层，输出选择 A 和 B 的概率。训练的核心代码：

```python
probs = policy(state)
dist = torch.distributions.Categorical(probs)
action = dist.sample()          # 按概率随机选一个动作
log_prob = dist.log_prob(action)  # log π(a|s)

reward = pull_arm(action.item())  # 执行动作

loss = -log_prob * reward         # REINFORCE 核心
```

运行 300 个 episode 后，选择 B 的概率会从 0.5 附近逐渐爬升到 0.85–0.95——策略学会了"偏爱赢率高的动作"。但曲线不是平滑上升的，而是充满锯齿和波动。把学习率从 0.01 改成 0.1，策略会在 A 和 B 之间剧烈摇摆。

这就是 REINFORCE 的核心痛点——**高方差**。

## REINFORCE 的方差问题

$G_t$ 是从时刻 $t$ 到 episode 结束的累积回报——它包含了这段路径上的所有随机性。同一个动作，不同的采样轨迹可能给出截然不同的 $G_t$：

| 情况   | 实际发生了什么         | $G_t$ |
| ------ | ---------------------- | ----- |
| 好运气 | 后续每步都恰好拿了高分 | 很大  |
| 坏运气 | 后续每步都恰好拿了低分 | 很小  |

策略梯度用 $G_t$ 来判断"这个动作好不好"——但 $G_t$ 的波动意味着，**同一个好的动作可能因为运气差而被惩罚，同一个差的动作可能因为运气好而被奖励**。这就像用一次考试的成绩来判断一个学生的水平——考砸了不代表学得差，可能只是那天状态不好。

### 具体数值：两条轨迹的梯度信号对比

回到赌博机场景：$\pi_\theta(B|s) = 0.7$，$\pi_\theta(A|s) = 0.3$。两次都选了 B（同一个动作），但后续运气不同。

**Episode 1：选 B，奖励 1.0（好结果）**

$$
\text{梯度项} = \nabla_\theta \log \pi_\theta(B|s) \times 1.0.
$$

梯度推高 $\pi_\theta(B|s)$。

**Episode 2：选 B，奖励 0.0（坏结果，运气差）**

$$
\text{梯度项} = \nabla_\theta \log \pi_\theta(B|s) \times 0.0 = 0.
$$

梯度信号为零，参数不变。

再看一个多步的例子。同一个 3 步 episode 结构，$\gamma=0.9$，策略在 $s_0$ 选了同一个动作"右"：

| Episode | $r_0$ | $r_1$ | $r_2$ | $G_0$                   | 梯度信号方向 |
| ------- | ----- | ----- | ----- | ----------------------- | ------------ |
| 1       | 1     | 2     | 3     | $1 + 1.8 + 2.43 = 5.23$ | 强正向       |
| 2       | 1     | 0     | 0     | $1 + 0 + 0 = 1$         | 弱正向       |

两次都在 $s_0$ 做了同样的动作"右"，$G_0$ 却相差 4 倍以上。Episode 1 会把 $\pi(\text{右}|s_0)$ 大幅推高，Episode 2 只会微弱推高。问题出在 $G_0$ 不只反映了"在 $s_0$ 选右好不好"，还包含了 $r_1$ 和 $r_2$ 的随机性——后面两步的奖励不是当前动作能控制的，但它们全部被算进了 $G_0$。

在赌博机实验中，这表现为训练曲线的锯齿和震荡。在更复杂的环境中（比如 CartPole），高方差会让训练更加不稳定——有时候策略学得很好，突然又被一次坏运气带偏。

## 离散与连续动作空间

本章实验用的是离散动作空间（选 A 或选 B、CartPole 左/右），但策略梯度定理对连续动作空间同样成立：

|                   | 离散动作空间              | 连续动作空间                                 |
| ----------------- | ------------------------- | -------------------------------------------- |
| 例子              | CartPole 左/右、LLM 选词  | 机器人关节角度、方向盘转角                   |
| 输出层            | Softmax（每个动作的概率） | 高斯分布参数（均值 $\mu$ 和标准差 $\sigma$） |
| 采样方式          | 按 Softmax 概率随机选     | 从 $\mathcal{N}(\mu, \sigma^2)$ 采样         |
| $\log \pi$ 的计算 | `log_softmax`             | 高斯分布的对数密度公式                       |

同一个策略梯度公式，换一个输出层就能从"左/右"切换到"连续力矩"。这就是策略梯度方法比 Value-Based 方法灵活的地方：DQN 的 $\arg\max$ 在连续空间中根本算不出来，而策略梯度直接对概率密度求梯度，天然适用于连续空间。

<details>
<summary>思考题：REINFORCE 和 Q-Learning 的更新有什么本质区别？</summary>

Q-Learning 更新的是价值函数 $Q(s,a)$（"这个动作值多少分"），策略是通过 $\arg\max Q$ 隐式得到的。REINFORCE 直接更新策略参数 $\theta$，跳过了 Q 值这一步。

这个区别带来两个关键后果：Q-Learning 是 off-policy 的（可以用旧数据反复训练），REINFORCE 是 on-policy 的（必须用当前策略的新数据）；Q-Learning 只能处理离散动作（需要遍历所有动作取 max），REINFORCE 可以处理连续动作（直接对概率密度求梯度）。

</details>

REINFORCE 能工作，但高方差让它几乎不可用。好在策略梯度定理有一个奇妙的性质：可以在梯度估计中减去一个不依赖于动作的"基线"，既不改变梯度的期望方向，又能大幅降低方差。这将在第 5.4 节展开。下一节先在 CartPole 上动手跑一遍 REINFORCE：[动手：CartPole 实战](./cartpole)。

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
