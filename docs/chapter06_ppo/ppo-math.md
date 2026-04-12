# 6.2 PPO 数学推导——从策略梯度到裁剪代理目标

上一节我们用 PPO 训练了月球着陆器，看到了训练曲线和关键指标。但 PPO 的公式是怎么来的？为什么它比朴素策略梯度更稳定？这一节我们从头推导 PPO 的完整数学原理，把第 5 章的策略梯度、重要性采样、裁剪机制串成一条完整的链条。

推导路线图：

```
策略梯度 → 带基线的策略梯度 → 代理目标（Surrogate Objective）→ TRPO（KL 约束）→ PPO-Clip（裁剪）
```

## 第一步：策略梯度回顾

第 5 章我们推导了策略梯度定理。策略 $\pi_\theta$ 的目标函数是期望累计奖励：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

策略梯度定理告诉我们，目标函数对参数的梯度可以写成：

$$\nabla_\theta J(\theta) = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \Psi_t \right]$$

其中 $\Psi_t$ 可以是累计回报 $G_t$、基线校正后的优势 $A_t$、或 TD Error $\delta_t$。选择 $\Psi_t = A_t$ 就是最常用的形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]$$

这就是 Actor-Critic 架构中 Actor 的更新方向——Critic 提供 $A_t$，Actor 沿着梯度调整 $\theta$。

**朴素策略梯度的致命问题**：这个梯度估计的方差很大。一次基于 mini-batch 的梯度更新可能导致策略发生大幅变化，而策略更新是不可逆的——一旦参数变了，之前收集的数据就不再适用。

## 第二步：重要性采样与代理目标

朴素策略梯度是**在线**（on-policy）的——必须用当前策略 $\pi_\theta$ 收集数据。能不能用旧策略 $\pi_{\theta_{\text{old}}}$ 收集的数据来更新新策略？可以，靠**重要性采样**。

核心恒等式：对于任意函数 $f$，

$$\mathbb{E}_{a \sim \pi_\theta} [f(a)] = \mathbb{E}_{a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \cdot f(a) \right]$$

定义**策略比率**（Policy Ratio）：

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)}$$

把重要性采样应用到策略梯度目标上，得到**代理目标**（Surrogate Objective）：

$$L^{\text{IS}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right]$$

这个目标的一阶梯度等于朴素策略梯度：

$$\nabla_\theta L^{\text{IS}}(\theta) \bigg|_{\theta = \theta_{\text{old}}} = \nabla_\theta J(\theta)$$

也就是说，在 $\theta = \theta_{\text{old}}$ 附近，代理目标和真实目标梯度一致。但**只要 $\theta$ 偏离 $\theta_{\text{old}}$，两者就会分叉**。偏离越远，代理目标就越不可靠。

## 第三步：为什么要限制更新幅度？

代理目标 $L^{\text{IS}}(\theta) = \mathbb{E}_t [r_t(\theta) \cdot A_t]$ 本身没有上限。如果某个动作的优势 $A_t > 0$，最大化这个目标会无限制地推高 $r_t(\theta)$，即无限制地增大 $\pi_\theta(a_t|s_t)$。

更形式化地说，策略比率 $r_t(\theta)$ 可以变得任意大——代理目标只关心"让好动作的概率尽量高"，不关心"高到什么程度"。一旦 $r_t$ 远离 1，重要性采样的方差急剧增大，估计变得不可靠。

这就是 TRPO 和 PPO 要解决的核心问题：**找到一种方式，在优化代理目标的同时，限制策略的变化幅度。**

## 第四步：TRPO——KL 散度约束

TRPO（Trust Region Policy Optimization, Schulman et al. 2015）的做法是在约束条件下优化代理目标：

$$\max_\theta \; L^{\text{IS}}(\theta) \quad \text{s.t.} \quad \bar{D}_{\text{KL}}(\theta_{\text{old}}, \theta) \leq \delta$$

其中 $\bar{D}_{\text{KL}}$ 是平均 KL 散度，$\delta$ 通常取 0.01。这是一个**约束优化**问题。

TRPO 的求解方法是对代理目标做一阶泰勒展开、对 KL 约束做二阶泰勒展开，然后用共轭梯度法求解。理论上很漂亮，但工程上需要计算 Fisher 信息矩阵与向量的乘积（Hessian-vector product），实现复杂且计算昂贵。

## 第五步：PPO-Clip——裁剪代理目标

PPO（Proximal Policy Optimization, Schulman et al. 2017）的核心思想：**不用约束优化，直接修改目标函数，让"过大的更新"不再被奖励。**

PPO-Clip 的目标函数：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot A_t, \; \overline{r}_t(\theta) \cdot A_t \right) \right]$$

其中裁剪后的比率：

$$\overline{r}_t(\theta) = \text{clip}(r_t(\theta), \; 1 - \varepsilon, \; 1 + \varepsilon)$$

$\varepsilon$ 是超参数，通常取 0.1 或 0.2。

### 分情况理解裁剪效果

**情况一：$A_t > 0$（好动作，应该增加概率）**

| $r_t$ 的范围               | 未裁剪项 $r_t \cdot A_t$ | 裁剪项 $\overline{r}_t \cdot A_t$   | $\min$ 取哪个    |
| -------------------------- | ------------------------ | ----------------------------------- | ---------------- |
| $r_t \leq 1 + \varepsilon$ | $r_t \cdot A_t$          | $r_t \cdot A_t$                     | 相等，正常优化   |
| $r_t > 1 + \varepsilon$    | $r_t \cdot A_t$（更大）  | $(1+\varepsilon) \cdot A_t$（常数） | 裁剪项，梯度为零 |

好动作的概率可以增加，但最多增到 $1 + \varepsilon$ 倍。超过之后目标函数"变平"——不再提供继续增大的动力。

**情况二：$A_t < 0$（坏动作，应该降低概率）**

| $r_t$ 的范围               | 未裁剪项 $r_t \cdot A_t$ | 裁剪项 $\overline{r}_t \cdot A_t$   | $\min$ 取哪个              |
| -------------------------- | ------------------------ | ----------------------------------- | -------------------------- |
| $r_t \geq 1 - \varepsilon$ | $r_t \cdot A_t$          | $r_t \cdot A_t$                     | 相等，正常优化             |
| $r_t < 1 - \varepsilon$    | $r_t \cdot A_t$（更负）  | $(1-\varepsilon) \cdot A_t$（常数） | 未裁剪项（更小），梯度为零 |

坏动作的概率可以降低，但最多降到 $1 - \varepsilon$ 倍。超过之后同样"变平"。

**核心直觉**：$\min$ 操作确保目标函数是代理目标的**下界**（lower bound）。当策略偏离太远时，下界变平，更新自动停止。

```python
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PPO-Clip 目标函数的几何直觉
# ==========================================
epsilon = 0.2
r = np.linspace(0.0, 2.0, 500)

def clip_objective(r, A, eps=0.2):
    r_clipped = np.clip(r, 1 - eps, 1 + eps)
    return np.minimum(r * A, r_clipped * A)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (A_val, title) in zip(axes, [
    (1.0, "A > 0 (好动作)"),
    (-1.0, "A < 0 (坏动作)"),
    (0.0, "A = 0 (中性动作)")
]):
    obj = clip_objective(r, A_val)
    ax.plot(r, r * A_val, 'b--', alpha=0.4, label='未裁剪 r·A')
    ax.plot(r, obj, 'r-', linewidth=2, label='PPO-Clip min(...)')
    ax.axvspan(1 - epsilon, 1 + epsilon, alpha=0.1, color='green', label='安全区间')
    ax.set_title(title)
    ax.set_xlabel('策略比率 r_t(θ)')
    ax.set_ylabel('目标值')
    ax.legend(fontsize=8)

plt.suptitle('PPO-Clip 目标函数的三种情况 (ε=0.2)', fontsize=13)
plt.tight_layout()
plt.savefig("ppo_clip_three_cases.png", dpi=150)
print("可视化已保存")
```

## 第六步：PPO 的完整损失函数

实际训练中，PPO 的损失函数不只是裁剪代理目标，而是**三项之和**：

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1 \cdot L^{\text{VF}}(\theta) + c_2 \cdot H[\pi_\theta]$$

### 1. 策略损失（Policy Loss）$L^{\text{CLIP}}$

就是上面推导的裁剪代理目标。注意前面要加负号（因为我们要最小化总损失）：

$$L^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot A_t, \; \overline{r}_t(\theta) \cdot A_t \right) \right]$$

### 2. 价值函数损失（Value Function Loss）$L^{\text{VF}}$

Critic 需要准确估计状态价值。价值损失是 Critic 的预测值 $V_\theta(s_t)$ 与目标回报 $V_t^{\text{targ}}$ 之间的均方误差：

$$L^{\text{VF}}(\theta) = \mathbb{E}_t \left[ \left( V_\theta(s_t) - V_t^{\text{targ}} \right)^2 \right]$$

其中 $V_t^{\text{targ}}$ 由 GAE 计算得到（下一节详细推导 GAE）。

### 3. 熵奖金（Entropy Bonus）$H[\pi_\theta]$

策略熵鼓励探索，防止策略过早收敛到确定性策略：

$$H[\pi_\theta] = -\mathbb{E}_t \left[ \sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]$$

熵越高，策略越"犹豫"（越均匀），探索越充分。系数 $c_2$ 通常取 0.01。

### 超参数总结

| 符号          | 名称         | 典型值  | 作用                       |
| ------------- | ------------ | ------- | -------------------------- |
| $\varepsilon$ | 裁剪范围     | 0.1–0.2 | 限制策略比率的变化范围     |
| $c_1$         | 价值损失系数 | 0.5     | 平衡策略更新和价值函数拟合 |
| $c_2$         | 熵奖金系数   | 0.01    | 鼓励探索                   |
| $\gamma$      | 折扣因子     | 0.99    | 未来奖励的衰减速度         |
| $\lambda$     | GAE 参数     | 0.95    | 优势估计中偏差-方差的权衡  |
| $T$           | rollout 长度 | 2048    | 每次收集多少步数据         |
| $K$           | epoch 数     | 10      | 同一批数据更新几轮         |

## 第七步：PPO 完整算法

把所有组件组装起来，PPO 的训练循环如下：

```
循环直到收敛:
    1. 用当前策略 π_θ 收集 T 步数据 {(s_t, a_t, r_t)}_{t=1}^{T}
    2. 用 GAE 计算优势估计 Â_t 和目标回报 V_t^targ
    3. 重复 K 轮:
        对每个 mini-batch:
            a. 计算策略比率 r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
            b. 计算 L^CLIP = -min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)
            c. 计算 L^VF = (V_θ(s_t) - V_t^targ)²
            d. 计算熵 H[π_θ]
            e. 总损失 L = L^CLIP + c_1 · L^VF - c_2 · H
            f. 梯度下降更新 θ
    4. 更新旧策略: π_old ← π_θ
```

几个关键设计决策的直觉：

- **重复利用数据 K 轮**：收集一次数据很贵（需要跑环境），所以用同一批数据更新多次。裁剪机制保证多轮更新不会让策略跑偏。
- **Mini-batch 更新**：把 $T$ 步数据分成若干 mini-batch，每个 mini-batch 独立计算梯度，提高训练效率。
- **每轮重新计算 r_t**：虽然用的是同一批数据，但每轮更新后 $\theta$ 变了，$r_t$ 也变了，裁剪会动态生效。

## 与 TRPO 的理论对比

| 维度         | TRPO                 | PPO-Clip               |
| ------------ | -------------------- | ---------------------- |
| 约束方式     | 硬约束（KL ≤ δ）     | 软约束（裁剪目标函数） |
| 优化方法     | 约束优化 + 共轭梯度  | 标准梯度下降           |
| 需要二阶信息 | 是（Fisher 矩阵）    | 否                     |
| 实现难度     | 高                   | 低                     |
| 理论保证     | 保证单调改进         | 经验上近似单调改进     |
| 大规模可行性 | 差（70B 模型不可行） | 好                     |

PPO 放弃了 TRPO 严格的理论保证，换来了工程上的简洁和可扩展性。在几乎所有实际任务中，PPO 的表现与 TRPO 相当甚至更好——因为 TRPO 的二阶近似本身也有误差，"精确求解不完美的近似"不一定比"直接裁剪"更好。

<details>
<summary>推导补充：PPO-Penalty 变体</summary>

PPO 论文中实际上提出了两种变体。除了 PPO-Clip，还有一种 **PPO-Penalty**（也叫 PPO-KL），它把 KL 约束直接加入目标函数作为惩罚项：

$$L^{\text{KL}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t - \beta \cdot D_{\text{KL}}(\pi_{\text{old}}, \pi_\theta) \right]$$

$\beta$ 是自适应系数：如果当前 KL 太大，就增大 $\beta$ 加强惩罚；如果 KL 太小，就减小 $\beta$ 放松约束。

PPO-Penalty 在某些场景下效果更好（特别是需要精确控制策略变化的场景），但实现比 PPO-Clip 复杂，且多了一个需要调节的自适应机制。实践中 PPO-Clip 更常用。

</details>

---

到这一步，你已经看到了 PPO 的完整数学图景——从策略梯度到代理目标，从 TRPO 的 KL 约束到 PPO 的裁剪机制，再到三项损失函数的完整组合。接下来的两节会分别深入两个关键细节：

- **裁剪机制的直觉和实验** → [信任域与裁剪](./trust-region-clipping)
- **GAE 的推导和 LLM 对齐中的应用** → [GAE、奖励模型与 LLM 对齐](./gae-reward-model)
