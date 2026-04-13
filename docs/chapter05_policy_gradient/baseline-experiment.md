# 5.4 基线实验与总结

前三节我们走了从直觉到理论再到架构的完整路径：赌博机实验让我们看到策略梯度能工作，策略梯度定理解释了为什么能工作，Actor-Critic 展示了怎么让它工作得更好。

但"更好"到底好了多少？基线真的能显著降低方差吗？这一节我们回到代码，用同一个赌博机环境做一个对比实验，亲眼看看有基线和没基线的差距。

## 实验设计

还是那个两臂赌博机（A: 30%，B: 70%），分别跑两个版本：

- **无基线**：标准 REINFORCE，`loss = -log_prob * G`
- **有基线**：减去运行中的平均回报，`loss = -log_prob * (G - baseline)`

为了公平起见，每个版本各跑 5 次取平均——单次运行的随机性太大，只有平均趋势才有说服力。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 策略网络（和 5.1 节相同）
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

# ==========================================
# 环境
# ==========================================
win_probs = [0.3, 0.7]
num_episodes = 500
lr = 0.01

def pull_arm(action):
    return 1.0 if random.random() < win_probs[action] else 0.0

# ==========================================
# 训练函数（支持有无基线）
# ==========================================
def train_reinforce(use_baseline=False, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    prob_history = []

    baseline = 0.0
    alpha_baseline = 0.05

    for ep in range(num_episodes):
        state = torch.tensor([1.0])
        probs = policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        reward = pull_arm(action.item())

        if use_baseline:
            # 核心区别：用"比平均好了多少"替代绝对回报
            advantage = reward - baseline
            loss = -log_prob * advantage
            # 基线跟随运行平均（指数移动平均）
            baseline = baseline + alpha_baseline * (reward - baseline)
        else:
            loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prob_history.append(policy(state)[1].item())

    return prob_history

# ==========================================
# 运行对比实验（各跑 5 次取平均）
# ==========================================
num_runs = 5
no_baseline_all = [train_reinforce(False, seed=i) for i in range(num_runs)]
with_baseline_all = [train_reinforce(True, seed=i+100) for i in range(num_runs)]

no_baseline_avg = np.mean(no_baseline_all, axis=0)
with_baseline_avg = np.mean(with_baseline_all, axis=0)
```

代码的关键差异就一处：`advantage = reward - baseline`。基线用指数移动平均来跟踪"到目前为止的平均回报"——这不是最精确的基线（最好的基线是 Critic 网络输出的 $V(s)$），但在这个无状态的赌博机场景中足够用了。

## 实验结果

把两个版本的训练曲线画在一起：

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：单次运行对比
ax1.plot(no_baseline_all[0], alpha=0.7, color='red', label='无基线')
ax1.plot(with_baseline_all[0], alpha=0.7, color='blue', label='有基线')
ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='最优 (P(B)=0.7)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('P(选择 B)')
ax1.set_title('单次运行')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：5次平均对比
ax2.plot(no_baseline_avg, color='red', linewidth=2, label='无基线 (5次平均)')
ax2.plot(with_baseline_avg, color='blue', linewidth=2, label='有基线 (5次平均)')
ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
ax2.set_xlabel('Episode')
ax2.set_ylabel('P(选择 B)')
ax2.set_title('5 次平均')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=150)
plt.show()
```

你会看到这样的画面：

```
单次运行对比                          5 次平均对比

 1.0 ┤                                1.0 ┤
     │      ╱━╮  ╱━╮ ╱━━━━          │         ╱━━━━━━━━━━━━
 0.9 ┤   ╱━╯  ╲╱  ╲╱                │    ╱━━━╱
     │  ╱ ╱╲╱                        │ ╱━╱
 0.8 ┤ ╱╱╱         有基线            │╱╱           有基线
     │╱╱╱╲╱╲╱╲╱╲                     │╱
 0.7 ┤─ ─ ─ ─ ─ ─ ─ ─ ─             ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     │╲╱ ╲╱╲ ╱╲╱╲╱ 无基线           │╲╱╲╱╲╱╲╱ 无基线
 0.6 ┤ ╲╱ ╲╱╲╱                       │ ╲╱
     │  ╲╱                            │  ╲
 0.5 ┤                                0.5 ┤
     └──────────────────              └──────────────────
      0  100 200 300 400 500           0  100 200 300 400 500
```

左图（单次运行）中，红线（无基线）在 0.5 到 0.9 之间剧烈震荡——某次采样到 A 赢了就大幅降低选 B 的概率，下次采样到 B 赢了又拉回来。蓝线（有基线）则平滑得多，稳定地爬升到高概率区间。

右图（5 次平均）进一步放大了这个差距。即使取了 5 次平均，红线仍然有明显的波动；蓝线则收敛得更快、更稳。

## 基线到底做了什么？

具体看一个例子。假设赌博机的运行平均回报（基线）是 0.5：

| 情况      | 实际发生了什么 | 无基线的梯度信号    | 有基线的梯度信号                  |
| --------- | -------------- | ------------------- | --------------------------------- |
| 摇 A 赢了 | A: 30% 概率    | reward=1 → 增加选 A | reward=1-0.5=**0.5** → 只微微增加 |
| 摇 A 输了 | A: 70% 概率    | reward=0 → 不变     | reward=0-0.5=**-0.5** → 降低选 A  |
| 摇 B 赢了 | B: 70% 概率    | reward=1 → 增加选 B | reward=1-0.5=**0.5** → 微微增加   |
| 摇 B 输了 | B: 30% 概率    | reward=0 → 不变     | reward=0-0.5=**-0.5** → 降低选 B  |

无基线版本中，"摇 A 赢了"和"摇 B 赢了"给出相同的梯度信号（都是 1）——但 B 赢的频率更高，所以 B 的概率最终会更高，只是过程非常嘈杂。有基线版本中，"摇 A 赢了"的信号被基线削弱为 0.5（"虽然赢了，但只比平均好了 0.5"），同时"摇 A 输了"会给出 -0.5 的反向信号（"比平均差了 0.5"）。因为 A 输的频率远高于 B，A 会收到更多的反向信号，B 的概率上升得更干脆。

这就是基线的核心作用：**把"赢了/输了"的二元信号变成"比平均好了多少"的连续信号**，让梯度估计更精确、更少被运气误导。

## 本章走过的路

从第 1 节到第 4 节，我们完成了一条完整的知识链：

**摇骰子赌博机**让我们亲手看到策略网络能学会"偏爱好的动作"。一个只有 Softmax 层的网络，通过 `loss = -log_prob * reward` 这一行代码，从随机选择进化到稳定选择赢率更高的 B。

**策略梯度定理**解释了为什么那一行代码能工作。数学推导的核心是：$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a|s) \cdot G_t]$——好结果强化对应动作的概率。对数导数技巧让不可计算的期望变成了可以用采样估计的梯度。

**Actor-Critic 架构**解决了 REINFORCE 的致命缺陷。引入基线 $V(s)$ 不改变梯度方向但降低方差，用 TD Error 替代 $G_t$ 使得每一步都能更新。Actor（策略网络）和 Critic（价值网络）通过优势函数 $A(s,a)$ 协作，成为后续 PPO、DPO、GRPO 的共同骨架。

**基线实验**用代码验证了上面的理论——有基线的版本收敛更快、更稳定，方差明显降低。

贯穿这整条链的暗线是**高方差问题**——从 REINFORCE 的"醉汉走路"，到基线的"减掉运气噪声"，到 Actor-Critic 的"用 Critic 稳定信号"，每一步改进都在降方差。这个主题在第 6 章 PPO 的 GAE 中还会继续深化。

## 后续章节的预告

| 概念         | 第 6 章 PPO       | 第 7 章 DPO    | 第 8 章 GRPO     |
| ------------ | ----------------- | -------------- | ---------------- |
| Actor-Critic | PPO 的骨架        | —              | 去掉 Critic      |
| 基线         | GAE 中的 $V(s)$   | 隐式基线       | 组内均值做基线   |
| 策略梯度     | Clip 限制更新幅度 | 概率比替代梯度 | 组内比较替代梯度 |
| 高方差       | GAE 控制方差      | 离线数据无方差 | 组内采样降方差   |

## 练习

1. **修改赢率差距**：把赌博机改为 A: 49%，B: 51%（差距极小）。REINFORCE 还能学会吗？有基线的版本呢？观察两者收敛速度的差异。

2. **多臂赌博机**：把赌博机改为 3 个摇臂（A: 20%，B: 50%，C: 80%），策略网络改为 3 输出。观察策略是否正确地学会了"最爱 C、最不爱 A"。

3. **基线的选择**：把基线改为固定值 0.5（而不是运行平均），效果如何？这说明了什么？

4. **连续动作空间**：假设你要训练一个机器人走路，状态是关节角度和速度，动作是每个关节的力矩（连续值）。策略网络应该输出什么？怎么从输出中采样动作？

## 参考文献

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (2000). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^3]: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. _International Conference on Machine Learning (ICML)_. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
