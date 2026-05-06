# 5.3 动手：Baseline 降方差

前两节我们走了从直觉到理论的完整路径：赌博机实验让我们看到策略梯度能工作，[策略梯度定理](./policy-gradient)解释了为什么能工作。

但 REINFORCE 有一个致命问题——方差太大。这一节我们用一个简单但有效的改进来压制噪声：**减掉一个基线**。通过对比实验，亲眼看看有基线和没基线的差距。

::: tip 本节会用到的前置知识
- [策略梯度公式](../chapter03_mdp/policy-objective) $\nabla_\theta J \propto \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$——基线要加在哪个公式里
- [状态价值 $V(s)$](../chapter03_mdp/bellman-equation)——最好的基线是什么
- [MC 方法](../chapter03_mdp/dp-mc-td)——为什么 REINFORCE 必须跑完 episode 才能更新
:::

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

这就是基线的核心作用：**把"赢了/输了"的二元信号变成"比平均好了多少"的连续信号**，让梯度估计更精确、更少被运气误导。这个"比平均好了多少"的思想，在后文中会被正式定义为[优势函数 $A(s,a)$](../chapter06_actor_critic/advantage-function)——第 6 章的核心概念。

## 本章走过的路

从第 1 节到第 3 节，我们完成了一条从直觉到理论再到改进的完整路径：

**摇骰子赌博机**让我们亲手看到策略网络能学会"偏爱好的动作"。一个只有 Softmax 层的网络，通过 `loss = -log_prob * reward` 这一行代码，从随机选择进化到稳定选择赢率更高的 B。REINFORCE 本质上就是第 3 章速览过的 [MC 方法](../chapter03_mdp/dp-mc-td)在策略空间的应用——跑完一整局（MC），用实际回报 $G_t$ 来调整策略。

**策略梯度定理**解释了为什么那一行代码能工作。数学推导的核心是：$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a|s) \cdot G_t]$——好结果强化对应动作的概率。但 REINFORCE 的致命缺陷是方差极大，同一策略跑两次，梯度估计可能天差地别。

**基线实验**给出了第一个压制噪声的方案：减掉一个基线。不要问"这趟跑了多少分"，而问"比平时好了多少"。最好的基线就是 $V(s)$——它衡量的是"平均能拿多少分"，减掉之后就只剩"因为这个动作多拿了多少分"。

贯穿这整条链的暗线是**高方差问题**。基线只解决了一半——我们用了一个简单的运行平均作为基线，但更精确的基线需要一个专门的网络来估计 $V(s)$。这个网络就是 **Critic**，它和 Actor（策略网络）一起构成了 **Actor-Critic 架构**——下一章的主题。

## 练习

1. **修改赢率差距**：把赌博机改为 A: 49%，B: 51%（差距极小）。REINFORCE 还能学会吗？有基线的版本呢？观察两者收敛速度的差异。

2. **多臂赌博机**：把赌博机改为 3 个摇臂（A: 20%，B: 50%，C: 80%），策略网络改为 3 输出。观察策略是否正确地学会了"最爱 C、最不爱 A"。

3. **基线的选择**：把基线改为固定值 0.5（而不是运行平均），效果如何？这说明了什么？

4. **连续动作空间**：假设你要训练一个机器人走路，状态是关节角度和速度，动作是每个关节的力矩（连续值）。策略网络应该输出什么？怎么从输出中采样动作？

## 参考文献

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^3]: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. _International Conference on Machine Learning (ICML)_. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
