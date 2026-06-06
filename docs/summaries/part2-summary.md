# Part 2: 理论与方法 — 知识总结

## 这一 Part 我们学了什么？

这五章是全书的理论核心。我们从最基础的问题"如何用数学描述决策"出发，一路走到当前工业界使用最广泛的 PPO 算法。掌握这些内容，就掌握了解读后面所有 LLM 对齐算法的钥匙。

学完这五章，你应该掌握以下核心知识点：

- **MDP 五元组** $(S, A, P, R, \gamma)$：用数学语言描述"智能体在环境中做决策"这件事。
- **价值函数与贝尔曼方程**：$V^\pi(s)$ 和 $Q^\pi(s,a)$ 分别衡量"一个状态值多少分"和"在某状态下做一个动作值多少分"。贝尔曼方程告诉我们——当前价值 = 即时奖励 + 下一步价值的折扣。
- **TD Error**：$\delta = r + \gamma V(s') - V(s)$，衡量"预测与现实的落差"，是贯穿所有 RL 算法的学习信号。
- **DQN 三大组件**：Q-Network（用神经网络逼近 Q 函数）、经验回放（打破样本相关性）、目标网络（稳定训练目标）。
- **策略梯度定理**：$\nabla_\theta J = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot G_t]$，直接对策略求梯度，天然支持连续动作。
- **Actor-Critic**：Actor 学策略，Critic 学价值，通过优势函数 $A(s,a) = Q(s,a) - V(s)$ 协作。
- **PPO 裁剪**：用 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 限制策略比率的变化幅度，防止更新过大导致训练崩溃。
- **GAE**：$\hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t+k}$，在偏差和方差之间插值。

下面让我们逐章复习这些内容。

## 第 3 章：MDP——用数学描述决策问题

### 马尔可夫决策过程

要严谨地讨论强化学习，首先要用一个数学框架来描述"智能体在环境中做决策"这件事。这个框架就是**马尔可夫决策过程**（Markov Decision Process, MDP），它由一个五元组 $(S, A, P, R, \gamma)$ 定义：

- $S$ 是状态集合。在 CartPole 中，$s = (\text{位置}, \text{速度}, \text{角度}, \text{角速度}) \in \mathbb{R}^4$。
- $A$ 是动作集合。离散情况下是 $\{a_1, a_2, \ldots\}$，连续情况下是某个实数区间。
- $P(s'|s,a)$ 是状态转移概率——在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率。"马尔可夫"的意思是：未来只依赖当前状态，和过去无关。就像下棋时，你只需要看当前局面，不需要知道之前每一步是怎么下的。
- $R(s,a)$ 是奖励函数——在状态 $s$ 执行动作 $a$ 后获得的即时奖励。
- $\gamma \in [0, 1)$ 是折扣因子——控制我们对"眼前收益"和"长远收益"的权衡。$\gamma$ 接近 1 表示重视长远，接近 0 表示只看眼前。

智能体的目标是找到一个策略 $\pi(a|s)$，使得从任意状态出发，获得的**折扣累积回报**最大化：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

为什么要有折扣因子 $\gamma$？一方面，从数学上说无穷级数需要收敛，$\gamma < 1$ 保证了 $G_t$ 是有限的。另一方面，从直觉上说，未来的奖励不如眼前的确定——就像"今天给你 100 元"比"明年给你 100 元"更有吸引力。

### 价值函数与贝尔曼方程

有了回报的定义，接下来要问：一个状态"值多少分"？一个状态-动作对"值多少分"？这就引出了两个核心概念。

**状态价值函数** $V^\pi(s)$ 衡量从状态 $s$ 出发、遵循策略 $\pi$ 的期望回报：

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s\right]$$

**动作价值函数** $Q^\pi(s, a)$ 衡量在状态 $s$ 选择动作 $a$、之后遵循策略 $\pi$ 的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s, a_t = a\right]$$

这两者之间有一个简单而深刻的关系：$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$。也就是说，状态价值等于所有动作价值的策略加权平均。

而贝尔曼方程进一步揭示了它们的递归结构——你不需要"看到未来"，只需要看一步：

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

直觉上：当前状态的价值 = 所有动作的期望（按策略加权），每个动作的价值 = 即时奖励 + 下一状态价值的折扣。这是一个自洽方程，提供了计算价值函数的基础。

### TD Error：贯穿所有 RL 算法的学习信号

在现实场景中，我们通常不知道转移概率 $P$ 和奖励函数 $R$（即不知道环境模型），只能通过与环境交互来获取样本。这时候，一个关键的学习信号出场了——**时序差分误差**（TD Error）：

$$\delta = r + \gamma V(s') - V(s)$$

TD Error 衡量的是"预测与现实的落差"。$V(s)$ 是我们预测当前状态的价值，$r + \gamma V(s')$ 是真实获得的一步奖励加上对下一步的估计。如果预测完全准确，$\delta = 0$；如果现实比预测好，$\delta > 0$，我们应该上调对 $V(s)$ 的估计。

这个简单的公式是贯穿整个 RL 的核心。从 Q-Learning 到 DQN，从 REINFORCE 到 PPO，所有算法的学习信号本质上都是 TD Error 或其变体。

### 从老虎机到 GridWorld：用代码理解 Q-Learning

在两台老虎机实验中，我们第一次体验了"探索 vs 利用"的矛盾：你想多选赢面大的那台（利用），但又怕另一台其实更好（探索）。在 4×4 GridWorld 中，我们把 Q-Learning 的完整流程跑了一遍：

```python
Q = np.zeros((n_states, n_actions))  # 初始化 Q 表

for episode in range(1000):
    state = env.reset()
    while not done:
        # ε-贪婪：以 ε 的概率随机探索，否则选当前最优
        action = epsilon_greedy(Q[state], epsilon)
        next_state, reward, done = env.step(action)
        # 用 TD Error 更新 Q 值
        td_target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (td_target - Q[state, action])
        state = next_state
```

这段代码包含了 RL 的全部要素：用 $Q(s,a)$ 表格存储价值估计，用 $\varepsilon$-贪婪平衡探索和利用，用 TD Error 引导更新。当状态空间变大（比如从 16 格棋盘变成 Atari 游戏的 $210 \times 160$ 像素），表格就放不下了——这就是 DQN 要解决的问题。

## 第 4 章：DQN——从表格到神经网络的跨越

### 维度灾难与函数逼近

CartPole 的状态虽然是连续的，但只有 4 维。然而，Atari 游戏的每一帧是 $84 \times 84 \times 4$ 的像素张量，状态空间大约有 $256^{28224}$ 种可能——这个数字比可观测宇宙中的原子还多。用表格存储每个状态的 Q 值显然不可行。

DQN 的解决方案是**用一个神经网络来近似 Q 函数**。网络输入状态 $s$，输出每个动作的 Q 值 $Q(s, a_1), Q(s, a_2), \ldots$。训练目标是最小化 TD Error 的平方：

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

这里 $\theta$ 是在线网络的参数，$\theta^-$ 是目标网络的参数。这个损失函数的意思是：让网络的预测 $Q(s, a)$ 尽量接近"一步真实的奖励 + 下一步的最大 Q 值"。

### DQN 的三大组件

光有神经网络还不够。如果直接把每次交互的数据拿来训练，样本之间高度相关（因为它们是连续时间步上的），训练会非常不稳定。DQN 引入了三个关键设计：

**经验回放（Experience Replay）** 把每一步的 $(s, a, r, s')$ 存入一个大缓冲区，训练时随机采样一个 batch。这打破了样本之间的时间相关性，让梯度更新更接近独立同分布的假设。

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

**目标网络（Target Network）** 是一份延迟更新的网络副本。每过若干步，把在线网络的参数硬拷贝给目标网络。这保证了 TD Target 在一段时间内保持稳定，避免了"追着一个移动的目标学习"的困境。想象你在练习投篮，如果篮筐每秒都在移动，你很难学会；但如果篮筐每隔一段时间才动一次，你就有时间调整。

**$\varepsilon$-贪婪探索** 随着训练推进逐步降低随机探索的概率，从早期的大范围探索过渡到后期的精细利用。

### DQN 家族

原始 DQN 在 2015 年的 Atari 论文中崭露头角，之后涌现了一系列改进。**Double DQN** 把"选动作"和"打分"解耦：用在线网络 $\theta$ 选择 $\arg\max_{a'} Q(s', a'; \theta)$，再用目标网络 $\theta^-$ 评估这个动作的价值。这解决了原始 DQN 对 Q 值过估计的问题——就像考试的出题人和阅卷人不能是同一个人。

**Dueling DQN** 把 Q 值分解为状态价值 $V(s)$ 和优势函数 $A(s,a)$ 两部分：

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$$

当某个状态下所有动作都差不多好时，$A(s,a) \approx 0$，网络只需要学好 $V(s)$。这让学习更高效。

## 第 5 章：策略梯度——直接学习策略

### 从 Value-Based 到 Policy-Based

DQN 走的是一条间接路线：先学 $Q(s,a)$，再用 $\arg\max$ 选动作。这条路有一个根本限制——只能处理**离散的、有限的**动作空间。机械臂的力矩是连续值 $[-10, 10]^6$，你没法给每种组合都算一个 Q 值。大模型生成文本就更不可能了——每一步要从几万个 token 里选。

策略梯度方法换了一条路：不学价值函数，**直接参数化策略** $\pi_\theta(a|s)$，然后优化策略参数 $\theta$ 使得期望回报最大。策略梯度定理告诉我们，目标函数 $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$ 的梯度可以这样估计：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

这个公式有一个优美的直觉：$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 指向"让动作 $a_t$ 在状态 $s_t$ 下更可能出现"的方向，$G_t$ 给出这个方向的好坏。如果 $G_t > 0$，就增加这个动作的概率；如果 $G_t < 0$，就减少。这就是 REINFORCE 算法的全部。

```python
def reinforce_update(policy, optimizer, states, actions, returns):
    log_probs = []
    for s, a in zip(states, actions):
        dist = Categorical(policy(s))
        log_probs.append(dist.log_prob(a))

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G  # 负号：梯度下降 = 最大化回报
    loss /= len(returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 基线与方差

REINFORCE 有一个致命问题：**方差太大**。如果每个 episode 的回报 $G_t$ 都在 100 左右，那即使是一个"好"动作也会得到一个很大的正值，只是相对没那么大而已。这让梯度估计非常不稳定。

解决方案是引入一个**基线** $b(s)$，把梯度公式改为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

只要 $b(s)$ 不依赖于动作 $a$，这个修改不会改变梯度的期望值（因为 $\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s)] = 0$），但会大幅降低方差。最常用的基线就是状态价值函数 $V(s)$——也就是 Critic。

## 第 6 章：Actor-Critic——用 Critic 降低方差

### Actor-Critic 架构

把 Actor（策略网络）和 Critic（价值网络）放在一起，就得到了 Actor-Critic 架构。Actor 负责选择动作，Critic 负责评估"这个动作比平均水平好多少"。这个"好多少"就是**优势函数**：

$$A(s, a) = Q(s, a) - V(s)$$

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
```

训练时，Actor 用优势函数 $A = r + \gamma V(s') - V(s)$ 来更新策略，Critic 用 TD Error 的平方来更新价值估计：

```python
# 一步交互后的更新
_, next_value = model(next_state)
td_target = reward + gamma * next_value * (1 - done)
td_error = td_target - value

actor_loss = -log_prob * td_error.detach()  # Actor：用优势更新策略
critic_loss = td_error ** 2                 # Critic：用 TD Error 更新价值
loss = actor_loss + critic_loss
```

## 第 7 章：PPO——让策略更新更稳定

### 信任域与裁剪

Actor-Critic 虽然比 REINFORCE 稳定，但策略更新仍然可能步子太大——如果一次更新把策略改得面目全非，之前收集的数据就全部作废了，训练会剧烈震荡。

PPO（Proximal Policy Optimization）用一个优雅的裁剪机制来解决这个问题。它定义了一个**策略比率**：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

当 $r_t = 1$ 时，新策略和旧策略完全一样。$r_t > 1$ 表示新策略更可能选择这个动作，$r_t < 1$ 表示更不可能。PPO 的目标是最大化：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\;\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

其中 $\varepsilon$ 通常取 0.2。当优势 $\hat{A}_t > 0$（好动作）时，PPO 最多让 $r_t$ 增加到 $1+\varepsilon$，防止过于激进地增加这个动作的概率；当 $\hat{A}_t < 0$（坏动作）时，最多让 $r_t$ 减小到 $1-\varepsilon$。这个"安全护栏"让策略更新始终在一个**信任域**内。

完整的 PPO 损失还包含两个额外项：

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 \mathcal{H}[\pi_\theta]$$

其中 $L^{\text{VF}}$ 是 Critic 的价值拟合损失（MSE），$\mathcal{H}$ 是策略的熵奖金，鼓励探索，防止策略过早收敛到次优解。

### GAE：在偏差与方差之间找到平衡

优势函数 $\hat{A}_t$ 的估计方式对 PPO 的性能至关重要。最朴素的两种极端是：用一阶 TD Error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$——偏差大但方差小；用完整轨迹回报 $G_t - V(s_t)$——无偏但方差大。**GAE**（Generalized Advantage Estimation）在两者之间插值：

$$\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}$$

其中 $\lambda \in [0, 1]$ 控制插值程度。$\lambda = 0$ 退化为一阶 TD（只看一步，偏差大），$\lambda = 1$ 退化为完整回报（看全部，方差大）。实践中通常取 $\lambda = 0.95$。

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

### 从 LunarLander 到 LLM

PPO 最初在 Gym 的 LunarLander 等传统 RL 环境上被验证，但它的真正威力在于大模型对齐。在 RLHF 中，PPO 同时管理四个模型：Actor（正在训练的语言模型）、Critic（价值网络）、Reference（冻结的原始模型，提供 KL 惩罚）、Reward Model（给回答打分的裁判）。其中 Bradley-Terry 偏好模型定义了 RM 的训练目标：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

这个框架是第 8 章的起点——而 DPO 的核心发现，正是这个框架可以绕过 RM 直接优化。

## 小结

Part 2 走过了 RL 理论的一条完整路径：MDP 提供了描述决策问题的语言 → 贝尔曼方程给出了计算价值的递归方法 → DQN 用神经网络逼近 Q 函数，解决了维度灾难 → 策略梯度直接优化策略，支持连续动作 → Actor-Critic 引入 Critic 降低方差 → PPO 用裁剪和 GAE 实现稳定训练。

这条路径上的每一个概念，在后面的 LLM 对齐中都会再次出现。理解了 PPO 的裁剪机制，才能理解 GRPO 的组内归一化；理解了 Actor-Critic 的分工，才能理解 RLHF 中四个模型的角色。

> **下一站**：[Part 3: LLM 时代](/chapter08_rlhf/intro)
