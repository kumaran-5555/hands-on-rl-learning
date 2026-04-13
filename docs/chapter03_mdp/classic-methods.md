# 经典方法速通：从理论到实践的桥梁

在前面的几个小节中，我们一步步搭建了 RL 的理论大厦：MDP 五元组是地基，价值函数是框架，贝尔曼方程是计算工具，TD Error 是学习信号。但理论归理论，怎么把这些数学工具变成能跑的算法？这就是本节要回答的问题。

我们将沿着一条清晰的演进路线——从 DP 到 MC 到 TD——看看 RL 的研究者们是如何一步步从"理想但不可用"走向"实用但不够完美"的。然后在旅程的终点，你会发现一个共同的瓶颈——表格装不下——以及一个优雅的解决方案。

## 第一代：动态规划（DP）——"坐在家里查地图"

如果你完全知道环境的转移概率 $P$ 和奖励函数 $R$——比如你在老虎机游戏中自己写的代码——那你根本不需要去"探索"。你可以坐在家里，直接用贝尔曼方程算出每个状态的精确价值。

这就是动态规划（Dynamic Programming），由理查德·贝尔曼在 1957 年提出 [^1]。它的更新规则就是贝尔曼最优方程的直接实现：

$$V(s) \leftarrow \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' | s, a) \, V(s') \right]$$

反复对所有状态执行这个更新，$V$ 最终会收敛到最优价值 $V^*$。完美、精确、优雅——理论上。

问题在于：现实中你几乎不可能知道完整的 $P$ 和 $R$。围棋有 $10^{170}$ 个状态，你不可能穷举所有转移概率；LLM 有天文数字的 token 序列组合，更不可能建出完整的转移矩阵。DP 就像一把只有在地狱里才能用的完美钥匙——理论最优，但现实不可用。

## 第二代：蒙特卡洛（MC）——"走完一整趟再回头看"

不知道环境模型？那就跑起来看看。

蒙特卡洛方法（Monte Carlo, MC）以摩纳哥的赌城命名——因为它本质上就是"靠运气采样"。这个名字最早由数学家斯塔尼斯拉夫·乌拉姆和约翰·冯·诺伊曼在 1940 年代的曼哈顿计划中使用，后来被系统性地引入 RL 领域。它的核心思想极其朴素：与其坐在家里算，不如实际走一趟，看看到底拿了多少分。采样完整的轨迹（从起点到终点），用实际回报 $G_t$ 来估计 $V(s)$：

$$V(s) \leftarrow V(s) + \alpha \left[ G_t - V(s) \right]$$

这里 $\alpha$ 是学习率，控制"新经验覆盖旧估计的速度"。$G_t - V(s)$ 就是"实际拿到的分数减去你之前的预测"——差多少就补多少。MC 方法给出的是无偏估计，因为你用的是真实的回报。但方差巨大——同一个状态，不同 episode 的 $G_t$ 可能天差地别。而且必须等到 episode 结束才能更新，不能"边走边学"。

> 第 5 章的 REINFORCE 就是 MC 方法在策略空间的直接应用——它需要跑完整个 episode，拿到完整的回报后才能更新策略。

## 第三代：时序差分（TD）——"每走一步就微调预判"

时序差分（Temporal Difference, TD）由安德鲁·巴托和理查德·萨顿在 1988 年正式提出 [^2]，被他们称为"RL 核心的、新颖的想法"。TD 是 DP 和 MC 的折中——它既不需要完整的环境模型（像 MC），又能边走边学（不像 MC 那样要等 episode 结束）。

$$V(s) \leftarrow V(s) + \alpha \underbrace{\left[ r + \gamma V(s') - V(s) \right]}_{\text{TD Error } \delta}$$

你认出来了吗？方括号里的就是我们上一节定义的 TD Error：$\delta = r + \gamma V(s') - V(s)$。TD 方法的更新规则用大白话来说就是：每走一步，用 TD Error 来微调你的价值估计——偏高就往下调，偏低就往上调。不需要等 episode 结束，也不需要知道环境模型。一步的反馈就够了。

Q-Learning 是 TD 方法最著名的变体，由克里斯·沃特金斯在 1989 年的博士论文中提出 [^3]，并在 1992 年与彼得·戴扬正式发表了收敛性证明。它直接学习 $Q$ 值而不是 $V$ 值：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

注意那个 $\max_{a'}$——它直接使用了贝尔曼最优方程的思想：不看所有动作的平均，只看最好的那个。这意味着 Q-Learning 学的是 $Q^*$（最优动作价值），不管你当前用什么策略在探索——这就是离策略（off-policy）学习。

三代方法的演进可以用一句话概括：DP 需要知道一切但不现实，MC 不需要知道一切但要等很久，TD 不需要知道一切也不需要等——它每走一步就能学一点。

## 动手：4×4 GridWorld 跑 Q-Learning

让我们用一个具体例子来感受 Q-Learning 的运作过程，亲眼看看 TD Error 是怎么从非零逐渐收敛到零的。

### 环境设定

```
┌───┬───┬───┬───┐
│ S │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │ G │
└───┴───┴───┴───┘
```

4×4 网格，左上角起点 $S$，右下角终点 $G$。每步奖励 -1（鼓励尽快到达终点），到达终点奖励 0。动作：上/下/左/右。初始 Q-table：全部为 0。

### 手算第 1 步：从 S 向右走

智能体从 $S = (0,0)$ 出发，选择向右走到 $(0,1)$。即时奖励 $r = -1$（每步都是 -1）。下一状态的所有 Q 值都是 0（初始化为 0）。所以 TD Target $= -1 + 0.9 \times 0 = -1$，TD Error $= -1 - 0 = -1$。新 Q 值 $= 0 + 0.1 \times (-1) = -0.1$。

TD Error = -1 的含义：之前 Q 值是 0（"我什么都不知道，猜测走这步不赚不亏"），实际走了一步却扣了 1 分——预测严重偏高，所以把 Q 值下调了 0.1。

第 2 步的情况类似：从 $(0,1)$ 继续向右走到 $(0,2)$，TD Error 仍然是 -1，新 Q 值也是 -0.1。因为周围的格子都还没学过，Q 值全是 0。

经过大量训练后，Q 值会收敛。以 $Q(S, \text{右})$ 为例：从 $S$ 到 $G$ 最短路径需要 6 步，每步 -1，考虑 $\gamma = 0.9$ 的折扣后 $Q((0,0), \text{右}) \approx -5.69$。此时 TD Error $\approx 0$——预判和实际一致了，学习完成。这个过程揭示了 Q-Learning 的本质：TD Error 从一开始的 -1，通过成百上千次的微调，逐渐趋近于 0。每一次微调都是在说"上次猜错了，这次修一点"。

### 用代码验证

```python
import numpy as np

# 4x4 GridWorld Q-Learning
Q = np.zeros((16, 4))  # 16 个状态, 4 个动作 (上右下左)
alpha, gamma, epsilon = 0.1, 0.9, 0.1
goal = 15  # 右下角的索引

def state_to_idx(row, col):
    return row * 4 + col

def step(state, action):
    """执行动作，返回 (下一状态, 奖励, 是否结束)"""
    row, col = state // 4, state % 4
    if action == 0: row = max(row - 1, 0)      # 上
    elif action == 1: col = min(col + 1, 3)     # 右
    elif action == 2: row = min(row + 1, 3)     # 下
    elif action == 3: col = max(col - 1, 0)     # 左
    next_state = state_to_idx(row, col)
    reward = 0 if next_state == goal else -1
    done = next_state == goal
    return next_state, reward, done

# 训练 1000 个 episode
for ep in range(1000):
    state = 0  # 起点 S
    while state != goal:
        # ε-贪婪：90% 选最优，10% 随机探索
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Q-Learning 更新
        td_target = reward + gamma * np.max(Q[next_state])
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state

# 打印从起点出发的最优路径
print("收敛后的 Q((0,0), 右) =", Q[0, 1].round(2))
print("最优路径（从 S 出发的动作序列）：")
state = 0
actions = ["↑", "→", "↓", "←"]
path = []
while state != goal:
    a = np.argmax(Q[state])
    path.append(actions[a])
    state, _, _ = step(state, a)
print(" → ".join(path))
```

预期输出：

```
收敛后的 Q((0,0), 右) = -5.69
最优路径（从 S 出发的动作序列）：
→ → → ↓ ↓ ↓
```

## 共同的瓶颈：表格装不下

到目前为止，DP、MC、TD 三代方法都有一个隐含的前提：你能把每个状态的 $V$ 值或 $Q$ 值存在一张表格里。GridWorld 只有 16 个格子，当然没问题。但真实世界的状态空间有多大？

猜硬币只有 2 个状态，轻松搞定。井字棋有 $3^9 \approx 20{,}000$ 种局面，勉强可以。但国际象棋有 $\approx 10^{47}$ 种局面，围棋有 $3^{361} \approx 10^{170}$ 种——而可观测宇宙中的原子总数也不过 $\sim 10^{80}$ 个。CartPole 的状态空间是 $\mathbb{R}^4$（无限连续），理论上存在无限多个不同的状态。更不用提大模型了——假设词表大小 50,000、序列长度 1,000，可能的状态数量是 $50000^{1000}$，这是一个比宇宙中原子总数还大无数倍的数字。

表格方法的问题在于：它对每个状态都是独立学习的——如果你从没见过状态 $s$，你就完全不知道 $V(s)$ 是多少。

### 解决方案：函数逼近

既然不可能为每个状态单独存一个值，那换一个思路：能不能学习一个函数，输入状态 $s$，输出 $V(s)$ 的近似值？

$$V(s) \approx f(s; \theta)$$

其中 $\theta$ 是函数的参数，通过训练来调整，使得 $f(s; \theta)$ 尽可能接近真实的 $V(s)$。这就是函数逼近（Function Approximation）的核心思想。

神经网络则是最强大的函数逼近器。根据通用逼近定理，一个带有一个隐藏层的前馈神经网络，只要隐藏层足够宽，就可以以任意精度逼近任何连续函数。更重要的是，神经网络能**举一反三**——假设智能体在训练中见过很多"接近直立"的 CartPole 状态，它们的 $V$ 值都很高。那么当它遇到一个从未见过但长得很像的状态时，神经网络能通过输入的相似性，推断出"这个状态的价值应该也很高"。这就是泛化能力——不需要穷举每一个状态，只要见过足够多的"代表"，就能推断未知状态的值。

还记得第 1 章我们用的 `MlpPolicy` 吗？它内部就是一个小型神经网络。当我们调用 `model.learn(20000)` 时，这个网络就在不断调整自己的参数 $\theta$，使得它对每个 CartPole 状态输出的价值估计越来越准确。只不过当时 SB3 把这一切都封装在了黑盒里。现在你知道了：那个黑盒里在做的事情，本质上就是用 TD Error 驱动的函数逼近。

理解了"神经网络 = 函数逼近器"这一点，后续所有深度 RL 架构就不再是魔法，而是数学工具的组合：第 5 章的 Critic 网络逼近 $V(s)$，第 4 章的 Q 网络逼近 $Q(s,a)$，第 6 章的奖励模型逼近奖励函数 $R$。"Deep Reinforcement Learning"中的"Deep"，指的就是用深度神经网络来做函数逼近的 RL。

## 闭环验证：老虎机的 V 值

在本章开头，我们用老虎机体验了 RL 的基本交互。现在，在学完了所有理论工具之后，让我们回到那里，做一个完整的闭环验证。

在贝尔曼方程一节中，我们用数学推导得到了"永远拉 A"策略下的理论 V 值：$V = 2.0$（A 台出奖率 60%，$\gamma = 0.9$）。现在用蒙特卡洛仿真来验证——跑大量 episode，看平均回报是否接近 2.0：

```python
import random

def simulate_bandit_value(prob_a=0.6, gamma=0.9, num_episodes=10000):
    """用蒙特卡洛采样验证贝尔曼方程的理论 V 值"""
    total_returns = []
    for _ in range(num_episodes):
        ret = 0
        discount = 1.0
        while True:
            # 永远拉 A
            reward = 1 if random.random() < prob_a else -1
            ret += discount * reward
            discount *= gamma
            if discount < 1e-10:  # 折扣足够小就截断
                break
        total_returns.append(ret)
    return sum(total_returns) / len(total_returns)

empirical_V = simulate_bandit_value()
theoretical_V = 0.2 / (1 - 0.9)

print(f"贝尔曼方程理论 V 值: {theoretical_V:.4f}")
print(f"蒙特卡洛仿真 V 值:   {empirical_V:.4f}")
print(f"误差:                {abs(empirical_V - theoretical_V):.4f}")
```

预期输出：

```
贝尔曼方程理论 V 值: 2.0000
蒙特卡洛仿真 V 值:   2.0018
误差:                0.0018
```

理论值和仿真值高度吻合。这验证了贝尔曼方程的正确性——数学推导和实际运行的结果是一致的。从老虎机的直觉体验，到 MDP 的数学形式化，到贝尔曼方程的计算，再到蒙特卡洛的仿真验证，我们走完了一个完整的闭环。

## RL 的全景地图：两条路线与一个关键区分

### 路线分野：Value-Based vs Policy-Based

所有 RL 算法可以分成两大阵营。

Value-Based 路线的思路是：先学 $V(s)$ 或 $Q(s,a)$，再从中推导策略。代表算法是 Q-Learning → DQN → Double DQN → Rainbow。它的优势是样本效率高（off-policy 可复用旧数据），但只能处理离散动作，策略是"隐式"的。

Policy-Based 路线的思路是：直接学策略 $\pi(a|s)$，不经过 Q。代表算法是 REINFORCE → Actor-Critic → PPO → DPO → GRPO。它能处理连续动作空间，策略更灵活，但样本效率低（on-policy 需不断采新数据）。

为什么本书选择 Policy-Based？因为大模型对齐任务（RLHF/DPO/GRPO）本质上是在连续的策略空间中优化——模型输出的每一步都是从几万个 token 中采样，这是一个连续概率分布的问题，Value-Based 方法不太擅长。但这并不意味着 Value-Based 不重要——第 4 章我们会回来走 DQN 路线，届时你会发现两个世界的底层逻辑惊人地一致。

### 关键区分：On-policy vs Off-policy

这个区分决定了训练数据能不能复用，在大模型时代尤其重要。

On-policy（同策略）只能用当前策略产生的数据。就像换了跑步姿势后，之前的训练记录就过期了。代表算法有 REINFORCE、PPO、GRPO。训练稳定，但数据用完就扔。

Off-policy（异策略）可以用任何策略产生的旧数据。就像可以用别人的训练笔记改进自己的技术。代表算法有 Q-Learning、DQN、SAC。样本效率高，但训练可能不稳定。

在大模型时代，这个区分有非常具体的体现：PPO 是 on-policy，所以每次 RLHF 训练都要用当前模型重新生成回答，非常吃显存和算力（第 6 章）。DQN 是 off-policy，可以把所有历史经验存进池子反复学习（第 4 章）。DPO 更极端，连在线生成都不需要，直接用固定的离线偏好数据集训练（第 7 章）。GRPO 是 on-policy，但用组内比较省掉了 Critic，降低了采样成本（第 8 章）。

## 本章小结

恭喜你走完了全书的理论基石章节。让我们回顾这段旅程中的关键收获：

我们从**两台老虎机**出发——亲手体验了 RL 的核心矛盾（探索 vs 利用），理解了期望回报如何量化策略好坏。然后掌握了 **MDP 五元组**——理解了 $S, A, P, R, \gamma$ 如何统一描述所有 RL 问题。接着通过手画宝藏地图"发明"了**贝尔曼方程**——把"评估一个局面"变成递归的"即时奖励 + 未来价值"。认识了 **TD Error**——预测与现实的落差，贯穿全书的学习信号。走过了 **DP → MC → TD** 三代方法的演进，亲手跑了 GridWorld Q-Learning，看到 TD Error 从 -1 收敛到 0。理解了"深度 RL"的本质——**神经网络作为函数逼近器**，让 RL 从表格走向高维世界。最后建立了方法全景图——Value-Based vs Policy-Based 的分野，On-policy vs Off-policy 的区分。

这些概念将在后续章节反复出现：第 5 章的 Critic 网络是 $V(s)$ 的实现，第 6 章的 GAE 基于 TD Error，第 4 章的 DQN 直接学习 $Q(s, a)$，第 7 章的 DPO 和第 8 章的 GRPO 则在策略空间中直接优化。

## 练习

1. **修改老虎机的概率**：把 `prob_a` 改为 0.3（A 台只有 30% 出奖率），重新计算"永远拉 A"策略的 $V$ 值（$\gamma = 0.9$）。最优策略应该是什么？期望回报是多少？

2. **GridWorld 实验**：在 Q-Learning 代码中，把网格改为 5×5，终点改为右上角。观察 Q 值收敛需要的 episode 数量如何变化。

3. **折扣因子的影响**：在老虎机中，分别用 $\gamma = 0.5, 0.9, 0.99, 0.999$ 计算"永远拉 A"的 $V$ 值，画出 $V$ 随 $\gamma$ 变化的曲线。你能从直觉上解释这条曲线的形状吗？

4. **状态空间思考**：假设你要设计一个扫地机器人 RL 系统。状态应该包含哪些信息？动作是什么？奖励怎么设计？（提示：考虑"覆盖率"和"电量"的权衡。）

## 参考文献

[^1]: Bellman, R. (1957). _Dynamic Programming_. Princeton University Press.

[^2]: Sutton, R. S., & Barto, A. G. (1998). _Reinforcement Learning: An Introduction_. MIT Press.

[^3]: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8(3-4), 279-292.

[^4]: Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518(7540), 529-533. [DOI](https://doi.org/10.1038/nature14236)
