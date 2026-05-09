# 4.3 观察训练过程

## 本节导读

**核心内容**

- 学会阅读 Reward 曲线，识别训练过程中的平台期、跳升与收敛三个阶段。
- 通过消融实验（去掉经验回放、去掉目标网络）验证 DQN 各组件的不可替代性。
- 理解超参数敏感性，掌握学习率、回放池容量、目标网络更新频率等关键调优方向。

**核心公式**

$$
y = r + \gamma (1-d) \max_{a'} Q(s', a'; \theta^-) \quad \text{（TD Target：用冻结的目标网络计算一步更新目标）}
$$

> **TD Target（时序差分目标）：**
>
> - $r$：这一步获得的即时奖励。
> - $\gamma$：折扣因子，控制对未来回报的重视程度。
> - $d$：终止标记。若 episode 已经真正结束，未来价值项被清零。
> - $\theta^-$：目标网络的参数，每隔若干步才从 Q-Network 复制一次，在两次复制之间保持冻结，提供稳定的更新目标。

$$
L(\theta) = \mathbb{E}\left[\left(y - Q(s, a; \theta)\right)^2\right] \quad \text{（DQN 损失函数：最小化预测 Q 值与 TD Target 的均方误差）}
$$

> **DQN 损失函数：**
>
> - $y$：TD Target，是"现实告诉你应该预测的值"。
> - $Q(s, a; \theta)$：Q-Network 当前的预测值。
> - 损失越小，说明网络的预测越接近真实的一步回报——训练的目标就是不断缩小这个差距。

**为什么需要这些公式**

上一节我们跑通了一个 LunarLander 上的 DQN，但它为什么能跑起来？靠的就是这两个公式在背后驱动：**TD Target** 给出"这一步应该学多少"——回忆一下，TD Target = $r + \gamma (1-d) \max_{a'} Q(s', a'; \theta^-)$，即"已经拿到的奖励 + 目标网络给出的未来最高分"，并在真正终止时切断未来项；**损失函数**把这个差距变成梯度去更新网络。训练过程是否健康，完全取决于这两个公式是否在正常工作。这一节我们不看公式本身，而是看公式的"临床表现"——Reward 曲线是否相对随机策略变好、Q 值是否从随机噪声变成了有意义的动作偏好、消融掉某个组件后训练是否会崩塌。理解这些，你才能从"跑通代码"真正进阶到"理解训练"。

但"跑通"只是第一步——真正理解 DQN，需要知道训练过程中每个组件在做什么、起了什么作用。就像一个医生不能只看病人"治好了"，还要理解每种药分别起了什么作用。这一节，我们会通过分析训练日志和消融实验，深入理解经验回放和目标网络各自的贡献。

## Reward 曲线：从平台期到跳升

先看最直观的指标——每个 episode 的总奖励。在 LunarLander 中，随机策略平均回报大约在 -200 左右，稳定解决通常需要平均回报接近 200。短训练的 DQN 不一定能直接解决环境，但应该能逐步脱离随机策略：从乱喷发动机、撞地、翻滚，变成至少学会减速、调姿态、靠近着陆区。

为什么会有平台期？因为在训练初期，经验回放池里的经验还很有限，而且大多是"坏着陆经验"——飞船速度过大、角度太歪、主发动机开得太晚或太早。这些失败经验虽然告诉网络"这样不好"，但不足以告诉网络"怎样才算好"。只有当智能体偶然出现一段不那么糟的下降轨迹，回放池里才会出现更有价值的对照样本。而这些样本一旦出现，就会被反复采样，加速学习——这就是曲线偶尔突然变好的原因。

这和人类练习着陆游戏很像：一开始你不是忘了开主发动机，就是角度修过头；后来某一次你碰巧慢慢落下来了，大脑立刻记住"这种速度和姿态比较安全"，之后就会围绕这类轨迹反复调整。

## Q 值的演化：从随机到有意义

让我们追踪 Q 值在训练过程中是如何变化的。在 LunarLander 的初始状态下，Q-Network 对 4 个动作的 Q 值最初只是接近 0 的随机数——因为网络参数是随机初始化的。随着训练进行，它应该逐渐学到动作偏好：下降太快时主发动机更有价值，角度偏了时某一侧喷口更有价值，已经稳定接近地面时乱喷反而可能降低回报。

```python
# 在训练的不同阶段检查初始状态的 Q 值
def check_q_values(agent, env):
    state, _ = env.reset()
    with torch.no_grad():
        q = agent.q_net(torch.FloatTensor(state).unsqueeze(0))
    actions = ["不喷", "左侧喷口", "主发动机", "右侧喷口"]
    for name, value in zip(actions, q[0].tolist()):
        print(f"  Q({name}) = {value:.3f}")
```

你会观察到这样的变化：

- **训练早期**：四个动作的 Q 值都接近随机初始化，彼此差别不稳定。
- **训练中期**：主发动机和侧喷口开始出现分工，但 Q 值可能忽高忽低，说明网络还在重新校准不同轨迹的价值。
- **训练后期**：动作偏好开始和状态对应起来。速度太快时主发动机更高，角度偏斜时对应侧喷口更高，稳定下降时“不喷”也可能是好动作。

Q 值从随机噪声演化到有意义的评估，这个过程就是 DQN 的"学习"。而驱动这个学习的是 TD Error——每一步的"预测与现实的落差"。

## 消融实验：拆掉一个零件会怎样？

理解一个系统的最好方式，是拆掉某个零件看看会出什么问题。让我们做两个消融实验。

### 消融一：去掉经验回放（逐帧训练）

把经验回放去掉，改为每走一步就立即用这条经验更新网络——不做采样，不存池子。

```python
# 无经验回放的版本：逐帧更新
for episode in range(num_episodes):
    state, _ = env.reset()
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 直接用当前经验更新，不存入回放池
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            td_target = reward + agent.gamma * agent.target_net(next_state_t).max() * (1 - done)
        q_value = agent.q_net(state_t)[0][action]
        loss = agent.loss_fn(q_value, td_target)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        state = next_state
        if done:
            break
```

你会发现训练变得极不稳定——reward 曲线剧烈波动，经常训练了 300 轮也无法收敛。原因就是我们之前讨论过的：连续帧几乎一模一样，逐帧训练让梯度被当前场景绑架。网络"记住"了最近几帧的特征，却"遗忘"了之前学到的策略。这在深度学习文献中被称为**灾难性遗忘**（Catastrophic Forgetting）。

经验回放通过随机采样解决了这个问题：每一批训练数据来自不同时间段的不同场景，梯度方向更加均衡，网络不会偏废。

### 消融二：去掉目标网络

把目标网络去掉，改为用 Q-Network 自己计算 TD Target——也就是说，TD Target $= r + \gamma (1-d) \max_{a'} Q(s', a'; \theta)$ 中的 $\theta$ 就是正在被更新的参数。

让我们仔细想想这意味着什么。正常情况下（有目标网络），TD Target 用的是冻结的参数 $\theta^-$：

$$\text{正常：} \quad y = r + \gamma (1-d) \max_{a'} Q(s', a'; \underbrace{\theta^-}_{\text{冻结，不动}})$$

去掉目标网络后，TD Target 用的是正在被更新的 $\theta$：

$$\text{无目标网络：} \quad y = r + \gamma (1-d) \max_{a'} Q(s', a'; \underbrace{\theta}_{\text{每步都在变！}})$$

这就像一个学生在考试，但标准答案是根据他的答案实时生成的——他每写一个字，答案就跟着变一次。学生会发现：不管怎么写，和"标准答案"的差距永远降不下来，因为答案在追着他跑。在数学上，这意味着优化的目标函数不是固定的，而是一个"移动靶"——梯度下降的收敛性保证就失效了。

```python
# 无目标网络的版本：用 Q-Network 自己计算 TD Target
with torch.no_grad():
    # 直接用 q_net 而不是 target_net
    next_q_max = agent.q_net(next_states).max(dim=1)[0]
    td_target = rewards + agent.gamma * next_q_max * (1 - dones)
```

你会发现训练同样不稳定，但表现和去掉经验回放有所不同。去掉经验回放的问题是波动大、收敛慢；去掉目标网络的问题更隐蔽——训练目标每一步都跟着 Q-Network 改，Q 值更容易震荡或发散。

另一个相关但不完全相同的问题是 **Q 值过估计**：TD Target 中的 $\max$ 操作会倾向于选择被高估的 Q 值。想象你在估测一组随机数的最大值：即使每个数的估计是无偏的，取最大值后结果也会偏高——因为你总是选了被高估得最厉害的那个。目标网络主要解决"目标移动"问题，能让估值变化慢一些；真正针对过估计的是 Double DQN，它把"选择动作"和"评估动作"拆给两个网络，我们下一节会讨论。

## 超参数敏感性

DQN 的训练效果对几个超参数比较敏感。以下是 LunarLander 上主要的调优方向：

**学习率（learning rate）**：控制参数更新的步长。太大会导致训练不稳定（Q 值震荡），太小则学习太慢。常用值：$10^{-4}$ 到 $10^{-3}$。

**经验回放池容量（buffer capacity）**：太小则经验多样性不足，太大则旧经验可能过时。对于 LunarLander，$10^5$ 通常是一个合理起点；对于 Atari，通常需要 $10^5$ 到 $10^6$。

**目标网络更新频率（target update）**：太频繁起不到稳定作用，太稀疏则目标过时。LunarLander 上可以从 500 到 1000 步更新一次开始试，比玩具任务更需要稳定目标。

**批次大小（batch size）**：每次更新采样的经验数量。太小时梯度噪声大，太大则计算成本高。常用值：32 到 128。

**$\varepsilon$ 衰减速度**：控制从探索到利用的过渡速度。衰减太快可能导致智能体还没充分探索就固化了次优策略，衰减太慢则浪费时间在已知不好的动作上。

下面这段代码可以帮你快速测试不同超参数的效果：

```python
# 超参数对比实验
configs = [
    {"name": "默认", "lr": 1e-3, "buffer": 10000, "target_update": 10},
    {"name": "小学习率", "lr": 1e-4, "buffer": 10000, "target_update": 10},
    {"name": "大回放池", "lr": 1e-3, "buffer": 50000, "target_update": 10},
    {"name": "频繁目标更新", "lr": 1e-3, "buffer": 10000, "target_update": 50},
]

for config in configs:
    agent = DQNAgent(state_dim=8, action_dim=4,
                     lr=config["lr"],
                     buffer_capacity=config["buffer"],
                     target_update=config["target_update"])
    # ... 运行训练循环 ...
    print(f"{config['name']}: 最终50轮平均奖励 = {avg_reward:.1f}")
```

<details>
<summary>思考题：为什么经验回放池大小对 Atari 的影响比 LunarLander 大得多？</summary>

LunarLander 的状态空间虽然是连续的，但观测只有 8 个数字，主要变化围绕位置、速度、姿态和接触状态展开。经验回放池需要覆盖不同着陆阶段，但还不需要记住大量视觉场景。Atari 游戏的画面变化丰富得多——同一个游戏的不同关卡、不同敌人的位置、不同的道具状态——需要的经验多样性远高于 LunarLander。所以 Atari 上通常需要更大的回放池来保证采样的多样性。

</details>

理解了 DQN 各组件的作用和调优方法，接下来让我们看看 DQN 在 2015 年之后的演进——[DQN 家族与视角迁移](./dqn-family)。
