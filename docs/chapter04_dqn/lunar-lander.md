# 4.4 LunarLander：更适合 DQN 的控制任务

## 本节导读

**核心内容**

- 理解为什么 `LunarLander-v3` 适合作为本章 DQN 的统一低维控制案例。
- 学会读懂 LunarLander 的 8 维状态、4 个离散动作和分层奖励信号。
- 用一个短版 DQN 训练脚本观察：当奖励信号更丰富时，经验回放和目标网络如何让策略逐步脱离随机控制。

**核心公式**

$$
y = r + \gamma (1-d)\max_{a'}Q(s',a';\theta^-)
\quad \text{（DQN 的 TD Target：终止时切断未来价值）}
$$

$$
\mathcal{L}(\theta)=
\mathbb{E}\left[\left(y-Q(s,a;\theta)\right)^2\right]
\quad \text{（用均方 TD Error 训练 Q 网络）}
$$

$$
\epsilon_t =
\epsilon_{\text{final}}+
(\epsilon_{\text{start}}-\epsilon_{\text{final}})
\max\left(1-\frac{t}{T}, 0\right)
\quad \text{（线性探索衰减：从多试错过渡到多利用）}
$$

**为什么需要这些公式**

前面我们已经拆开了 DQN 的三个核心组件：Q 网络、经验回放和目标网络。现在需要一个足够具体的环境，把这些组件放回真实训练过程里观察。这个环境不能太简单，否则 DQN 看起来会过于轻松；也不能一上来就是像素游戏，否则图像预处理和 CNN 会盖过算法本身。

换个角度，MountainCar 能暴露探索不足的问题，但它对入门代码太苛刻：在到达山顶之前几乎没有可用的正向信号，最小 DQN 往往长时间卡在 -200。作为“失败诊断”它很有价值，作为“DQN 的第二个正向案例”却不够友好。

LunarLander 正好在中间。它仍然是低维连续状态、离散动作，适合 DQN；但奖励比 CartPole 更复杂，比 MountainCar 更有梯度。飞船靠近着陆区、速度变慢、姿态更稳、腿接触地面，都会影响奖励。也就是说，环境会给 agent 更多“你正在变好或变坏”的线索。真正的问题不再是“完全没有信号”，而是“如何把多维状态、多个动作和噪声较大的回报稳定地学起来”。

## LunarLander 环境

`LunarLander-v3` 是 Gymnasium 的经典控制任务。智能体控制一艘小型登月舱，让它从空中下降并尽量平稳地落在两个旗帜之间。状态是 8 维连续向量，动作是 4 个离散选择。

```python
import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=0)

print(f"状态空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")
print(f"初始状态: {np.round(obs, 3)}")
```

预期输出（`Box` 的上下界较长，这里省略中间数字）：

```text
状态空间: Box([...], [...], (8,), float32)
动作空间: Discrete(4)
初始状态: [ 0.006  1.399  0.578 -0.528 -0.007 -0.131  0.     0.   ]
```

8 个状态分量可以这样理解：

| 分量        | 含义                     | 直觉问题                 |
| ----------- | ------------------------ | ------------------------ |
| `x, y`      | 飞船相对着陆区的位置     | 离中心远不远，高度多少？ |
| `vx, vy`    | 水平和垂直速度           | 是不是下降太快？         |
| `angle`     | 飞船倾斜角               | 姿态有没有歪？           |
| `angular`   | 角速度                   | 正在越转越快吗？         |
| `left_leg`  | 左腿是否接触地面，0 或 1 | 有没有碰到地面？         |
| `right_leg` | 右腿是否接触地面，0 或 1 | 是否两条腿都落稳？       |

4 个动作分别是：什么都不做、开左侧姿态喷口、开主发动机、开右侧姿态喷口。这里的难点不只是“落下去”，而是要同时控制位置、速度和角度。主发动机能减速，但会耗油；侧喷口能调姿态，但用错方向会越调越偏。

与前两个环境对比：

| 环境            | 状态     | 动作     | 奖励信号                        | 适合讲什么               |
| --------------- | -------- | -------- | ------------------------------- | ------------------------ |
| CartPole        | 4 维连续 | 2 个离散 | 每活一步 +1                     | DQN 基本组件能否跑通     |
| MountainCar     | 2 维连续 | 3 个离散 | 每步 -1，成功前几乎没有正向信号 | 稀疏奖励和探索失败       |
| **LunarLander** | 8 维连续 | 4 个离散 | 位置、速度、姿态、接触都有反馈  | 更真实的离散动作控制任务 |

LunarLander 的好处在于：它不会像过于简单的入门环境那样很快满分，也不会像 MountainCar 那样把最小 DQN 长时间按在失败区间。你能在较短训练中看到策略从随机喷火，逐渐变成“至少知道要减速和调姿态”。

## 随机策略基线

先看随机策略。这个基线很重要：如果 DQN 没有明显超过随机，就不能说它学到了控制策略。

```python
import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3")
rng = np.random.default_rng(0)

returns = []
for ep in range(50):
    obs, _ = env.reset(seed=ep)
    total_reward = 0.0
    for step in range(1000):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    returns.append(total_reward)

print(f"随机策略平均回报: {np.mean(returns):.1f}")
print(f"最好一轮: {np.max(returns):.1f}")
print(f"最差一轮: {np.min(returns):.1f}")
```

预期输出：

```text
随机策略平均回报: -210.2
最好一轮: 8.3
最差一轮: -460.8
```

随机策略偶尔看起来没那么糟，但整体很差。飞船会乱喷发动机，有时提前撞地，有时转得很厉害，有时明明接近地面却没有控制住速度。这个结果给了我们一个清楚的参照：DQN 至少应该把平均回报从 -200 左右往上推。

## 用 DQN 训练 LunarLander

这段代码使用 Stable-Baselines3 的 DQN。这里不是为了把库当黑盒，而是为了快速验证一件事：同样是 DQN 思路，当环境奖励更有梯度时，训练曲线确实会比 MountainCar 友好得多。

```python
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

seed = 0
set_random_seed(seed)

train_env = Monitor(gym.make("LunarLander-v3"))
train_env.reset(seed=seed)
eval_env = Monitor(gym.make("LunarLander-v3"))

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=0,
    seed=seed,
)

last_steps = 0
for steps in [5_000, 10_000, 20_000, 40_000, 80_000]:
    model.learn(total_timesteps=steps - last_steps, reset_num_timesteps=False)
    last_steps = steps
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
        warn=False,
    )
    print(
        f"timesteps={steps:6d} "
        f"eval10_mean={mean_reward:8.1f} "
        f"std={std_reward:6.1f}"
    )
```

一次实测运行结果如下。由于 `LunarLander-v3` 依赖物理模拟，DQN 又依赖神经网络优化，不同机器、线程和底层库版本下，数值可能不会逐行完全一致；这里要看的不是每个 checkpoint 的精确分数，而是它是否明显超过随机策略基线。

```text
timesteps=  5000 eval10_mean=   -46.4 std= 106.1
timesteps= 10000 eval10_mean=  -135.9 std=  21.7
timesteps= 20000 eval10_mean=   -93.3 std= 108.3
timesteps= 40000 eval10_mean=   -63.8 std=  28.8
timesteps= 80000 eval10_mean=   -65.2 std=  40.9
```

先看直觉：这条曲线并不单调，强化学习本来就有波动。但和随机策略的 -210 左右相比，80k 步时的平均回报已经明显提高。它还没有稳定达到 `LunarLander-v3` 通常定义的“解决”（平均回报约 200 分），但已经能说明 DQN 在这个任务上学到了比随机更好的控制行为。

真正的问题在于，LunarLander 的奖励虽然更丰富，但仍然有很多局部选择：该不该开主发动机、什么时候修角度、是否为了减速牺牲一点燃料。DQN 的经验回放让这些不同阶段的经验能反复采样，目标网络让 TD Target 不至于每一步剧烈漂移；这正是前面两个组件在更复杂任务中的作用。

## 读懂这条训练曲线

不要只盯最后一个数字。LunarLander 的短训练通常有三个现象：

**第一，前期可能突然变好。** 因为随机探索偶尔会产生“不那么糟”的落地轨迹，回放池反复采样后，网络会先学到一些粗糙规则：下降太快时开主发动机，角度太歪时用侧喷口。

**第二，中期会波动。** DQN 的 Q 值估计会影响动作选择，动作选择又影响新收集到的数据。某个阶段策略看起来变好，不代表之后不会退步。即使继续训练到更长步数，也可能出现短暂回落；这不是代码坏了，而是 off-policy 控制任务常见的不稳定。

**第三，短训练不是最终性能。** 如果目标是稳定解决 LunarLander，通常要更长训练、更仔细的超参数、更大的评估集，甚至换 Double DQN、Dueling DQN 或 PPO。这里的目的更朴素：用一个低维、离散动作、但比玩具平衡任务更真实的环境，看见 DQN 的学习信号确实能起来。

## 和 MountainCar 的区别

换个角度看，LunarLander 并不是“更简单”，而是“更适合这个阶段”。MountainCar 的关键行为是左右摆动借势，但环境在成功前几乎不给中间鼓励；LunarLander 则在接近目标、降低速度、保持姿态、腿部接触时都提供了更细的反馈。

因此，本节想强调的不是“DQN 一定很强”，而是：**例子要和教学目标匹配**。如果目标是讲稀疏奖励失败，MountainCar 很好；如果目标是统一承载本章 DQN 的训练分析、组件作用和后续改进，LunarLander 更合适。

## 本节收获

- `LunarLander-v3` 是本章 DQN 的统一低维控制案例：状态连续、动作离散、奖励结构足够丰富。
- 随机策略平均回报约 -210，短训练 DQN 可以明显超过这个基线，但仍可能波动。
- 经验回放和目标网络在 LunarLander 中不只是“代码结构”，而是在噪声较大的控制任务中稳定训练的关键。
- MountainCar 更适合讲稀疏奖励和探索失败；LunarLander 更适合讲 DQN 从表格方法走向更真实控制任务。

下一节我们来看 DQN 的后续演进——Double DQN、Dueling DQN 和 Rainbow，看看研究者在“让 Q 值估计更准确、更稳定”这条路上还做了哪些改进。[DQN 家族与视角迁移](./dqn-family)

## 参考文献

[^1]: Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518(7540), 529-533.

[^2]: Raffin, A., et al. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. _Journal of Machine Learning Research_, 22(268), 1-8.
