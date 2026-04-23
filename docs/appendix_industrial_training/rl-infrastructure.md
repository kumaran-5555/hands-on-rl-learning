# B.1 RL 采样基础设施

> RL 训练的核心循环是"采样 → 训练 → 再采样"，**瓶颈永远在采样这一步**。但不同规模的 RL，采样瓶颈的原因完全不同——单机卡在 GPU 调度，游戏 RL 卡在环境交互，LLM RL 卡在推理生成。
>
> 本节只回答一个问题：**RL 的采样为什么是瓶颈，各规模怎么解决？** 按规模分三个阶段，每个阶段讲清楚瓶颈在哪、用什么技术解决。

## 为什么 RL 的基础设施比监督学习复杂

监督学习的训练循环是**静态**的：

```
数据集 → DataLoader → 前向 → 反向 → 更新
```

RL 的训练循环是**动态**的：

```
策略采样 → 环境交互 → 收集轨迹 → 前向 → 反向 → 更新 → 用新策略重新采样
```

关键区别：**训练数据是策略自己生成的**。这意味着采样速度直接决定训练速度，环境交互往往是瓶颈。

更工程一点地说，RL 系统的吞吐由三类速率共同决定：

- 采样端能产出多少数据（`steps/s` 或 `tokens/s`）
- 训练端能消化多少数据（每步训练的 batch size、并行策略、反向速度）
- 奖励/环境反馈有多快（环境 step、判题、Reward Model/Judge 推理）

它们只要有一块跟不上，最终吞吐就会被那一块“卡死”。后续 B.2（异步训练）和 B.6（监控与排障）基本都在围绕这个矛盾做工程化处理。

## 阶段一：单机 RL（Gymnasium）

**场景**：CartPole、LunarLander、学术实验

**瓶颈**：单环境太慢，GPU 大部分时间在等 CPU 做环境 step。

**解法——向量化环境（vectorized env）**：

```python
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

# 同步向量化：N 个环境批量 step
envs = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
obs, info = envs.reset()            # shape: (8, obs_dim)
actions = policy(obs)               # 一次推理得到 8 个动作
obs, rewards, ... = envs.step(actions)  # 批量交互
```

| 方式             | 原理              | 适用场景                    |
| ---------------- | ----------------- | --------------------------- |
| `SyncVectorEnv`  | 主进程中顺序 step | 轻量环境（CartPole、Atari） |
| `AsyncVectorEnv` | 多进程并行 step   | 计算密集环境（物理仿真）    |

这一阶段全部在单机完成，CPU 做环境交互，GPU 做策略推理和训练。

## 阶段二：游戏/仿真 RL（并行化环境）

**场景**：Atari、MuJoCo、Isaac Gym 机器人仿真

**瓶颈**：转移到**环境交互**。一帧 Atari 画面需要 CPU 模拟，一个 MuJoCo 步需要物理引擎计算。

**解法一——CPU 并行**：

```python
# 典型配置：32-64 个并行环境
envs = AsyncVectorEnv([make_atari_env() for _ in range(64)])
# 吞吐量：单环境 ~500 fps → 64 并行 ~30000 fps
```

**解法二——GPU 并行仿真（Isaac Gym）**：

NVIDIA Isaac Gym 把物理仿真直接搬到 GPU 上，数万个环境并行：

```
传统方式：  CPU 物理引擎 × 64 环境 → GPU 策略推理
Isaac Gym： GPU 物理仿真 × 4096 环境 + GPU 策略推理（全在 GPU 上）
```

| 对比     | CPU 并行 (MuJoCo × 64) | GPU 并行 (Isaac Gym × 4096) |
| -------- | ---------------------- | --------------------------- |
| 采样速度 | ~10K fps               | ~1M fps                     |
| 数据传输 | CPU→GPU 每步           | 零拷贝                      |
| 适用场景 | 少关节机器人           | 人形机器人、灵巧手          |

**Sample Factory 极致吞吐**：异步 Actor-Learner 架构，Actor 进程持续收集轨迹，Learner 进程持续更新策略，共享内存零拷贝。Atari 上可达 100K+ fps。

如果你把“并行环境”继续往上推，会很快遇到 Python 调度/进程通信的天花板。下一步通常是把系统拆成 Actor（采样）和 Learner（训练）两类进程：Actor 只负责把轨迹稳定地喂出来，Learner 只负责把 GPU 吃满。这类解耦架构的代表之一是 IMPALA。  

## 阶段三：LLM RL 采样基础设施

**场景**：PPO/GRPO 训练 LLM（7B-70B）

**瓶颈**：转移到**推理生成**。LLM RL 的环境不再是 Gym，而是文本生成本身。

三个核心挑战：

1. **Rollout 推理密集**：GRPO 的 k=16 意味着每个 prompt 要生成 16 个回答
2. **模型本身就很大**：7B 模型的推理需要一整张 GPU
3. **生成和训练的 GPU 需求波动**：Rollout 时训练闲置，反之亦然

**核心架构：Rollout-Training 分离**

```
┌─────────────────────┐     ┌─────────────────────┐
│   Rollout Workers    │     │   Training Workers   │
│   (vLLM 推理集群)    │     │   (FSDP/Megatron)    │
│   批量生成轨迹       │────→│  梯度更新策略         │
│   高吞吐推理         │ 数据 │  显存密集反向传播      │
└─────────────────────┘     └─────────────────────┘
```

vLLM 的 PagedAttention 把 KV Cache 管理起来，使批量推理吞吐量提升 2-4x。

在 LLM RL 里，“采样”基本等价于“自回归生成”。生成的成本由两部分组成：

- prefill（把 prompt 喂进模型）：接近线性随 prompt 长度增长
- decode（逐 token 生成）：接近线性随生成长度增长，但更难 batch，通常更贵

当你把 GRPO 的 `k`（每个 prompt 采样多少条回答）和长输出（2K-32K token）叠加起来，rollout 很容易变成系统的绝对瓶颈。

生成阶段消耗 >90% 的训练时间，推理和训练必须并发运行。这涉及三种部署模式（同步 / Colocated / Disaggregated）、权重同步、数据过期管理等关键设计——这些在 **[B.2 异步训练架构](./async-training)** 中展开。

模型如何分布式地切到多张卡上，详见 **[B.3 分布式并行策略](./parallelism)**。

Agentic RL（多轮交互 + 工具调用）的基础设施需求更高，详见 **[B.4 Agentic RL 基础设施](./agentic-rl-infra)**。

## 规模化路径总结

```
单机 Gym                    游戏/仿真并行              LLM RL 集群
─────────────────────────────────────────────────────────────────────
gym.make()               AsyncVectorEnv           vLLM + FSDP
SyncVectorEnv            Isaac Gym (GPU)          Rollout-Training 分离
CPU 采样 + GPU 训练      Sample Factory           Ray 分布式编排
~1K steps/s              ~100K steps/s            ~10K prompts/hour

瓶颈：GPU 训练            瓶颈：环境交互            瓶颈：Rollout 推理
```

**选型决策**：

- 环境轻量 + 单卡能装下模型 → 阶段一（Gymnasium + CleanRL；LLM 原型可用 TRL）
- 环境重 + 需要高吞吐采样 → 阶段二（Sample Factory / Isaac Gym）
- LLM + 需要大规模 PPO/GRPO → 阶段三（OpenRLHF / veRL）

## 参考文献

[^gym_vec]: Gymnasium Documentation, [Vector Environments (SyncVectorEnv / AsyncVectorEnv)](https://gymnasium.farama.org/api/vector/).

[^impala]: Espeholt L, Soyer H, Munos R, et al. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://proceedings.mlr.press/v80/espeholt18a.html), ICML 2018.

[^sf]: Petrenko A, Huang Z, Kumar T, Sukhatme G S, Koltun V. [Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning](https://arxiv.org/abs/2006.11751), ICML 2020.

[^isaac]: Makoviychuk V, Wawrzyniak L, Guo Y, et al. [Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning](https://research.nvidia.com/labs/srl/publication/makoviychuk-2021-isaac/), NeurIPS 2021 (Datasets and Benchmarks).

[^vllm]: Kwon W, Li Z, Zhuang S, et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023. (vLLM / PagedAttention)

[^hf_async]: Hugging Face Blog, [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape), 2026.

[^pt_primer]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.

[^openrlhf]: OpenRLHF, [OpenRLHF (GitHub) + Technical Report (arXiv:2405.11143)](https://github.com/OpenRLHF/OpenRLHF).

[^hybridflow]: veRL Project, [veRL (GitHub)](https://github.com/verl-project/verl). (相关论文：HybridFlow, arXiv:2409.19256)
