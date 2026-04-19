# 第 9 章：Agentic RL——工具调用、多轮交互与智能体训练

从第 4 章的 DQN 到第 8 章的 GRPO，我们一直在处理"单轮"的 RL 问题：输入一个 prompt，模型输出一段完整的回答，Reward Model 打一个分数，然后更新策略。这个范式在过去两年里被证明非常有效——ChatGPT、Claude、DeepSeek 都是这样训练出来的。

但真正的智能体不是这样工作的。当你让一个 Agent "帮我查一下明天北京的天气，然后根据天气安排行程"，它需要：先调用搜索工具查天气，再读取搜索结果，然后根据结果决定是否需要进一步查询，最后把所有信息整合成一份行程建议。这是一个**多步、多工具、多轮交互**的过程。前面的单轮 RL 范式在这里完全失效——你不能在每一步都给一个 Reward，因为中间步骤本身没有"对错"之分，只有最终结果能告诉你整条路径是否正确。

## 两个范式：传统 LLM RL vs Agentic RL

Zhang et al. 在综述 [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547) 中，对这一范式转变做了系统性的形式化。他们指出，过去大部分 LLM 强化微调（PBRFT，即 Preference-Based Reinforcement Fine-Tuning）本质上是一个**退化的 MDP**：

$$
\langle S_{\text{trad}},\ A_{\text{trad}},\ P_{\text{trad}},\ R_{\text{trad}},\ T=1 \rangle
$$

状态空间只有一个 prompt（$S = \{s_0\}$），动作空间是纯文本（$A = A_{\text{text}}$），episode 只做一步决策就结束。优化目标是 $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$——让单轮输出尽可能好。

而 **Agentic RL** 则被建模为一个**部分可观测马尔可夫决策过程（POMDP）**：

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle
$$

这里的核心变化体现在下表中：

|              | 传统 LLM RL（PBRFT）                                     | Agentic RL                                                                             |
| ------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **状态空间** | 单个 prompt，episode 立即结束                            | 环境状态 $s_t$ 随交互动态演化，agent 只看到部分观测 $o_t = O(s_t)$                     |
| **动作空间** | 纯文本序列 $A_{\text{text}}$                             | 文本 + 结构化操作 $A = A_{\text{text}} \cup A_{\text{action}}$（工具调用、环境交互等） |
| **状态转移** | 确定性终止，$P(s_1 \| s_0, a) = 1$                       | 动态转移 $s_{t+1} \sim P(s_{t+1} \| s_t, a_t)$，环境充满不确定性                       |
| **奖励信号** | 单步标量 $r(a)$，无中间反馈                              | 步级奖励，可能是稀疏的任务完成信号，也可以是密集的子任务奖励                           |
| **优化目标** | $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$，优化单轮输出质量 | $\mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t R(s_t, a_t)]$，优化多步交互策略     |

这个形式化带来的核心洞察是：**Agentic RL 的创新重点，很多时候不在"RL 公式本身"，而在"让 RL 能作用于真实 agent loop 的系统设计"**——怎么定义状态和动作、怎么设计 reward、怎么搭环境、怎么处理长时程 credit assignment。

## 为什么只靠 SFT / Prompting 不够？

你可能会有疑问：ReAct、Toolformer 这些方法已经能让 LLM 调用工具了，为什么还需要 RL？

关键区别在于：SFT 和 prompting 教会模型的是**模仿**——复制人类演示中"什么时候调工具、调什么工具"的模式。但真实 agent 任务中，工具使用的最优策略高度依赖上下文：

- 搜索 query 怎么写？什么时候该点开网页？什么时候该停止搜索开始总结？
- 代码修改了一个地方，测试仍然不通过，是继续调试还是换一个方向？
- 多个来源信息冲突，该信任哪一个？

这些问题本质上是**策略学习问题**，不是单纯的语言建模问题。演示数据很难覆盖所有路径，而 RL 可以根据任务结果反过来塑造工具调用、规划、记忆管理等行为习惯。正如 Zhang et al. 所强调的：RL 在 agent 时代的价值，不只是对齐，而是**把语言模型变成行为主体**。

## 本章结构

Zhang et al. 的综述从两个维度组织了 Agentic RL 的研究版图：一条按**核心能力**（规划、工具使用、记忆、推理、自我改进、感知），另一条按**任务领域**（搜索代理、代码代理、数学代理、GUI 代理、具身代理、多智能体）。本章将沿着实战路线展开，把能力与任务交织在一起：

::: tip 前置知识
本章会频繁用到以下概念，建议先复习：

- [GRPO 与 RLVR](../chapter08_grpo_rlvr/intro)——"可验证奖励"是 Agentic RL 的天然选择
- [PPO 与奖励模型](../chapter06_ppo/intro)——策略优化的基础框架
  :::

| 小节                                                                | 核心问题                                         |
| ------------------------------------------------------------------- | ------------------------------------------------ |
| [动手：Mini Agent Loop——ORM vs PRM 对比](./agent-loop-hands-on)     | 多轮交互中每一步怎么给奖励？ORM 和 PRM 差异有多大？ |
| [多轮交互 RL 与信用分配](./multi-turn-rl)                           | 7 轮交互失败了，该怪谁？ORM vs PRM；规划能力     |
| [轨迹合成与数据工程](./trajectory-synthesis)                        | 训练数据从哪来？拒绝采样、图谱合成、闭环迭代     |
| [工具调用 RL：Web Agent、Code Agent 与搜索推理](./tool-use-agents)  | SFT 教格式，RL 教策略——SearchR1 与工具调用       |
| [Agentic RL 工程实战与总结](./agentic-engineering)                  | 沙箱、基础设施、框架选择——把 Agentic RL 跑起来   |
| [工业界实战：各家的 Agentic RL 都怎么做的？](./industrial-practice) | 真实工业场景中的 Agentic RL 实践                 |
| [Agentic 评测体系与 Benchmark 全景](./evaluation-benchmarks)        | 怎么评估 Agent 的好坏——评测基准与 Badcase 分析   |
| [深度研究智能体：Deep Research Agent](./deep-research-agent)        | 从搜索推理到报告生成——Agentic RL 的前沿综合应用  |
| [延伸阅读索引](./extended-readings)                                 | 13 个主题板块、120+ 篇论文的开源索引——按兴趣深入 |

准备好了吗？让我们先动手搭一个 Mini Agent Loop，亲手感受多轮信用分配的挑战——[动手：Mini Agent Loop](./agent-loop-hands-on)。
