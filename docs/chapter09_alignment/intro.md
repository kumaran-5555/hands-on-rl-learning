# 第 9 章：对齐与推理强化（DPO / GRPO / RLVR）

上一章我们走完了 [RLHF 的完整流水线](../chapter08_rlhf/standard-rlhf-pipeline)。如果亲自动手跑过那条流水线，你一定对几个数字印象深刻：四个模型同时驻留显存（Actor、Ref、Critic、Reward Model）、训练一轮要生成大量 on-policy 回答、Reward Model 的质量直接决定对齐效果、还得时刻盯着 reward hacking 别跑偏。

现在退一步，问自己一个更根本的问题：**这四个模型里，有没有哪个是"可以被省掉的"？**

这个问题不是偷懒——而是工业界在 2023-2025 年间一直在追问的核心问题。每省掉一个组件，就意味着更少的显存、更短的开发周期、更简单的工程管线。而答案，远远超出了"省一点资源"的范畴——它重新定义了 LLM 后训练的范式。

## 三个问题，三个答案

让我们沿着这个"省零件"的思路，逐步拆解：

**第一个被省掉的是 Reward Model。** RLHF 中，Reward Model 的作用是把人类的偏好判断（"回答 A 比回答 B 好"）转化成标量分数。但 DPO（Direct Preference Optimization，2023）发现了一个巧妙的数学事实：偏好数据本身就隐含了奖励信号，不需要额外训练一个 RM 来"翻译"。只需要改一个损失函数，直接在偏好对上训练策略模型，效果和 RLHF 等价。这一步把"四模型"变成了"两模型"。

**第二个被省掉的是 Critic。** PPO 需要 Critic 网络来估计优势函数（回顾[第 6 章优势函数](../chapter06_actor_critic/advantage-function)：$A_t = \sum \lambda^l \delta_{t+l}$），而 Critic 是一个和 Actor 差不多大的网络，显存开销直接翻倍。GRPO（Group Relative Policy Optimization，2025）说：何必单独养一个 Critic？对同一个 prompt 生成一组回答，用组内的均值和标准差做归一化，就能替代 Critic 的基线估计。这一步把"两模型"进一步精简。

**第三个被省掉的是人工标注本身。** 无论是 RLHF 的偏好数据还是 DPO 的 chosen/rejected 对，都需要人类（或强模型）来判断"哪个回答更好"。但数学题有标准答案，代码有测试用例，逻辑推理有可验证的结论——这些领域的"好坏判断"根本不需要人类参与。RLVR（Reinforcement Learning with Verifiable Rewards，2025）直接用规则引擎当裁判，把标注成本降到了接近零。

三条路线不是孤立的。它们共享一个深层洞察：**RL 的核心不是 PPO 算法本身，而是训练信号从哪来**。RLHF（[上一章](../chapter08_rlhf/ppo-rlhf-loop)）用人类偏好当信号，DPO 发现偏好数据本身就能编码信号，GRPO 省掉了信号处理中的一个组件，RLVR 则彻底换了一种信号源。理解了这个"信号源"的演进逻辑，你就掌握了本章的主线。

## 从对齐到推理：一个意外的发现

上面三条路线最初都是为"对齐"设计的——让模型更安全、更有用、更符合人类偏好。但 2024-2025 年的研究揭示了一个意外：**同样的 RL 训练框架，用在数学推理和代码生成上，效果远超对齐**。

DeepSeek-R1 的实验最具代表性：他们发现，用 RLVR 训练的模型不只在数学题上表现更好，还自发涌现出了"思维链"（Chain-of-Thought）行为——模型学会了自我验证、回头修正、分步推理，这些能力从未出现在训练数据中。这意味着 RL 不只是"对齐工具"，更可能是"推理能力的催化剂"。

这个发现把 LLM 后训练分成了两个目标：

| 目标                         | 核心方法    | 训练信号                       |
| ---------------------------- | ----------- | ------------------------------ |
| **对齐**（让模型安全、有用） | DPO / RLHF  | 人类偏好（chosen vs rejected） |
| **推理强化**（让模型会思考） | GRPO / RLVR | 可验证奖励（答案对不对）       |

本章两部分都会覆盖。前半部分（DPO 及家族）侧重对齐，后半部分（GRPO / RLVR / DAPO）侧重推理强化。两者共享底层的 RL 数学，但训练信号和工程细节截然不同。

## 与上一章的关系：互补而非替代

一个常见的误解是"DPO 取代了 RLHF"或"GRPO 比 PPO 更好"。实际情况更微妙：

- **RLHF（[上一章](../chapter08_rlhf/intro)）** 是"全能型"方案——只要有偏好数据，它能对齐任何能力。代价是工程复杂度高。
- **DPO** 是 RLHF 的"轻量替代"——在偏好数据质量高、任务简单的场景下，效果相当但成本低得多。但当任务复杂到需要在线探索时，DPO 的离线特性会成为瓶颈。
- **GRPO / RLVR** 则开辟了一条全新的路线——不做偏好对齐，而是用规则验证来强化推理。它和 RLHF 不是竞争关系，而是互补关系：先用 RLHF 对齐，再用 GRPO/RLVR 强化推理。

工业界的实际做法通常是组合使用。DeepSeek-R1 先用 SFT+RLHF 做基础对齐，再用 RLVR 做推理强化。Qwen3 类似，但对齐阶段用 DPO 代替了 RLHF（因为 DPO 更省资源），推理阶段用 GRPO。理解了每种方法的适用边界，你就能为具体场景做出正确的选择。

## 本章路线图

| 小节                                                                                  | 核心问题                                                                      | 你会获得的能力                                |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------- |
| [DPO 原理、数学与选型](./dpo-theory-and-family)                                       | DPO 是怎么从 RL 目标推导出分类损失的？DPO/KTO/SimPO 各适合什么场景？          | 理解 DPO 的数学推导，能为场景选择合适的变体   |
| [动手：DPO 对齐实验](./dpo-hands-on)                                                  | DPO 训练的指标怎么看？Reward Accuracy、Margin、β 敏感性分别意味着什么？       | 能读懂 DPO 训练日志，诊断训练问题             |
| [动手：GRPO 训练与核心机制](../chapter09_grpo_rlvr/grpo-practice-and-mechanism)       | GRPO 怎么用组内归一化替代 Critic？省掉 Critic 后显存能省多少？                | 能跑通 GRPO 训练，理解它和 PPO 的关键差异     |
| [DeepSeek-R1 与 DAPO](../chapter09_grpo_rlvr/deepseek-dapo)                           | 纯 RL 训练不用 SFT 行不行？开源 SimpleRL 实验说明了什么？DAPO 怎么优化 GRPO？ | 理解 R1-Zero、SimpleRL 复现和 DAPO 的工程改进 |
| [RLVR：可验证奖励](../chapter09_grpo_rlvr/rlvr)                                       | 可验证奖励能取代 RM 吗？1-Shot RLVR 为什么能工作？                            | 理解 RLVR 的训练范式和验证器设计              |
| [动手：金融 API 工具调用 GRPO](../chapter09_grpo_rlvr/financial-tool-calling-grpo)    | 企业 API 调用如何构造成可验证奖励？小模型怎样通过 GRPO 学会稳定调用工具？     | 能设计 tool-calling verifier 和评估指标       |
| [RL Scaling 与前沿展望](../chapter12_future_trends/rl-scaling-outlook)                | Online vs Offline 怎么选？RL Scaling 的天花板在哪？                           | 建立对 RL 训练范式的全局判断力                |
| [知识蒸馏——从大模型到小模型的知识迁移](../chapter09_grpo_rlvr/on-policy-distillation) | 蒸馏为什么对小模型比 RL 更有效？Teacher 的 log-prob 怎么当 reward？           | 理解蒸馏的 RL 本质和工业界实践                |
| [工业界后训练实践全景](./industrial-post-training)                                    | 国内外大厂在 SFT、RLHF、DPO、RLVR、Agentic RL 上到底怎么落地？                | 能把论文算法和真实后训练系统对上号            |

准备好了吗？让我们先从 DPO 的数学结构开始——[DPO 原理、数学与选型](./dpo-theory-and-family)。
