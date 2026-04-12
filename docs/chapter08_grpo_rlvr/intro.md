# 第 8 章：GRPO、DAPO 与 RLVR——从策略优化到可验证奖励

第 6 章我们学到了 PPO——当前最主流的 on-policy RL 算法，但它需要四个模型同时跑（Actor、Critic、Reference、RM），工程复杂度极高。第 7 章我们看到 DPO 绕过了 RM，把四模型简化为两模型，但它是 offline 方法——不能在线探索，上限受数据质量限制。

现在我们要问一个更激进的问题：**能不能把 Critic 也去掉？能不能连 RM 都不要？**

2025 年，DeepSeek 给出了答案。GRPO（Group Relative Policy Optimization）说："Critic 太占显存了，我用组内的均值和标准差自己算基线，干掉 Critic 网络。"RLVR（Reinforcement Learning with Verifiable Rewards）说："RM 的标注成本太高了，数学题有标准答案、代码有测试用例，直接用规则验证，不需要人工打分。"整个逻辑链条完美闭环——从 PPO 的四模型到 GRPO 的两模型，从 RLHF 的人工标注到 RLVR 的自动验证，RL 的训练成本和工程复杂度被一步步压缩到了极致。

本章沿着"动手 → 机制 → 前沿 → 展望"的路径展开。先在 GSM8K 数学题上跑一个 GRPO 实验，再深入理解组内归一化的原理，然后追踪 DeepSeek-R1-Zero 和 DAPO 的最新进展，最后展望 RL Scaling 和 Test-time Scaling 的未来方向。

| 小节                                                   | 你会回答的问题                                                             |
| ------------------------------------------------------ | -------------------------------------------------------------------------- |
| [动手：GRPO 训练数学推理](./grpo-hands-on)             | GRPO 训练过程是什么样的？省掉了 Critic 后显存能省多少？                    |
| [GRPO 核心机制](./grpo-mechanism)                      | 组内归一化为什么能替代 Critic？k 值怎么选？                                |
| [DeepSeek-R1-Zero、DAPO 与 RLVR](./deepseek-dapo-rlvr) | 纯 RL 训练不用 SFT 行不行？可验证奖励能取代 RM 吗？                        |
| [RL Scaling 与未来展望](./rl-scaling-outlook)          | Online vs Offline 怎么选？RL Scaling 的天花板在哪？PRM 和 ORM 有什么区别？ |

准备好了吗？让我们从 GRPO 的动手实验开始——[动手：GRPO 训练数学推理](./grpo-hands-on)。
