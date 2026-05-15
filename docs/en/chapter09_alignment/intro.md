---
title: 9. Post-Training Alignment
---

# Alignment and Reasoning RL

In the previous chapter, we walked through the full [RLHF pipeline](../chapter08_rlhf/standard-rlhf-pipeline). If you actually ran that pipeline, a few numbers probably stayed with you:

- four models may live in memory at the same time: Actor, Reference, Critic, and Reward Model
- each training round needs many on-policy generations
- the Reward Model directly determines alignment quality
- reward hacking must be monitored constantly

Now step back and ask a more fundamental question:

**Among these four models, can any component be removed?**

This question is not about cutting corners. It was one of the central industrial questions from 2023 to 2025. Every removed component means less memory, shorter iteration cycles, and a simpler engineering pipeline. But the answer goes far beyond saving resources. It reshapes how we understand LLM post-training.

## Three Questions, Three Answers

Let us follow the idea of "removing components" step by step.

**The first component to remove is the Reward Model.**

In RLHF, the Reward Model translates human preference judgments ("answer A is better than answer B") into scalar scores. DPO (Direct Preference Optimization, 2023) shows a clever mathematical fact: preference data already contains an implicit reward signal. We do not need to train a separate RM to translate it. By changing the loss function, we can train directly on preference pairs and obtain an objective equivalent to KL-regularized RLHF. This turns the four-model setup into a two-model setup.

**The second component to remove is the Critic.**

PPO needs a Critic to estimate advantages. But in LLM training, the Critic can be almost as large as the Actor, doubling memory pressure. GRPO (Group Relative Policy Optimization, 2025) asks: why maintain a separate Critic at all? For the same prompt, generate a group of responses, then normalize rewards within the group using the mean and standard deviation. This group-relative statistic can replace the Critic's baseline estimate.

**The third component to remove is human labeling itself.**

RLHF preference data and DPO chosen/rejected pairs usually require humans or stronger models to judge which response is better. But math problems have final answers, code has unit tests, and many reasoning tasks have verifiable conclusions. In these domains, we can use rules as the judge. RLVR (Reinforcement Learning with Verifiable Rewards, 2025) replaces human preference labels with rule-based verification and pushes labeling cost close to zero.

These three routes share one deeper insight:

**The core of RL is not PPO itself. The core is where the training signal comes from.**

RLHF uses human preference as the signal. DPO shows that preference pairs can encode the signal directly. GRPO removes a signal-processing component. RLVR changes the signal source entirely. Once you understand this evolution of training signals, you understand the main thread of this chapter.

## From Alignment to Reasoning

These methods were originally designed for alignment: making models safer, more helpful, and more consistent with human preferences. But research in 2024-2025 revealed a surprising fact:

**The same RL framework is even more powerful when applied to mathematical reasoning and code generation.**

DeepSeek-R1 is the representative example. RLVR training did not merely improve math accuracy. It also led to emergent chain-of-thought-like behavior: self-checking, backtracking, stepwise reasoning, and revision. These behaviors were not explicitly written into supervised training examples.

This suggests that RL is not only an alignment tool. It may also be a catalyst for reasoning ability.

We can separate two post-training goals:

| Goal                                                     | Core methods | Training signal                             |
| -------------------------------------------------------- | ------------ | ------------------------------------------- |
| **Alignment**: make the model safe and useful            | DPO / RLHF   | human preferences (chosen vs rejected)      |
| **Reasoning reinforcement**: make the model think better | GRPO / RLVR  | verifiable rewards (is the answer correct?) |

This chapter covers both. The first part focuses on DPO and its family. The second part focuses on GRPO, RLVR, and DAPO. They share the same underlying RL mathematics, but their signal sources and engineering details are very different.

## Relationship to RLHF: Complement, Not Replacement

A common misunderstanding is "DPO replaces RLHF" or "GRPO is simply better than PPO." The reality is more nuanced:

- **RLHF** is the general solution: with preference data, it can align many kinds of behavior. The cost is high engineering complexity.
- **DPO** is a lightweight alternative: when preference data is high-quality and the task is relatively simple, it can reach similar outcomes with much lower cost. But its offline nature becomes a bottleneck when the task needs online exploration.
- **GRPO / RLVR** open a different route: instead of preference alignment, they use rule-based verification to strengthen reasoning. They are not a competitor to RLHF; they are often complementary.

In industrial practice, these methods are usually combined. A system may use SFT and RLHF for basic alignment, DPO for efficient preference optimization, and GRPO/RLVR for reasoning improvement. The important skill is not memorizing one method as "the best", but knowing when each method fits.

## Chapter Roadmap

| Section                                                                         | Core question                                                                                                   | What you will gain                                               |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [DPO theory and method selection](./dpo-theory-and-family)                      | How does DPO derive a classification loss from a KL-regularized RL objective? When should we use DPO/KTO/SimPO? | Understand DPO mathematically and choose variants by scenario    |
| [Hands-on: DPO alignment](./dpo-hands-on)                                       | How do we read DPO metrics such as reward accuracy, margin, and beta sensitivity?                               | Diagnose DPO training logs                                       |
| [Hands-on: GRPO mechanisms](../chapter09_grpo_rlvr/grpo-practice-and-mechanism) | How does GRPO replace the Critic with group normalization? How much memory does it save?                        | Run GRPO training and understand the difference from PPO         |
| [DeepSeek-R1 and DAPO](../chapter09_grpo_rlvr/deepseek-dapo)                    | Can pure RL work without SFT? What do SimpleRL and DAPO teach us?                                               | Understand R1-Zero, SimpleRL reproduction, and DAPO improvements |
| [RLVR: verifiable rewards](../chapter09_grpo_rlvr/rlvr)                         | Can verifiable rewards replace reward models? Why does 1-shot RLVR work?                                        | Understand verifier design and RLVR training                     |
| [RL scaling outlook](../chapter12_future_trends/rl-scaling-outlook)             | When should we choose online vs offline RL? Where are the scaling limits?                                       | Build global judgment about RL training paradigms                |
| [On-policy distillation](../chapter09_grpo_rlvr/on-policy-distillation)         | Why can distillation be more effective than RL for small models? How can teacher log-probs act like rewards?    | Understand the RL nature of distillation                         |
| [Industrial post-training practice](./industrial-post-training)                 | How do major labs combine SFT, RLHF, DPO, RLVR, and Agentic RL in real systems?                                 | Connect papers to production post-training pipelines             |

Ready? We begin with the mathematical structure of DPO:
[DPO theory, mathematics, and method selection](./dpo-theory-and-family).
