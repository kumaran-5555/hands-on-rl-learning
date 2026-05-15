---
title: 'Chapter 7: PPO'
---

# Chapter 7: PPO — The Art of Stable Training

In the previous chapter, we built the [Actor-Critic architecture](../chapter06_actor_critic/actor-critic): the Actor is responsible for choosing actions, and the Critic is responsible for judging whether those actions are good or bad. The two cooperate through the [advantage function](../chapter06_actor_critic/advantage-function) $A(s,a)$. On CartPole, Actor-Critic performs quite well. But when you move the same architecture to more complex environments (for example, LunarLander) or to much larger models (for example, language models with billions of parameters), a serious issue starts to surface: **training instability**.

[Policy gradient methods](../chapter05_policy_gradient/reinforce) have a notorious weakness: if a single update step is too large, the policy can "collapse." Imagine learning to ride a bicycle. If you shift your center of gravity too aggressively in one attempt, you do not ride better, you simply crash. The [TD Error](../chapter06_actor_critic/critic-training) signal in Actor-Critic does reduce variance, but it does not fundamentally solve this problem. What we need is a mechanism that constrains how much the policy is allowed to change at each step, so the policy can "move fast in small steps" rather than "leap to the finish in one jump." This is the core problem that PPO (Proximal Policy Optimization) is designed to solve.

::: tip Prerequisites (Quick Review)
This chapter will frequently use the following concepts:

- [Policy gradient $\nabla_\theta J$](../chapter05_policy_gradient/reinforce): PPO adds constraints on top of policy gradients
- [The high-variance issue in REINFORCE](../chapter05_policy_gradient/cartpole): why we need a series of improvements
- [Advantage function $A(s,a)$](../chapter06_actor_critic/advantage-function): PPO's policy update depends on advantage signals
- [TD Error and Critic training](../chapter06_actor_critic/critic-training): how the Critic is trained in PPO
- [Actor-Critic architecture](../chapter06_actor_critic/actor-critic): PPO is a variant of Actor-Critic
  :::

This chapter follows the path "hands-on → theory → constraints → estimation." We will first run a continuous-control experiment on BipedalWalker and see the training curves, policy entropy, clipping fraction, and KL divergence with our own eyes. Then we will unpack the mathematics behind PPO: the derivation, the clipping mechanism, and methods for advantage estimation. LunarLander has already served as the introductory task in earlier chapters, so we will not repeat it here. Instead, we will move directly into continuous action spaces, where PPO's characteristics become more visible.

| Section                                                            | The Question You Will Answer                                                                                                                             |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Hands-on: BipedalWalker continuous control](./ppo-bipedal-walker) | What does PPO training look like in practice? How does it handle continuous action spaces? How should we read Reward, Entropy, Clip Fraction, and KL?    |
| [PPO: Mathematical derivation](./ppo-math)                         | Where do PPO's formulas come from? What is the full chain from policy gradients to the clipped surrogate objective? What terms are in the complete loss? |
| [Trust regions and clipping](./trust-region-clipping)              | Why does a too-large update step cause collapse? How do TRPO's KL constraint and PPO's clipping work, respectively?                                      |
| [GAE, reward models, and LLM alignment](./gae-reward-model)        | How does GAE interpolate between bias and variance? In PPO for LLM alignment, how many models need to run at the same time?                              |
| [PPO game projects](./ppo-game-benchmark)                          | Which games have already been trained with PPO? Where are the player entry points, training environments, and reproduction evidence?                     |
| [RL exploration in long-horizon tasks](./rl-long-horizon-planning) | How does classical RL deal with long-horizon tasks? How do hierarchical RL, HER, world models, and reward shaping work?                                  |

Let's start by running PPO and looking at the results: [Hands-on: BipedalWalker continuous control](./ppo-bipedal-walker).
