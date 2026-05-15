---
title: 6. Actor-Critic
---

# Chapter 6: Actor-Critic, Where Two Lines of Thought Converge

Chapter 4 followed Line 1 (Value-Based): learn $Q(s,a)$ and pick the action with the highest score (review: [Q(s,a) and the Greedy Policy](../chapter03_mdp/value-q)). This tends to produce accurate scoring, but it is not good at exploration, and it can only handle discrete actions. Chapter 5 followed Line 2 (Policy-Based): directly optimize $J(\\theta)$ (review: [Policy Objective](../chapter03_mdp/policy-objective)). This is good at exploration and supports continuous actions, but its variance is too large: run the same policy twice, and the gradient estimates can be wildly different.

At the end of the previous chapter, we found a key clue: subtracting a baseline reduces variance (review: [Policy Gradient Improvements](../chapter05_policy_gradient/pg-improvements)), and the best baseline is $V(s)$ (review: [State-Value Function](../chapter03_mdp/value-bellman)). But $V(s)$ itself must be learned, which means we need a dedicated network to estimate it. This network is the **Critic**.

In this chapter, we will stitch the two lines together: train a Critic using the methods from Line 1 to evaluate how good an action is, and train an Actor using the methods from Line 2 to choose actions. This is the **Actor-Critic architecture**.

::: tip Prerequisites (Quick Review)
This chapter is a synthesis of everything we have built so far. The following concepts will appear repeatedly:

- [State-value $V(s)$ and the Bellman equation](../chapter03_mdp/value-bellman): the theoretical foundation of the Critic. $V^\\pi(s)$ measures "starting from state $s$, how many points do we get on average?"
- [Action-value $Q(s,a)$](../chapter03_mdp/value-q): the difference between $Q$ and $V$ is the advantage function
- [DP / MC / TD: three ways to estimate values](../chapter03_mdp/dp-mc-td): three concrete strategies for training the Critic
- [TD Error $\\delta = r + \\gamma V(s') - V(s)$](../chapter03_mdp/dp-mc-td): the Critic's core training signal
- [Policy objective $J(\\theta)$ and policy gradients](../chapter03_mdp/policy-objective): the Actor's optimization target
- [REINFORCE and baselines](../chapter05_policy_gradient/pg-improvements): why we need $V(s)$ as a baseline
  :::

## Chapter Roadmap

| Section                                                      | Core Question                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| [Advantage Function](./advantage-function)                   | What is the advantage function $A(s,a)$? Why is it better than using $G_t$ directly? |
| [Training the Critic](./critic-training)                     | How do we train a Critic to estimate $V(s)$? Concrete implementations of DP/MC/TD    |
| [Actor-Critic Architecture](./actor-critic)                  | How do the Actor and Critic collaborate? How does TD Error replace $G_t$?            |
| [Frontier-Scale Applications of Actor-Critic](./ac-frontier) | AlphaStar, SAC robots, Isaac Lab: how AC lands in industrial-scale practice          |
| [Hands-on: Pendulum Balancing](./pendulum)                   | How does Actor-Critic handle continuous action spaces?                               |
| [Hands-on: BipedalWalker](./bipedalwalker)                   | Can Actor-Critic learn complex continuous control?                                   |

Let’s begin with the advantage function. It is the bridge that connects the Actor and the Critic: [Advantage Function](./advantage-function).
