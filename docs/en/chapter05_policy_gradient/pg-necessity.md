---
title: 5.1 Why Policy Gradients
---

# 5.1 Why Policy Gradients

## Reading Guide

**Key ideas**

- Recall the core idea of DQN in Chapter 4: learn $Q(s,a)$, then choose actions via $\arg\max$.
- Understand the fundamental limitation of value-based methods: they can only handle a finite set of discrete actions.
- See why policy-based methods learn $\pi_\theta(a|s)$ directly, and how the two routes differ in action spaces, exploration mechanisms, and data reuse.

## What DQN Got Right

In Chapter 4, DQN follows a clean route: use a neural network to approximate $Q(s,a)$, score every action, and then pick the best one via $\arg\max_a Q(s,a)$. On tasks like CartPole (2 actions) and LunarLander (4 actions), this works well because the action set is small: you can simply compare the $Q$ values for all actions.

The underlying logic is: instead of learning "what to do" directly, you first learn "how good each action is," and only then choose the best. The policy is implicit: it is hidden inside the $\arg\max$ over the $Q$ values.

## Where $\arg\max$ Breaks Down

The $\arg\max$ rule requires you to compare $Q$ values for all possible actions. As long as the number of actions is finite, this is fine. But many real-world tasks have **continuous** action spaces, where there are infinitely many actions.

Robotic arm control is a canonical example. The shoulder, elbow, and wrist joints may each have multiple degrees of freedom, and for each degree you can apply a continuous torque in a range like $[-10, 10]$. If we have 6 joints, the action space becomes $[-10, 10]^6$: infinitely many points in a six-dimensional continuous space. You cannot compute a $Q$ value for every point, and you certainly cannot take an $\arg\max$ over infinitely many points.

Large language model text generation faces a similar issue from another angle. At each step you sample from a vocabulary of tens of thousands of tokens; the "policy" is a probability distribution. A greedy $\arg\max$ decoder exists, but it always outputs the single most probable token and destroys diversity. Good text generation often needs "sometimes choose the second best, sometimes the third best," which is exactly the logic of sampling from a probability distribution, not scoring-and-argmax.

## Learning the Policy Directly

If "score-then-pick" cannot be made to work, we switch routes: **skip the $Q$ function and learn a policy $\pi_\theta(a|s)$ directly**. Instead of asking "how many points is this action worth?", we learn "in this situation, what should we do?"

This is the central idea of Chapter 3, [Route 2: the policy objective $J(\theta)$](../chapter03_mdp/policy-objective): define a policy objective function $J(\theta)$ and then optimize the parameters $\theta$ to maximize $J(\theta)$ directly.

One analogy makes the difference vivid. Value-based methods are like a food critic: they rate every dish and then choose the highest-rated one. Policy-based methods are like an experienced chef: they do not need to score every option; they directly know what to cook given the ingredients and the occasion.

## Two Routes, Different Tradeoffs

|                           | Value-Based (DQN)                            | Policy-Based (Policy Gradient)                      |
| ------------------------- | -------------------------------------------- | --------------------------------------------------- | ---------------------------------- |
| What it learns            | $Q(s,a)$: how good each action is            | $\pi\_\theta(a\\                                    | s)$: a probability for each action |
| How actions are chosen    | $\arg\max_a Q(s,a)$ (take the highest score) | Sample from $\pi\_\theta(\\cdot\\                   | s)$                                |
| Policy form               | Deterministic (always pick the best)         | Stochastic (outputs a distribution)                 |
| Action space              | Discrete only                                | Discrete + continuous                               |
| Exploration               | Added externally ($\\varepsilon$-greedy)     | Built in (a distribution naturally explores)        |
| Data reuse                | Off-policy (replay buffer reuses old data)   | On-policy (must use fresh data from current policy) |
| Variance                  | Low (TD targets are relatively stable)       | High (Monte Carlo returns can be noisy)             |
| Representative algorithms | DQN (Chapter 4)                              | REINFORCE (this chapter) $\\to$ PPO (Chapter 7)     |

Let's explain the key differences row by row.

**Action space.** This is often the decisive factor when choosing a route. DQN's $\arg\max$ is fundamentally hard to compute in continuous spaces. Policy gradients output a probability distribution directly: for discrete actions you can use a Softmax; for continuous actions you can output, for example, a Gaussian distribution. In many cases you can switch action spaces simply by changing the output head.

**Exploration.** DQN's policy is deterministic (it always takes the $\arg\max$), so exploration must be injected from outside via $\\varepsilon$-greedy (review: [the three components of DQN](../chapter04_dqn/dqn-components)). The $\\varepsilon$ schedule must be tuned by hand: too large wastes samples, too small fails to explore. A policy gradient method naturally outputs a distribution, so exploration is built in. If the network assigns 30% probability to an action, it will try it 30% of the time.

**Data reuse.** This is the most practical engineering difference. DQN is off-policy: the replay buffer stores old experience that can be reused many times. Policy gradients are on-policy: the expectation $\\mathbb{E}_{\\pi_\\theta}$ in the gradient estimator requires data generated by the current policy. Once the policy updates, old data no longer matches it. This makes policy gradients inherently less data-efficient than DQN, and it is the biggest engineering drawback of the plain policy-gradient route.

## The Two Routes Are Not Enemies

Each route has strengths and weaknesses, but they are not mutually exclusive. In Chapter 6, Actor-Critic methods combine them: a policy network makes decisions, while a value network reduces variance. Before we get there, we need to build a solid mathematical foundation for the policy-based route.

In the next section, we start from the policy objective, derive the policy gradient theorem, and arrive at the REINFORCE algorithm: [REINFORCE](./reinforce).
