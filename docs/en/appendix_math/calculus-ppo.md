---
title: E.3.3 PPO and Adam
---

# E.3.3 PPO Clipping, Adam, and Update Stability

> **Prerequisite**: [E.3.2 Policy Gradients and Advantage Functions](./calculus-policy-gradient). You need to know policy gradients and advantage functions.

---

## The Intuition Behind PPO Clipping

Policy gradients tell us to raise the probability of good actions and lower the probability of bad actions, but they do not tell us how much to adjust each time. If one update is too aggressive, the policy may be distorted completely. Learning to ride a bicycle gives a useful analogy: turning the handlebar too sharply can be more dangerous than not turning at all. PPO is designed to prevent this situation. It first focuses on one quantity, the probability ratio:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

Consider a concrete number. The old policy selected an action with probability $0.2$, and the new policy changes that probability to $0.3$:

$$
r_t=\frac{0.3}{0.2}=1.5.
$$

This means the new policy is $1.5$ times as inclined to choose that action as the old policy. If the advantage is $\hat{A}_t=4$, the unclipped objective term is

$$
r_t\hat{A}_t=1.5\times4=6.
$$

But PPO restricts the probability ratio to a range. Suppose the clipping range is $[0.8,1.2]$. Then $1.5$ is compressed to $1.2$:

$$
\mathrm{clip}(r_t,0.8,1.2)\hat{A}_t=1.2\times4=4.8.
$$

PPO acknowledges that this action is good and is willing to increase its probability, but it does not allow one sample to cause an excessive change. The final `min` chooses the more conservative value between the clipped and unclipped versions.

---

## Adam: An Optimizer That Adapts the Step Size

So far, our tools for controlling update size have been the learning rate and PPO clipping. In real training, however, gradient magnitudes often fluctuate sharply. In the same batch of data, some parameters may have very large gradients while others have very small ones. If all parameters share the same learning rate, parameters with large gradients may update too aggressively, while parameters with small gradients may barely move. The Adam optimizer is designed to solve this problem of uneven step sizes.

PPO clipping controls the size of policy changes, while the Adam optimizer controls the step size of gradient updates. Ordinary gradient descent updates parameters using only the current gradient, but gradients often jump between large and small values, which can make training unstable. Adam maintains two statistics at the same time:

- First moment: a moving average of the gradient, representing the common direction.
- Second moment: a moving average of the squared gradient, representing the scale of fluctuation.

A simplified way to understand Adam is this: if the gradient of a parameter has recently been consistently positive, Adam becomes more confident about updating in the positive direction. If the gradient is large one moment and small the next, or if the direction is unstable, Adam automatically reduces the effective step size to avoid moving off course.

Gradient noise in reinforcement learning is usually larger than in supervised learning, because the training data is not fixed. It comes from interaction between a changing policy and the environment. For this reason, Adam, gradient clipping, and learning-rate schedules are almost standard equipment in RL training.

---

## Summary

This page discussed two ways to control update stability:

| Tool         | What it controls          | Core idea                                                                           |
| ------------ | ------------------------- | ----------------------------------------------------------------------------------- |
| PPO clipping | Policy change size        | Restrict the probability ratio so the policy does not move too far in one update    |
| Adam         | Gradient update step size | Use first and second moments to adapt the effective learning rate of each parameter |

Used together, PPO limits policy drift at a macroscopic level, while Adam smooths gradient noise at a microscopic level. The next page examines the derivation skeleton behind policy gradients and PPO.

> **Next**: [E.3.4 Derivation Tools: Log Trick and Taylor](./calculus-derivations), which explains the derivation skeleton behind policy gradients and PPO.
