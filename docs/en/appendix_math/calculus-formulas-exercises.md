---
title: E.3.6 Formulas and Exercises
---

# E.3.6 Calculus and Optimization Formula Reference and Exercises

> **Prerequisite**: This page summarizes all formulas in module E.3. It is best reviewed after reading [E.3.1](./calculus-basics) through [E.3.5](./calculus-advanced-formulas). If this is your first time through the material, read the main sections first.

The previous pages moved from one-dimensional derivatives all the way to PPO clipping and GRPO normalization, introducing many formulas along the way. This page collects them for reference, points out several common misunderstandings, and ends with a few exercises to check your understanding.

---

## Optimization Formulas You Will Encounter in This Book

| Concept                       | Formula                                                                                | Meaning in reinforcement learning                        |
| ----------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Gradient ascent               | $\theta \leftarrow \theta + \alpha \nabla J(\theta)$                                   | Maximizes average return                                 |
| Gradient descent              | $\theta \leftarrow \theta - \alpha \nabla L(\theta)$                                   | Minimizes value error or model loss                      |
| Chain rule                    | $\frac{dL}{d\theta}=\frac{dL}{dy}\frac{dy}{d\theta}$                                   | Foundation of backpropagation                            |
| Policy gradient               | $\nabla J \approx G_t\nabla\log\pi_\theta(a_t\mid s_t)$                                | Good outcomes increase the probability of actions        |
| Policy gradient theorem       | $\nabla_\theta J=\sum_s d^\pi(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a)$      | How parameter changes affect average return              |
| Log-derivative trick          | $\nabla_\theta\pi=\pi\nabla_\theta\log\pi$                                             | Turns hard probability gradients into sampleable form    |
| Advantage-weighted update     | $\nabla J \approx \hat{A}_t\nabla\log\pi_\theta(a_t\mid s_t)$                          | Strengthens actions that are better than average         |
| PPO probability ratio         | $r_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}$                           | Measures the change between old and new policies         |
| Second-order Taylor expansion | $f(\theta+\Delta)\approx f(\theta)+\nabla f^\top\Delta+\frac{1}{2}\Delta^\top H\Delta$ | Understands curvature and trust regions                  |
| PPO clipping term             | $\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)$                 | Prevents overly large policy updates                     |
| GRPO group advantage          | $\hat{A}_i=\frac{r_i-\mu}{\sigma}$                                                     | Replaces a Critic with relative rewards inside the group |

---

## From Intuition to Formula: Reviewing the Thread

At this point, the whole learning path can be connected. We first started from a one-dimensional derivative and one parameter update to understand that "the gradient tells us where to move". Then the chain rule and partial derivatives extended this tool to multi-parameter networks. Next, policy gradients wrote the intuition "increase the probability of good actions" as a mathematical expression. Advantage functions made the update signal more precise. PPO clipping and Adam controlled the update step size. Finally, GRPO replaced the Critic with relative comparison inside a group. Whenever you see a complicated optimization formula, try to identify three things: what the objective function is, where the gradient direction comes from, and how the update size is controlled.

---

## Common Traps

1. **Treating the gradient as the final answer.** A gradient is only a direction. The actual update must still be multiplied by a learning rate and may pass through clipping, normalization, or Adam adjustment. Having the gradient is not the same as completing the update.
2. **Assuming larger returns always make updates safer.** High-return samples can produce overly large updates and damage the policy. PPO's probability-ratio clipping is designed precisely to prevent this.
3. **Ignoring Critic error.** The Actor's advantage estimate depends on the Critic's prediction. When the Critic is inaccurate, the policy update direction can become biased as well.

::: warning Common Pitfall
When reading optimization formulas, the three easiest mistakes are: (1) treating the gradient as the final answer and forgetting the learning rate and clipping; (2) assuming larger returns always make updates safer, while ignoring that high returns can cause policy collapse; (3) ignoring that Critic error propagates into the Actor update direction.
:::

---

## Short Exercises

1. Differentiate $J(\theta)=-(\theta-1)^2+2$, then perform one gradient-ascent step from $\theta=0$ with learning rate $0.1$.
2. The old policy probability is $0.4$, the new policy probability is $0.6$, and the advantage is $3$. What is the unclipped PPO term? If $\epsilon=0.2$, what is the clipped value?
3. If $V(s_t)=5$, $R_{t+1}=2$, $V(s_{t+1})=6$, and $\gamma=0.9$, what is the one-step TD error?
