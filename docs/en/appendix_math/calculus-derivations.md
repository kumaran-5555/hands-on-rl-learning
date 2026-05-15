---
title: E.3.4 Derivation Tools
---

# E.3.4 Policy Gradient, Taylor, and GRPO Derivations

> **Prerequisite**: [E.3.2 Policy Gradients and Advantage Functions](./calculus-policy-gradient). You need to know the basic form of the policy gradient.

---

## How the Log-Derivative Trick Leads to Policy Gradients

So far, we have been using the conclusion of policy gradients. Now let us see how it is derived. Directly taking the gradient of the policy probability $\pi_\theta(a\mid s)$ is often inconvenient: $\pi$ may be produced by complicated functions such as softmax, making the gradient expression cumbersome. The **log-derivative trick** converts the difficult $\nabla_\theta \pi$ into the easier $\pi \cdot \nabla_\theta \log \pi$, so the gradient can be estimated by sampling without knowing the environment transition probabilities. The most common form of the policy gradient is derived from this trick:

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\pi[
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
].
$$

Here $\mathbb{E}_\pi[\cdot]$ means "the expectation when sampling according to policy $\pi$". In other words, it is a weighted average over all possible state-action pairs, where the weights are the probabilities with which the policy chooses each action. The key to deriving this formula is a simple identity:

$$
\nabla_\theta \log \pi_\theta(a\mid s)
=
\frac{\nabla_\theta \pi_\theta(a\mid s)}
{\pi_\theta(a\mid s)}.
$$

Multiplying both sides by $\pi_\theta(a\mid s)$ gives an equivalent but more useful form:

$$
\nabla_\theta \pi_\theta(a\mid s)
=
\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s).
$$

The benefit of this transformation is that directly differentiating $\pi_\theta$ is often hard, while the gradient of $\log\pi_\theta$ is usually much simpler. Next, substitute this trick into the objective function. In a discrete action space, the objective can be written as

$$
J(\theta)=\sum_a \pi_\theta(a\mid s)Q^\pi(s,a).
$$

When taking the gradient with respect to the parameters, $Q^\pi(s,a)$ does not depend on $\theta$; only $\pi_\theta$ contains $\theta$:

$$
\nabla_\theta J(\theta)
=\sum_a \nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a).
$$

Substitute the log-derivative trick and replace $\nabla_\theta\pi_\theta$:

$$
\nabla_\theta J(\theta)
=\sum_a
\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)
Q^\pi(s,a).
$$

Look carefully at this summation. Every term contains $\pi_\theta(a\mid s)$ as a weight. This is exactly a weighted average when sampling according to the policy, that is, an expectation:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
[
\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)
].
$$

The expression above considers only one state. If we take a weighted average over all states, where $d^\pi(s)$ is the frequency of visiting state $s$ under policy $\pi$, we obtain the full policy gradient theorem:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)Q^\pi(s_t,a_t)
].
$$

In practical algorithms, $Q^\pi(s_t,a_t)$ is difficult to know exactly, so it is commonly replaced by a sampled cumulative return $G_t$ or an advantage estimate $\hat{A}_t$:

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t
].
$$

This is the shared gradient structure behind algorithms such as REINFORCE, Actor-Critic, and PPO.

---

## Taylor Expansion, the Hessian, and PPO's Second-Order Intuition

Gradient descent looks only at the first derivative, the slope at the current position, and then takes one step along that slope. But if the step is too large, the first-order approximation becomes unreliable: you may think you are still climbing uphill, while in fact you have already passed the summit and started descending. **Taylor expansion** is the tool for analyzing how large a step can be before the first-order approximation stops being trustworthy. A first-order expansion looks only at slope; a second-order expansion also considers curvature, meaning how the function bends and in which direction. The trust-region idea behind PPO and TRPO comes from the concern that when parameter updates are too large, first-order approximations are no longer reliable. Taylor expansion helps us understand this mathematically.

$$
f(x+h)\approx f(x)+f'(x)h.
$$

Consider a numerical example. Let

$$
f(x)=x^2,\qquad x=3,\qquad h=0.1.
$$

The true value is

$$
f(3.1)=9.61.
$$

The first-order approximation is

$$
f(3)+f'(3)h=9+6\times0.1=9.6.
$$

It is already close, with an error of $0.01$. A second-order Taylor expansion adds a curvature correction term:

$$
f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2.
$$

For $f(x)=x^2$, $f''(x)=2$, so

$$
9+6\times0.1+\frac{1}{2}\times2\times0.1^2=9.61.
$$

In the multivariable case, the $f''$ in the second-order term becomes the Hessian matrix $H$, which records how the function curves in each direction:

$$
f(\theta+\Delta\theta)
\approx
f(\theta)
\nabla f(\theta)^\top\Delta\theta
\frac{1}{2}\Delta\theta^\top H\Delta\theta.
$$

The trust-region idea behind PPO and TRPO is exactly the concern that when parameter updates are too large, first-order approximations are no longer reliable and the second-order curvature term begins to matter. If we still take large steps based only on first-order information, we may damage the policy.

For the PPO probability ratio

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)},
$$

expand around $\theta_{old}$:

$$
r_t(\theta)
\approx
1
+\nabla_\theta r_t^\top(\theta-\theta_{old})
+\frac{1}{2}(\theta-\theta_{old})^\top
\nabla_\theta^2 r_t
(\theta-\theta_{old}).
$$

The three terms mean:

| Term              | Meaning                                               |
| ----------------- | ----------------------------------------------------- |
| $1$               | When the new and old policies match, the ratio is $1$ |
| First-order term  | Linear change caused by a small update                |
| Second-order term | Extra change from curvature after the step grows      |

Although PPO clipping does not explicitly compute the Hessian, it indirectly avoids the risk of uncontrolled higher-order terms by restricting the range of $r_t(\theta)$.

---

## Group Normalization in GRPO

We have discussed policy gradients, PPO clipping, and Taylor expansion. All of these methods need an advantage estimate $\hat{A}_t$. Traditional methods such as PPO use a trained Critic network to estimate the advantage, but the Critic itself also has to be trained, which adds engineering complexity. **The core idea of GRPO is to avoid the Critic and instead construct advantages through relative comparison among answers in the same group.** Imagine a teacher grading an open-ended problem: put four students' scores together, give positive signals to answers above the group average, and negative signals to answers below the group average. There is no need for an additional "standard-score judge". Concretely, suppose the same prompt samples four answers with rewards

$$
r=[2,4,6,8].
$$

The mean is

$$
\mu=\frac{2+4+6+8}{4}=5.
$$

The standard deviation is

$$
\sigma=
\sqrt{
\frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4}
}
=\sqrt{5}.
$$

The standardized advantage of the fourth answer is

$$
\hat{A}_4=\frac{8-5}{\sqrt{5}}\approx1.34.
$$

The general form is

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma}.
$$

The whole calculation has two steps:

1. Subtract the mean: decide whether this answer is better or worse than the group average.
2. Divide by the standard deviation: put rewards from different questions onto a comparable scale. Some questions naturally produce higher scores and some lower scores; after dividing by the standard deviation, they can be compared across questions.

GRPO can remove the traditional PPO Critic because it constructs the baseline from relative comparison within the group. It does not care about the absolute score of an answer; it cares about where that answer ranks within its group.

---

## Summary

This page introduced three derivation tools:

| Tool                     | Core formula                                     | Role                                                                   |
| ------------------------ | ------------------------------------------------ | ---------------------------------------------------------------------- |
| Log-derivative trick     | $\nabla\pi = \pi\nabla\log\pi$                   | Turns probability gradients into sampleable log form                   |
| Taylor expansion         | $f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2$ | Explains PPO trust regions and clipping through second-order intuition |
| GRPO group normalization | $\hat{A}_i=(r_i-\mu)/\sigma$                     | Replaces the Critic with relative comparison inside a group            |

These three tools correspond to the derivation skeleton of policy gradients, the theoretical basis for limiting update size, and an alternative to using a Critic. The next page organizes them into a complete formula reference.

> **Next**: [E.3.5 Complete Optimization Formulas](./calculus-advanced-formulas), a formula reference for PG, DQN, GAE, PPO, and GRPO.
