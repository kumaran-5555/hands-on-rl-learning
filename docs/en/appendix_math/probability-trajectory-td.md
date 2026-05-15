---
title: E.2.4 Trajectory Probability, Baselines, and GAE
---

# E.2.4 Trajectory Probability, Baselines, and GAE

> **Prerequisites**: [E.2.1 Probability Basics](./probability-basics) and [E.2.2 State Values](./probability-value) -- you should know conditional probability, expectation, and variance.

---

Earlier we studied policy probabilities $\pi(a\mid s)$ and environment transition probabilities $p(s'\mid s,a)$ separately. These two probabilities are local: one controls how the policy chooses actions, and the other controls how the environment returns the next state. But what actually happens in RL is a full trajectory: starting from a state, the policy chooses an action, the environment returns a new state, and the process repeats. The objective of policy gradients is to optimize the average return over all possible trajectories. If we do not know the probability of a trajectory, we cannot weight trajectories correctly when taking that average. This is why we need the probability of a full trajectory.

A trajectory alternates between states and actions:

$$
\tau=(s_0,a_0,s_1,a_1,s_2).
$$

The probability of this trajectory is the product of three kinds of factors:

1. The probability of the initial state $p(s_0)$.
2. The probability that the policy chooses each action, $\pi(a_t\mid s_t)$.
3. The probability of each environment transition, $p(s_{t+1}\mid s_t,a_t)$.

Therefore:

$$
p(\tau\mid\pi)=p(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t).
$$

Consider a two-step numerical example:

| Term                                               | Value  |
| -------------------------------------------------- | ------ |
| Initial state probability $p(s_0)$                 | $0.6$  |
| First action probability $\pi(a_0\mid s_0)$        | $0.5$  |
| First transition probability $p(s_1\mid s_0,a_0)$  | $0.8$  |
| Second action probability $\pi(a_1\mid s_1)$       | $0.25$ |
| Second transition probability $p(s_2\mid s_1,a_1)$ | $0.4$  |

The probability of this trajectory is:

$$
0.6\times0.5\times0.8\times0.25\times0.4=0.024.
$$

Why is trajectory probability so important? Because the policy gradient objective is "average over all possible trajectories":

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[G(\tau)]=\sum_{\tau}p_\theta(\tau)G(\tau).
$$

Each trajectory return $G(\tau)$ is weighted by the probability of that trajectory under the current policy. When we derive the policy gradient theorem later, this expansion is the starting point.

---

## Why a Baseline Does Not Change the Expectation

Policy gradients use the return $G_t$ to measure whether an action was good. But the absolute return can be very large or very small: one episode may return $100$, another may return $-5$. If these numbers are used directly to update parameters, the gradient estimate can have very high variance and training can fluctuate sharply. A natural idea is not to ask "how many points did this episode get?" but "how much better was this episode than average?" That replaces $G_t$ with $G_t-b(s_t)$. Here $b(s_t)$ is the **baseline**, usually chosen as the average value $V(s_t)$ of the state. Intuitively this is more stable, but one question must be answered: after subtracting something, is the gradient direction still correct? Could we accidentally weaken a good action? The result below proves that **subtracting a baseline does not change the expected gradient; it only changes the variance**.

The key is that the following term is zero:

$$
\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)b(s)\right]=0.
$$

First pull out $b(s)$, which does not depend on the action:

$$
b(s)\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s).
$$

Use the log-derivative trick:

$$
\nabla_\theta\log\pi_\theta(a\mid s)
=\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}.
$$

Substitute it:

$$
b(s)\sum_a\pi_\theta(a\mid s)
\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s).
$$

Move the gradient outside the sum:

$$
b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s).
$$

The action probabilities always sum to $1$:

$$
\sum_a\pi_\theta(a\mid s)=1.
$$

Therefore:

$$
b(s)\nabla_\theta 1=0.
$$

This shows that subtracting a baseline does not change the expected gradient; it only changes the variance. The policy gradient can therefore be written as:

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\pi\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
(G_t-b(s_t))
\right].
$$

When $b(s_t)=V^\pi(s_t)$, the term in parentheses is the advantage estimate:

$$
A^\pi(s_t,a_t)=G_t-V^\pi(s_t).
$$

---

## From TD Error to GAE

Once we have the theoretical guarantee for baselines, the practical question is: how should we compute $G_t$ in $G_t-b(s_t)$? Monte Carlo uses the full trajectory return. It has low bias but high variance, because a trajectory may be long and randomness accumulates over many steps. Temporal-difference learning uses only one step, "reward plus next-state estimate." It has lower variance but higher bias, because it looks only one step ahead. GAE, or Generalized Advantage Estimation, is introduced to find a compromise between these extremes: not relying too heavily on one-step estimates, which can be biased, and not relying too heavily on full trajectories, which can have high variance.

Start with the TD error, the basic building block of GAE. The TD error measures the gap between "what we observed for one step" and "our current estimate":

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

Consider numbers first. Suppose:

$$
r_t=2,\qquad \gamma=0.9,\qquad V(s_{t+1})=5,\qquad V(s_t)=6.
$$

Then:

$$
\delta_t=2+0.9\times5-6=0.5.
$$

This means that the observed "reward plus next-state value" is $0.5$ higher than the current estimate, so the current state value may be underestimated.

A one-step TD error has low variance but can have high bias. A full Monte Carlo return has low bias but high variance. GAE takes a weighted average of a sequence of TD errors:

$$
\hat{A}_t^{GAE(\gamma,\lambda)}
=\sum_{k=0}^{T-t-1}(\gamma\lambda)^k\delta_{t+k}.
$$

If $\lambda=0$, only the first term remains:

$$
\hat{A}_t=\delta_t.
$$

If $\lambda$ is close to $1$, many later TD errors participate, making the result closer to a Monte Carlo advantage. Thus $\lambda$ becomes a bias-variance knob:

| $\lambda$ | Resembles   | Characteristics           |
| --------- | ----------- | ------------------------- |
| $0$       | One-step TD | Low variance, higher bias |
| $1$       | Monte Carlo | Low bias, higher variance |
| $0.95$    | Compromise  | Common in PPO             |

---

## The Probability Idea in PPO's Clipped Objective

So far we know that trajectory probability describes how likely a trajectory is, a baseline reduces variance without changing the expectation, and GAE trades off between TD and MC. These tools all converge in PPO. The core problem PPO solves is: **how can we use data collected by an old policy to update a new policy while ensuring that the new policy does not move too far in one step?** Importance sampling provides a way to reuse old data, but probability ratios can become large and cause variance to explode. GAE provides stable advantage estimates. PPO's clipping mechanism adds one more layer: it directly restricts the probability ratio to a controlled range.

The core PPO objective is:

$$
L^{CLIP}(\theta)
=
\mathbb{E}\left[
\min\left(
r_t(\theta)\hat{A}_t,\,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right],
$$

where:

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}.
$$

This probability ratio is the importance sampling weight introduced earlier. The old policy collected the sample, and now we want to evaluate the new policy, so we use "new policy probability / old policy probability" as the correction.

Consider a numerical example. Under the old policy, an action has probability $0.2$; under the new policy, it becomes $0.3$:

$$
r_t=\frac{0.3}{0.2}=1.5.
$$

If the advantage is $\hat{A}_t=4$, the unclipped term is:

$$
r_t\hat{A}_t=1.5\times4=6.
$$

If $\epsilon=0.2$, the allowed range is $[0.8,1.2]$. After clipping:

$$
\mathrm{clip}(1.5,0.8,1.2)\times4=4.8.
$$

Take the smaller value:

$$
\min(6,4.8)=4.8.
$$

So although PPO's formula looks complicated, it combines three ideas:

1. Use a probability ratio for off-policy correction.
2. Use an advantage function to decide whether an action should be reinforced or weakened.
3. Use clipping to restrict the size of the policy update.

---

## Summary

This article assembled local probability tools into the full probabilistic framework needed for policy gradients and PPO:

| Tool                   | Problem solved                                                       | Mathematical core                                                                |
| ---------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Trajectory probability | What is the probability of a whole trajectory?                       | Product of policy probabilities and environment transition probabilities         |
| Baseline               | Gradient estimates have high variance and make training unstable     | Subtracting a baseline does not change the expectation; it only reduces variance |
| GAE                    | MC has high variance and TD has high bias, so a compromise is needed | Exponentially decayed weighted sum of multi-step TD errors                       |
| PPO clipping           | Large importance-sampling weights make updates unstable              | Clip the probability ratio to a controlled range                                 |

Trajectory probability lets us average over "all possible futures." A baseline addresses the variance problem where two policies can have the same average return but very different training stability. GAE provides a tunable balance between bias and variance. PPO combines importance sampling, advantage estimation, and clipping into a stable objective function. These tools build on one another and eventually form one of the most widely used policy optimization methods in modern RL.

> **Next**: [E.2.5 Bellman Expectation Equation and Action Values](./probability-bellman-advanced) -- expanding value functions into Bellman equations, then introducing action values and advantage functions.
