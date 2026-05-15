---
title: E.2.5 Bellman Expectation Equation and Action Values
---

# E.2.5 Bellman Expectation Equation and Action Values

> **Prerequisites**: [E.2.1 Probability Basics](./probability-basics) and [E.2.2 State Values](./probability-value) -- you should know conditional probability and the linearity of expectation.

---

Earlier we interpreted the value function as "the average return from a state":

$$
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
$$

This definition is correct, but it hides all details inside the expectation symbol $\mathbb{E}_\pi$. If we do not know what is being summed inside the expectation or what weights are being used, we cannot actually compute the value. It is like knowing that "the average grade is 85" without knowing how many courses there are or how each course is weighted. This article expands the expectation step by step and shows which sums live behind the notation. The recursive relationship that appears after this expansion is the **Bellman expectation equation**. Its purpose is to compress "the value of all future steps" into the recursive form "one-step reward plus next-state value," so we can compute values without running a full trajectory.

First split $G_t$ into "one-step reward plus future return":

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

Substitute this into the value function:

$$
v_\pi(s)=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].
$$

Expectation has a basic property: linearity. The expectation of a sum is the sum of expectations. Therefore:

$$
v_\pi(s)=\mathbb{E}_\pi[R_{t+1}\mid S_t=s]+\gamma\mathbb{E}_\pi[G_{t+1}\mid S_t=s].
$$

The left term, $\mathbb{E}_\pi[R_{t+1}\mid S_t=s]$, is the average immediate reward. It expands as:

$$
\mathbb{E}_\pi[R_{t+1}\mid S_t=s]
=\sum_a \pi(a\mid s)\sum_r p(r\mid s,a)r.
$$

The right term, $\mathbb{E}_\pi[G_{t+1}\mid S_t=s]$, is the average next-state value:

$$
\mathbb{E}_\pi[G_{t+1}\mid S_t=s]
=\sum_a \pi(a\mid s)\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

Combining the two terms gives the Bellman expectation equation:

$$
v_\pi(s)=\sum_a\pi(a\mid s)
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
\right].
$$

The formula looks complex, but each layer is only a probability-weighted average:

- $\sum_a\pi(a\mid s)$: average over all possible actions.
- $\sum_r p(r\mid s,a)r$: average over all possible rewards.
- $\sum_{s'}p(s'\mid s,a)v_\pi(s')$: average over the values of all possible next states.

### Checking With the Two-State Example

Return to the two-state example from the introduction. Suppose the policy is deterministic. In $s_1$, it chooses one action, which always moves to $s_2$ and gives reward $2$. In $s_2$, it chooses one action, which always moves to $s_1$ and gives reward $1$. Let $\gamma=0.5$.

Because the policy is deterministic, $\pi(a\mid s)$ is $1$ for the chosen action and $0$ for all others, so the "sum over actions" layer disappears. Expanding $s_1$:

$$
v(s_1)=\underbrace{2}_{\text{immediate reward}}+\gamma\underbrace{(1\times v(s_2))}_{\text{next state }s_2\text{ with probability }1}=2+0.5v(s_2).
$$

Expanding $s_2$:

$$
v(s_2)=\underbrace{1}_{\text{immediate reward}}+\gamma\underbrace{(1\times v(s_1))}_{\text{next state }s_1\text{ with probability }1}=1+0.5v(s_1).
$$

These are the two equations from the introduction. This verifies that the Bellman expectation equation is doing exactly "one-step reward plus probability-weighted future value."

### What If the Policy Is Stochastic?

If the policy has two possible actions in $s_1$, with probabilities $\pi(\text{left}\mid s_1)=0.3$ and $\pi(\text{right}\mid s_1)=0.7$:

- Choosing left gives reward $1$ and moves next to $s_1$.
- Choosing right gives reward $3$ and moves next to $s_2$.

Then the Bellman expectation equation for $s_1$ is:

$$
v(s_1)=0.3\times[1+0.5v(s_1)]+0.7\times[3+0.5v(s_2)].
$$

Each bracket contains "the immediate reward plus future value after choosing one action," and the coefficient in front is the probability of choosing that action. **Expanding layer by layer means taking probability-weighted averages layer by layer**.

---

## Action Values and Advantage Functions

The state value function $v_\pi(s)$ answers: "starting from state $s$, what average return can we get?" But when training a policy, we often face a more specific question: in state $s$, is action left better or action right better? Knowing only the average return cannot answer this, because the average may mix one good action and one bad action. We need to evaluate each action separately. That is why we need the **action value function**.

$$
q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

The action value function $q_\pi(s,a)$ is defined as: after choosing action $a$ in state $s$, and then continuing under policy $\pi$, what average return do we get?

Its Bellman form is:

$$
q_\pi(s,a)=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

Notice the relationship between $q$ and $v$: the state value is the weighted average of action values:

$$
v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a).
$$

Intuitively, "the average return from $s$" equals "the sum of each action's return multiplied by the probability of choosing that action."

The **advantage function** is the difference between action value and state value:

$$
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s).
$$

It measures how much better action $a$ is than the average action level in state $s$. The earlier example "return 10, average 8, so the advantage is 2" is a numerical version of this formula.

### The Relationship Between q, v, and A in Numbers

Suppose that in a state $s$, the policy chooses left with probability $0.4$ and right with probability $0.6$. We know:

| Action | Action value $q(s,a)$ |
| ------ | --------------------- |
| left   | $5$                   |
| right  | $8$                   |

The state value is the weighted average of action values:

$$
v(s)=0.4\times5+0.6\times8=2+4.8=6.8.
$$

The advantage values are:

$$
A(s,\text{left})=5-6.8=-1.8, \qquad A(s,\text{right})=8-6.8=1.2.
$$

Left is $1.8$ below average, so it has negative advantage. Right is $1.2$ above average, so it has positive advantage. Policy gradients use this sign to decide that the probability of right should go up and the probability of left should go down.

---

## Trajectory-Form Importance Sampling

We have already seen the one-step importance weight:

$$
\rho_t=\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}.
$$

If a full trajectory is sampled by a behavior policy $b$, and we want to estimate the return of a target policy $\pi$, then we multiply the probability ratios across all steps:

$$
\rho_{0:T}=\prod_{t=0}^{T}\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}.
$$

Here $\prod$ means "multiply all terms together," just as $\sum$ means "add all terms together." The off-policy Monte Carlo estimate becomes:

$$
\hat{v}_\pi(s)=\frac{1}{N}\sum_{i=1}^N \rho^{(i)}G^{(i)}.
$$

This method works, but it is risky. If many probability ratios are multiplied, the weight can become extremely large and cause variance to explode. Practical algorithms therefore often use truncation, weighted importance sampling, or other more stable methods.

### Numerical Example: Trajectory Importance Weights

Suppose a two-step trajectory has the following target-policy and behavior-policy probabilities at each step:

| Step  | $\pi(a_t\mid s_t)$ | $b(a_t\mid s_t)$ | One-step weight |
| ----- | ------------------ | ---------------- | --------------- |
| $t=0$ | $0.6$              | $0.3$            | $0.6/0.3=2$     |
| $t=1$ | $0.8$              | $0.4$            | $0.8/0.4=2$     |

The trajectory importance weight is the product of the two steps:

$$
\rho_{0:2}=2\times2=4.
$$

If this trajectory has return $G=5$, its weighted contribution is $4\times5=20$. This may still look acceptable. But if every step has weight $3$, then after $10$ steps the weight becomes $3^{10}=59049$. This is what "variance explosion" means. PPO clipping and normalization are, in essence, methods for controlling such extreme weights.

---

## Covariance and Policy Gradient Variance

The tendency of two random variables to change together is measured by **covariance**:

$$
\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])].
$$

The correlation coefficient further normalizes covariance to $[-1,1]$:

$$
\rho_{X,Y}=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

In reinforcement learning, gradient estimates are often random variables. For example, a policy gradient sample is:

$$
g_t=\hat{A}_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

If $\hat{A}_t$ fluctuates heavily, the variance of $g_t$ also grows. Baselines, advantage normalization, and GAE are all ways of controlling the variance of this stochastic gradient.

### Numerical Example: Covariance

Suppose two policy gradient samples have advantages $\hat{A}=[2, -1]$, and the corresponding gradient norms are $[4, 1]$.

Means: $\bar{A}=0.5$, $\bar{g}=2.5$.

Covariance:

$$
\mathrm{Cov}=\frac{(2-0.5)(4-2.5)+(-1-0.5)(1-2.5)}{2}=\frac{1.5\times1.5+(-1.5)\times(-1.5)}{2}=\frac{4.5}{2}=2.25.
$$

The covariance is positive, which means the gradient tends to be larger when the advantage is larger. The two signals are positively correlated. If the covariance is close to $0$, the advantage size and gradient size have no stable relationship, so the gradient estimate becomes noisier.

---

## Summary

This article expanded the value function from a "black-box expectation" into the computable Bellman expectation equation, then introduced action values and advantage functions:

| Concept                        | Formula                                                                                   | Role                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Bellman expectation equation   | $v_\pi(s)=\sum_a\pi(a\mid s)[\sum_r p(r\mid s,a)r+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')]$ | Compresses "all future steps" into "one step plus recursion"                                |
| Action value                   | $q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$                                          | Evaluates the average return after choosing a specific action                               |
| Advantage function             | $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$                                                          | Measures how much better this action is than the average                                    |
| Trajectory importance sampling | $\rho_{0:T}=\prod_t\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}$                               | Uses old-policy data to evaluate a new policy; watch for variance explosion                 |
| Covariance                     | $\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$                        | Measures joint fluctuation of two random variables and helps reason about gradient variance |

The core idea of the Bellman expectation equation is recursion: we do not need to run a full trajectory if we know "one-step reward plus next-state value." Action values let us compare actions, and the advantage function turns that comparison into a number. Trajectory importance sampling lets us reuse old data, but it carries the risk of variance explosion.

> **Next**: [E.2.6 Probability and Statistics Formula Reference and Exercises](./probability-formulas-exercises) -- a formula summary for this module, with exercises.
