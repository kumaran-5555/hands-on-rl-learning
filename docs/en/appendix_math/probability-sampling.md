---
title: E.2.3 Monte Carlo, Incremental Averages, and Importance Sampling
---

# E.2.3 Monte Carlo, Incremental Averages, and Importance Sampling

> **Prerequisites**: [E.2.2 Random Variables, Returns, and State Values](./probability-value) -- you should know the definitions of expectation and variance.

---

The previous article defined expectation as "averaging all possible outcomes after weighting them by probability." The problem is that in real environments, we often do not know all probabilities and outcomes, so we cannot compute the expectation directly. What should we do? The most intuitive method is to try many times and take the average.

## Monte Carlo Estimation

Suppose we sample 5 trajectories from the same starting state and obtain returns:

$$
8, \quad 10, \quad 7, \quad 9, \quad 6.
$$

The sample average is:

$$
\hat{v}(s)=\frac{8+10+7+9+6}{5}=8.
$$

If we keep sampling, the average will get closer and closer to the true expectation. This is the Monte Carlo method: use a sample average to approximate the true expectation.

In code-level notation, Monte Carlo value estimation looks like this:

$$
V(s) \leftarrow \frac{G_1+G_2+\cdots+G_N}{N}.
$$

It does not require knowing the environment's transition probabilities. It only requires the ability to sample trajectories.

---

## Incremental Averages

Monte Carlo is intuitive, but it has a practical engineering problem: if we store all historical returns and recompute the average every time, memory grows and computation gets slower. Can we keep only the "current average" and update it once when a new sample arrives, without looking back through old data? **Incremental averaging** solves this problem. It turns Monte Carlo estimation into an online algorithm: every new sample immediately updates the estimate, and no history needs to be stored.

Suppose we have already observed 4 returns, and the current average return is $7$. Now the fifth return is $11$. The new average is:

$$
\frac{4\times7+11}{5}=7.8.
$$

It can also be written as:

$$
7 + \frac{1}{5}(11-7)=7.8.
$$

The general form is:

$$
\mu_n = \mu_{n-1}+\frac{1}{n}(x_n-\mu_{n-1}).
$$

This form looks simple, but it is the shared skeleton of temporal-difference learning, stochastic approximation, and many RL update rules:

$$
\text{new estimate} = \text{old estimate} + \text{step size} \times (\text{target} - \text{old estimate}).
$$

### Correspondence With RL Updates

This "skeleton" appears throughout RL. Compare a few concrete examples:

| Algorithm   | Update formula                                        | Skeleton correspondence                                |
| ----------- | ----------------------------------------------------- | ------------------------------------------------------ |
| Monte Carlo | $V(s)\leftarrow V(s)+\frac{1}{N}[G-V(s)]$             | Step size $1/N$, target $G$                            |
| TD(0)       | $V(s)\leftarrow V(s)+\alpha[r+\gamma V(s')-V(s)]$     | Step size $\alpha$, target $r+\gamma V(s')$            |
| Q-learning  | $Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max Q-V(s)]$ | Step size $\alpha$, target $r+\gamma\max_{a'}Q(s',a')$ |

The shared idea is: **move the old estimate a small step toward the target**. The difference is how the target is built. Monte Carlo uses the full return, TD uses one-step bootstrapping, and Q-learning uses the maximum action value. Later, GAE will interpolate between these kinds of targets.

---

## The Intuition Behind Importance Sampling

Monte Carlo and incremental averaging both assume that data is sampled directly from the policy being evaluated. In other words, to evaluate policy A, we must use policy A to collect data. But RL often faces a dilemma: running policy A to collect new data is expensive, while an old policy B has already collected many trajectories. Can we reuse the old data to evaluate policy A? The problem is that the new and old policies choose actions with different probabilities. The old policy may prefer left while the new policy prefers right, so directly averaging old data would be biased. **Importance sampling** is introduced to solve this off-policy problem: evaluating a new policy using data from an old policy. Its core idea is to correct each sample's weight by a probability ratio.

How do we correct it? Use the probability ratio. For example:

- The behavior policy chooses action right with probability $0.5$.
- The target policy chooses action right with probability $0.8$.

If a sampled trajectory actually chose right, then this sample is more common under the target policy than under the behavior policy, so it should receive a larger weight. The correction weight is:

$$
\rho=\frac{0.8}{0.5}=1.6.
$$

If this sample's return is $10$, its weighted contribution is $1.6 \times 10 = 16$. This is importance sampling: **correct the bias caused by which policy generated the sample using a probability ratio**.

---

## Summary

This article introduced three practical methods for computing expectations:

| Method                 | Core idea                                                      | RL role                                                    |
| ---------------------- | -------------------------------------------------------------- | ---------------------------------------------------------- |
| Monte Carlo estimation | Approximate expectation with a sample average                  | Estimate values when transition probabilities are unknown  |
| Incremental average    | Update once per sample without storing history                 | Shared skeleton of TD updates and stochastic approximation |
| Importance sampling    | Correct the bias from old-policy data using probability ratios | Foundation for off-policy learning and PPO                 |

Monte Carlo solves the problem "we cannot directly compute the expectation without transition probabilities": sample repeatedly and average. Incremental averaging solves Monte Carlo's memory problem: keep only the current estimate and update it online. Importance sampling solves the problem "the data was collected by an old policy": weight samples by probability ratios to correct the bias. Together, these three methods form the basis for almost all value estimation and policy update algorithms in RL.

> **Next**: [E.2.4 Trajectory Probability, Baselines, and GAE](./probability-trajectory-td) -- from single-step sampling to full trajectories, and from expectations to variance control.
