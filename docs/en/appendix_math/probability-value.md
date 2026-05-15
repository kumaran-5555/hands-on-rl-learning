---
title: E.2.2 Random Variables, Returns, and State Values
---

# E.2.2 From Random Trajectories to State Values

> **Prerequisites**: [E.2.1 Probability, Conditional Probability, and Expectation](./probability-basics) -- you should know the definition of expectation.

---

The previous article defined expectation as "averaging by weighting outcomes with their probabilities." This article applies expectation to reinforcement learning. Starting from a state, the trajectory is random and the return is random. The value of that state is the expectation over all possible returns.

## From Random Trajectories to State Values

Starting from state $s$, suppose there are three possible trajectories:

| Trajectory | Probability | Discounted return |
| ---------- | ----------- | ----------------- |
| A          | $0.5$       | $8$               |
| B          | $0.3$       | $4$               |
| C          | $0.2$       | $-2$              |

What is the value of state $s$ under policy $\pi$? By the definition of expectation, multiply each outcome by its probability and add:

$$
v_\pi(s)=0.5\times8+0.3\times4+0.2\times(-2)=4.8.
$$

This number is the concrete expansion of

$$
v_\pi(s)=\mathbb{E}_\pi[G_t \mid S_t=s].
$$

The subscript $\pi$ in $\mathbb{E}_\pi$ means "weighted by probabilities under policy $\pi$." The vertical bar $\mid S_t=s$ means "under the condition that the starting state is $s$." Read the whole line as: **starting from state $s$, following policy $\pi$, take the weighted average of all possible returns**.

Notice that $v_\pi(s)=4.8$ is not the return of one trajectory. Any actual sampled trajectory may return $8$, $4$, or $-2$. The number $4.8$ is the average over infinitely many trajectories. This is the difference between "return" and "value": a return is single-run, while a value is an expectation.

### How the Discount Factor Affects Value

The example above did not explicitly consider discounting. Each trajectory directly supplied a return. In real RL, the return is discounted:

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots.
$$

The closer $\gamma$ is to $1$, the more the policy "cares about the future." The smaller $\gamma$ is, the more the policy "focuses on the immediate present." Consider a numerical example.

Suppose one trajectory has the immediate reward sequence $2, 1, 3, 0, 1$.

| $\gamma$ | Discounted return $G$                                      | Meaning                                |
| -------- | ---------------------------------------------------------- | -------------------------------------- |
| $0.5$    | $2+0.5\times1+0.25\times3+0.125\times0+0.0625\times1=3.19$ | Almost only the first two steps matter |
| $0.9$    | $2+0.9+0.81\times3+0.729\times0+0.6561\times1=5.97$        | Looks fairly far ahead                 |
| $0.99$   | $2+0.99+0.9801\times3+\cdots\approx6.92$                   | Almost all rewards are counted         |

When $\gamma=0.5$, the fifth reward contributes only $0.0625$ to the total return, so it is almost negligible. When $\gamma=0.99$, the fifth reward still contributes about $0.96$, nearly as important as an immediate reward.

This directly changes the numerical value of the value function. With a larger $\gamma$, state values are often higher because they include more future rewards, but they are also harder to compute and estimate because the calculation must account for a longer and more uncertain future.

---

## Variance: Measuring Instability

Two policies can have the same average return while feeling completely different during training.

Policy A has three returns:

$$
4, \quad 5, \quad 6.
$$

Policy B has three returns:

$$
0, \quad 5, \quad 10.
$$

Both averages are $5$. But policy B fluctuates much more: sometimes it receives $0$, sometimes $10$. If these returns are used to update policy parameters, the gradient direction for policy B swings sharply from update to update, making training unstable. Variance measures this kind of fluctuation.

For policy B, the deviations from the average are $-5, 0, 5$. Squaring them and averaging gives:

$$
\mathrm{Var}(G)=\frac{(-5)^2+0^2+5^2}{3}=\frac{50}{3}=16.67.
$$

$\mathrm{Var}$ is short for variance. The formula is $\mathrm{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2]$: take each value's deviation from the mean, square it, and average.

In reinforcement learning, high variance means the learning signal is unstable. Policy gradient methods often need baselines, advantages, GAE, and related techniques to reduce variance. The mathematical basis for these techniques appears in the following articles.

### What Variance Means During Training

Consider policy gradients. Suppose the gradient update signal for "choose right" in a state is $\hat{A}\cdot\nabla\log\pi$, where $\hat{A}$ is an advantage estimate.

**Low-variance policy A**: the advantage estimate stays near $2$. Each update points in roughly the same direction, so the parameters move steadily.

**High-variance policy B**: the advantage estimate jumps between $-8$ and $+12$. One update says "right is bad, reduce its probability." The next says "right is good, increase its probability." The parameters oscillate, and training becomes inefficient.

This is why reducing variance is almost always one of the central issues in RL training. Baselines, GAE, and PPO clipping are all answering the same question in different forms: **how can we make the gradient update signal more stable?**

---

## Summary

This article applied the basic tools of probability theory to core RL concepts:

| Concept         | Definition                                                                  | RL role                                                                              |
| --------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| State value     | Conditional expectation of return, $v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$ | Measures the average return from a state                                             |
| Discount factor | $G_t=\sum_{k=0}\gamma^k R_{t+k+1}$                                          | Controls how far the policy looks ahead; larger $\gamma$ values emphasize the future |
| Variance        | $\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]$                           | Measures return fluctuation and affects training stability                           |

State value is the conditional expectation of return: it compresses the returns of many random trajectories into one representative number. The discount factor controls how far this compression looks: a small $\gamma$ focuses on the present, while a large $\gamma$ values the future but is harder to estimate. Variance tells us how stable the gradient signal is. Two policies may have the same average return, but the one with larger variance is harder to train. Later methods such as Monte Carlo estimation, baselines, and GAE all try to reduce variance while keeping the expectation as unchanged as possible.

> **Next**: [E.2.3 Monte Carlo, Incremental Averages, and Importance Sampling](./probability-sampling) -- approximating expectations with sample averages when transition probabilities are unknown.
