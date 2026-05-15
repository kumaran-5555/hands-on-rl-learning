---
title: E.2.6 Probability and Statistics Formula Reference and Exercises
---

# E.2.6 Probability and Statistics Formula Reference and Exercises

> **Prerequisites**: This page summarizes all formulas from module E.2. It is best reviewed after reading [E.2.1](./probability-basics) through [E.2.5](./probability-bellman-advanced). If this is your first pass, start with the main articles first.

---

This page collects all formulas used in module E.2 for review. It is meant as a reference after you have read the preceding articles.

## Probability Formulas Used in This Book

| Concept                      | Formula                                                                                     | RL meaning                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Policy probability           | $\pi(a\mid s)$                                                                              | Probability of choosing action $a$ in state $s$               |
| State transition probability | $p(s'\mid s,a)$                                                                             | Probability of entering the next state after taking an action |
| Expectation                  | $\mathbb{E}[X]=\sum_x p(x)x$                                                                | Average reward, average return, value function                |
| State value                  | $v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$                                                    | Average discounted return from state $s$                      |
| Variance                     | $\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]$                                           | Size of learning-signal fluctuation                           |
| Monte Carlo estimate         | $\hat{v}(s)=\frac{1}{N}\sum_i G_i$                                                          | Estimate value using sample averages                          |
| Trajectory probability       | $p(\tau\mid\pi)=p(s_0)\prod_t\pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)$                        | Probability that a policy generates a full trajectory         |
| Baseline variance reduction  | $\mathbb{E}[\nabla\log\pi(a\mid s)b(s)]=0$                                                  | Subtracting a baseline does not change the expected gradient  |
| GAE                          | $\hat{A}_t^{GAE}=\sum_k(\gamma\lambda)^k\delta_{t+k}$                                       | Trade off between TD and MC                                   |
| Importance weight            | $\rho=\frac{\pi(a\mid s)}{b(a\mid s)}$                                                      | Off-policy correction                                         |
| PPO clipped objective        | $L^{CLIP}=\mathbb{E}[\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)]$ | Limit extreme changes in importance weights                   |

---

## Summary

At this point, the basic tools of probability theory are in place: probability describes randomness, expectation describes averages, variance describes fluctuation, Monte Carlo approximates expectations through sampling, and importance sampling corrects bias with probability ratios. The structure of this page is: start from probability tables, weighted averages, and sample averages, then extend to the Bellman expectation equation, action values, trajectory importance sampling, and stochastic-gradient variance. When reading more complex formulas later, first ask what the formula is averaging over: actions, rewards, next states, trajectories, or gradient samples.

---

## Common Pitfalls

1. **Treating one return as the value.** A value is an expectation; the return from one trajectory is only one sample.
2. **Looking only at the mean and ignoring variance.** Two policies can have the same average return but different variance, so their training stability can be very different.
3. **Thinking importance sampling reuses data for free.** Probability ratios can correct bias, but they may also greatly increase variance.

---

## Exercises

1. Three trajectories have returns $10,4,-2$ and probabilities $0.2,0.5,0.3$. What is the state value?
2. Sample returns are $2,6,10$. What are the mean and variance?
3. The behavior policy probability is $0.25$, and the target policy probability is $0.75$. What is the one-step importance weight?
