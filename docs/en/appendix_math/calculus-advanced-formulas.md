---
title: E.3.5 Complete Formulas
---

# E.3.5 Complete Formulas for PG, DQN, GAE, PPO, and GRPO

> **Prerequisite**: This page summarizes all formulas in module E.3. It is best reviewed after reading [E.3.1](./calculus-basics) through [E.3.4](./calculus-derivations).

On the previous pages, we derived the single-sample form of the policy gradient, the log-derivative trick, PPO clipping, and GRPO group normalization. This page organizes those results into complete formulas and adds the DQN loss function and the derivation of GAE. You can treat this page as a quick reference and return to it whenever an unfamiliar symbol appears.

---

## Policy Gradient Theorem

We have already seen the single-sample form:

$$
\nabla_\theta J(\theta) \approx G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

The complete policy gradient theorem is written as

$$
\nabla_\theta J(\theta)
=\sum_s d^\pi(s)\sum_a q_\pi(s,a)\nabla_\theta\pi_\theta(a\mid s).
$$

Each symbol means:

- $d^\pi(s)$ is the frequency with which state $s$ is visited under policy $\pi$. You can read it as "how much time the policy spends in state $s$ when it runs".
- $q_\pi(s,a)$ is the action-value function. It represents the average future return after taking action $a$ in state $s$.
- $\nabla_\theta\pi_\theta(a\mid s)$ describes how the probability of choosing action $a$ changes when the parameter $\theta$ changes.

Using the log-derivative trick derived in the previous section, this formula can be rewritten in the more common log form. Because

$$
\nabla_\theta\pi_\theta(a\mid s)
=\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s),
$$

we have

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^\pi,a\sim\pi}
\left[q_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

If the sampled return $G_t$ is used to estimate $q_\pi(s_t,a_t)$, we get REINFORCE:

$$
\nabla_\theta J(\theta)
\approx
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

If the action value is replaced by an advantage function, we get a more stable form:

$$
\nabla_\theta J(\theta)
=\mathbb{E}
\left[A_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

Seen as a whole, the complicated formula does not appear out of nowhere. It is simply the intuition "increase the probability of good actions and decrease the probability of bad actions" written as a weighted average over all states and actions.

---

## Loss for Value Function Approximation

Policy gradients handle "how to update the policy", but training also needs a module that estimates "how many points a state or action is worth". This is the job of the Critic or DQN. **Why do we need this module?** Because the advantage estimate $\hat{A}_t$ in policy gradients depends on an accurate estimate of the value $V(s)$. If the value estimate is inaccurate, the policy update direction can become biased. The learning objective is direct: make the predicted value as close as possible to the target value.

Given a sample $(s_t,a_t,r_{t+1},s_{t+1})$, the TD target in DQN is

$$
y_t=r_{t+1}+\gamma\max_{a'}Q_{\theta^-}(s_{t+1},a').
$$

Here $\theta^-$ denotes the target-network parameters. The loss function is

$$
L(\theta)=\frac{1}{2}\left(Q_\theta(s_t,a_t)-y_t\right)^2.
$$

Taking the gradient gives

$$
\nabla_\theta L(\theta)
=\left(Q_\theta(s_t,a_t)-y_t\right)
\nabla_\theta Q_\theta(s_t,a_t).
$$

The first term is the TD error:

$$
\delta_t=y_t-Q_\theta(s_t,a_t).
$$

The second term, $\nabla_\theta Q_\theta(s_t,a_t)$, tells us how the parameters change the predicted value. DQN training repeatedly reduces this prediction error.

---

## GAE: Estimating Advantages by Accumulating TD Errors

Policy gradients need the advantage function $\hat{A}_t$ to measure how much better an action is than the average level, but this quantity cannot be observed directly. There are two extreme estimation methods. Monte Carlo methods use the return of the whole trajectory: low bias but high variance, because randomness accumulates over many steps. Temporal difference methods use only one step, "reward plus next-state estimate": low variance but high bias, because only one step of information is used. **GAE (Generalized Advantage Estimation) is introduced to adjust flexibly between these two extremes**. It accumulates future multi-step TD errors with decreasing weights, using the parameter $\lambda$ to control whether the estimate leans more toward MC or TD. Start with the one-step TD error:

$$
\delta_t=R_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

If $\delta_t>0$, the actual outcome is better than the Critic expected. If $\delta_t<0$, the actual outcome is worse than expected. TD error only looks one step ahead. GAE accumulates future TD errors with decreasing weights:

$$
\hat{A}^{GAE}_t
=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

Here $\lambda\in[0,1]$ controls the bias-variance tradeoff:

- Small $\lambda$: relies more on short-term TD errors, with lower variance but potentially higher bias.
- Large $\lambda$: closer to the full return, with lower bias but potentially higher variance.

GAE is common in PPO because it provides a convenient knob for balancing stability and accuracy.

---

## PPO Clipped Objective

We have already seen the intuition behind the probability ratio and clipping:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

The PPO clipped objective is

$$
L^{CLIP}(\theta)=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$

The formula looks complex, but it is not difficult if we unpack it part by part.

If $\hat{A}_t>0$, the action is better than average. We want to increase its probability, but only up to $1+\epsilon$ times the old probability.

If $\hat{A}_t<0$, the action is worse than average. We want to decrease its probability, but only down to $1-\epsilon$ times the old probability.

Therefore, the combination of `min` and `clip` implements a simple and effective principle: **update the policy in the correct direction, but do not let it move too far in one step**.

---

## GRPO Group-Normalized Advantage

GRPO uses relative comparison within a group of answers. Suppose the same question generates $n$ answers, with rewards

$$
r_1,r_2,\ldots,r_n.
$$

First compute the mean:

$$
\mu=\frac{1}{n}\sum_{i=1}^n r_i.
$$

Then compute the standard deviation:

$$
\sigma=\sqrt{\frac{1}{n}\sum_{i=1}^n(r_i-\mu)^2}.
$$

The standardized advantage of each answer is

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma+\epsilon}.
$$

For example, if the rewards are $[2,4,10]$, the mean is $5.33$. The third answer is clearly above average, so its advantage is positive. The first answer is below average, so its advantage is negative. The benefit of this within-group relative comparison is that the model does not need an additional Critic network. It can update the policy using only comparisons among answers in the same group.

---

## Summary

This page summarized all core formulas in module E.3:

| Formula                 | Core expression                                                        | Use                                               |
| ----------------------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| Policy gradient theorem | $\nabla_\theta J=\mathbb{E}[\nabla\log\pi\cdot Q^\pi]$                 | Theoretical foundation of policy optimization     |
| DQN loss                | $L=\frac{1}{2}(Q_\theta-y_t)^2$                                        | Training objective for value functions            |
| GAE                     | $\hat{A}^{GAE}_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$                  | Advantage estimate with bias-variance tradeoff    |
| PPO clipping            | $\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)$ | Limits policy update size                         |
| GRPO group advantage    | $\hat{A}_i=(r_i-\mu)/(\sigma+\epsilon)$                                | Within-group relative comparison without a Critic |

When you encounter an unfamiliar symbol, return to this page for reference. The next page uses exercises to check your understanding of these formulas.

> **Next**: [E.3.6 Formula Reference and Exercises](./calculus-formulas-exercises), which summarizes all formulas in this module and checks understanding through exercises.
