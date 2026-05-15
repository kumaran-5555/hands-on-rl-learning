---
title: E.4.5 Complete Formulas
---

# E.4.5 Complete Formulas for KL, RLHF, DPO, and Mutual Information

> **Prerequisites**: This page summarizes all formulas in module E.4. It is best reviewed after reading [E.4.1](./information-basics) through [E.4.4](./information-mutual-info).

---

This page collects the complete formulas from module E.4 for review. It is recommended that you read the previous articles first and then use this page as a reference table.

## The Relationship Between KL, Cross-Entropy, and Entropy

We have looked at entropy, cross-entropy, and KL divergence separately. They are actually tied together by one equation, and this equation is the foundation for understanding all later formulas.

$$
D_{KL}(P\|Q)=H(P,Q)-H(P).
$$

Expanding the definitions:

$$
H(P,Q)-H(P)
= -\sum_x P(x)\log Q(x) + \sum_x P(x)\log P(x).
$$

Combining the sums:

$$
=\sum_x P(x)\log\frac{P(x)}{Q(x)}
=D_{KL}(P\|Q).
$$

This shows that KL divergence can be understood as the extra information cost paid when the true distribution is $P$ but you encode it with $Q$.

In machine learning:

- Minimize cross-entropy $H(P,Q)$.
- When $P$ is fixed, this is equivalent to minimizing $D_{KL}(P\|Q)$.

This is the mathematical basis of cross-entropy loss in classification models, reward models, and language model training.

---

## Advanced Formula: KL-Regularized Objective in RLHF

This section places KL divergence inside the full RLHF optimization objective. The reward term pushes the model toward high-scoring answers, while the KL term acts like a safety rope that pulls the model back near the reference model.

RLHF policy optimization is often written as:

$$
\max_\pi \; \mathbb{E}_{x,y\sim\pi}[r(x,y)]
-\beta D_{KL}(\pi(\cdot\mid x)\|\pi_{ref}(\cdot\mid x)).
$$

where:

- $r(x,y)$ is the score assigned by the reward model to answer $y$.
- $\pi$ is the current policy model being optimized.
- $\pi_{ref}$ is the reference model, usually the SFT model.
- $\beta$ controls the tradeoff between "pursue reward" and "do not drift away from the reference model."

If $\beta$ is too small, the model can drift too far in pursuit of reward and produce reward hacking. If $\beta$ is too large, the model barely dares to change and learning becomes weak.

---

## Advanced Formula: DPO's Log Probability Ratio

DPO does not explicitly train a reward model and then run PPO. Instead, it directly optimizes the policy using preference data. Its core tool is the log probability ratio, which compares how much the current model prefers a certain answer relative to the reference model.

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

For a preference sample $(x,y_w,y_l)$, where $y_w$ is the better answer and $y_l$ is the worse answer, the DPO loss is often written as:

$$
\mathcal{L}_{DPO}(\theta)
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\right).
$$

This expression can be understood from a simple example:

- If the model raises the winner's probability more than the reference model does, the first term becomes larger.
- If the model raises the loser's probability more than the reference model does, the second term becomes larger and offsets the advantage.
- The larger the difference, the more the model agrees with the preference data.

The core of DPO is not to make the winner's probability infinitely large. It is that, relative to the reference model, the winner should be preferred over the loser. This is the implicit form of KL regularization in preference learning.

---

## Advanced Formula: Mutual Information and Representation Learning

Mutual information combines entropy and KL divergence to answer how much information two random variables share. In representation learning, it is used to evaluate whether a state representation keeps information related to task return.

$$
I(X;Y)=D_{KL}(P_{XY}\|P_XP_Y)=H(X)-H(X\mid Y).
$$

In reinforcement learning representation learning, we may want the state representation $\phi(s)$ and future return $G_t$ to have high mutual information:

$$
I(\phi(s);G_t) \text{ is large}.
$$

This means the representation preserves information related to task return. At the same time, we may want the representation to have low mutual information with irrelevant noise, improving generalization.

Formulas like this do not necessarily appear directly in basic algorithms, but they are common in exploration, representation learning, world models, and unsupervised RL.

---

## Summary

This page summarized the core formulas from module E.4:

| Formula category         | Core equation/expression                                                                                       | Intuition                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| KL-cross-entropy-entropy | $D_{KL}(P\|Q)=H(P,Q)-H(P)$                                                                                     | The extra encoding cost is the distribution gap                |
| RLHF objective           | $\max_\pi \mathbb{E}[r]-\beta D_{KL}(\pi\|\pi_{ref})$                                                          | Pursue reward but do not move too far from the reference model |
| DPO loss                 | $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})$ | A larger relative probability gap is better                    |
| Mutual information       | $I(X;Y)=H(X)-H(X\mid Y)=D_{KL}(P_{XY}\|P_XP_Y)$                                                                | How much uncertainty in $X$ is reduced after knowing $Y$       |

> **Next**: [E.4.6 Formula Reference and Exercises](./information-formulas-exercises) -- review all formulas in this module and check your understanding with exercises.
