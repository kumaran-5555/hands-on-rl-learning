---
title: E.4.2 Cross-Entropy and KL
---

# E.4.2 Cross-Entropy and KL Divergence

> **Prerequisites**: [E.4.1 Self-Information, Entropy, and Exploration](./information-basics). You need to know the definition of entropy.

---

In the previous section, we used entropy to measure the randomness of a single policy. But in training, the more common situation is comparing two distributions: the model's predicted distribution vs. the true label distribution, or a new policy vs. an old policy. This requires two new tools: cross-entropy and KL divergence.

## Cross-Entropy: The Cost of Walking with the Wrong Map

Cross-entropy measures the cost of making predictions with the wrong distribution. Classification models and reward models use it as a training loss. When the true distribution is $P$ and you predict with $Q$, cross-entropy tells you how large the cost is.

Imagine trying to find a road while holding an inaccurate map. If the map is close to the real roads, you only take a small detour. If the map is badly distorted, you may get completely lost. Cross-entropy measures this "cost of walking with the wrong map." Mathematically, it is defined as:

$$
H(P,Q)=-\sum_x P(x)\log Q(x).
$$

It looks similar to entropy. The difference is that the logarithm contains $Q$ rather than $P$. In other words, you are encoding with the "wrong" distribution.

Consider a classification example. Suppose the correct answer is the first class:

$$
P=[1,0].
$$

The model predicts:

$$
Q=[0.8,0.2].
$$

Because $P=[1,0]$, the cross-entropy keeps only the term for the correct class:

$$
H(P,Q)=-\log 0.8.
$$

If the model predicts the correct class with more confidence, for example $Q=[0.95,0.05]$, the loss becomes $-\log 0.95$, which is smaller than $-\log 0.8$. The more accurate the prediction, the lower the cross-entropy. This is why it is widely used as a training loss for classification models. In RLHF, reward models, preference models, and policy models all rely on it.

---

## KL Divergence: The "Surprise" Between Two Distributions

Cross-entropy tells us how much it costs to predict one distribution with another. But if we subtract the entropy of the true distribution itself from that cost, the remaining part purely reflects the difference between the two distributions. This is the problem KL divergence solves. KL divergence measures the difference between two distributions, and PPO and RLHF use it to prevent a policy from changing too much.

Intuitively, KL divergence measures this: if your true belief is distribution $P$, but you must act according to distribution $Q$, how "surprised" would you be? The formula is:

$$
D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}.
$$

$\frac{P(x)}{Q(x)}$ is the ratio of two probabilities. If $P$ and $Q$ agree on some $x$, the ratio is close to 1, $\log 1=0$, and there is no surprise. If they disagree strongly, the ratio moves away from 1 and the KL divergence becomes larger.

Consider a practical RL situation. PPO and RLHF often need to compare old and new policies. Suppose the old policy is:

$$
\pi_{old}=[0.5,0.5].
$$

Two candidate new policies are:

$$
\pi_{new}^{(1)}=[0.6,0.4], \qquad \pi_{new}^{(2)}=[0.9,0.1].
$$

Intuitively, new policy 2 is farther from the old policy. Using KL divergence, with the old policy as $P$ and new policy 1 as $Q$:

$$
D_{KL}(\pi_{old}\|\pi_{new}^{(1)})
=0.5\log\frac{0.5}{0.6}+0.5\log\frac{0.5}{0.4}.
$$

With new policy 2 as $Q$:

$$
D_{KL}(\pi_{old}\|\pi_{new}^{(2)})
=0.5\log\frac{0.5}{0.9}+0.5\log\frac{0.5}{0.1}.
$$

The second value is larger, showing that new policy 2 deviates from the old policy more aggressively.

---

## Why KL Divergence Is Not Symmetric

After understanding the basic use of KL divergence, one common pitfall is that KL divergence does not satisfy commutativity. That is, $D_{KL}(P\|Q)\neq D_{KL}(Q\|P)$. The choice of direction is not arbitrary. PPO and RLHF use different directions, emphasizing different types of error.

$$
D_{KL}(P\|Q)\neq D_{KL}(Q\|P).
$$

To understand the asymmetry, consider:

$$
P=[0.99,0.01], \qquad Q=[0.5,0.5].
$$

Looking at $Q$ from the perspective of $P$, namely $D_{KL}(P\|Q)$: in the real world, the first action almost always happens, but your model $Q$ assigns half of the probability to the second action. This error, being vague when reality is almost certain, receives a large penalty.

Looking at $P$ from the perspective of $Q$, namely $D_{KL}(Q\|P)$: in the real world, both actions are possible, but your model $P$ assigns almost all probability to the first one. This error, being overconfident when reality is ambiguous, has a completely different character.

So when using KL divergence, the direction is not arbitrary. PPO uses $D_{KL}(\pi_{old}\|\pi_{new})$, meaning "from the old policy's perspective, how much has the new policy changed?" RLHF uses $D_{KL}(\pi_\theta\|\pi_{ref})$, meaning "from the current model's perspective, how far is it from the reference model?" Different directions emphasize different biases.

---

## Summary

This article introduced two tools for measuring distance between distributions:

| Concept       | Problem it solves                                       | Core formula                                    | Role in RL                                      |
| ------------- | ------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| Cross-entropy | How costly it is to predict with the wrong distribution | $H(P,Q)=-\sum_x P(x)\log Q(x)$                  | Training loss for classifiers and reward models |
| KL divergence | How far apart two distributions are                     | $D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$ | Constrains policy drift in PPO/RLHF             |
| KL asymmetry  | Different KL directions mean different things           | $D_{KL}(P\|Q)\neq D_{KL}(Q\|P)$                 | PPO and RLHF use different directions           |

The relationship between cross-entropy and KL divergence, $D_{KL}(P\|Q)=H(P,Q)-H(P)$, is the key bridge for understanding PPO, RLHF, and DPO in the next article.

> **Next**: [E.4.3 Information Theory in PPO, RLHF, and DPO](./information-rlhf-dpo) -- applying cross-entropy and KL to alignment training.
