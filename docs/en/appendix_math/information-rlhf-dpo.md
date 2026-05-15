---
title: E.4.3 RLHF and DPO
---

# E.4.3 Information Theory in PPO, RLHF, and DPO

> **Prerequisites**: [E.4.1 Entropy](./information-basics) and [E.4.2 Cross-Entropy and KL](./information-cross-entropy-kl). You need to know the definitions of entropy, cross-entropy, and KL divergence.

---

The previous two sections built the mathematical foundations of entropy, cross-entropy, and KL divergence. Now we will see how these tools operate in frontier alignment training. We will proceed in the order "why KL constraints are needed," then "how RLHF uses them," and finally "how DPO bypasses them." The three ideas form a clear narrative line.

## KL Constraints in PPO and RLHF

When training a policy, if the new policy changes too much in one step, it can overfit or exploit loopholes in the reward function, a problem often called reward hacking. To limit this drift, PPO and RLHF introduce a KL penalty. KL divergence measures how far the new policy is from the old policy or the reference model, and the optimization objective constrains that distance.

Start with a concrete scenario. Suppose the old policy assigns probability $0.01$ to a certain token, and the new policy suddenly changes it to $0.20$. The probability ratio is:

$$
\frac{0.20}{0.01}=20.
$$

The new policy has amplified its preference for this token by a factor of 20. Even if this token brings high reward on the current sample, such a drastic change is dangerous. It may indicate overfitting, or it may mean the model is exploiting the reward function, namely reward hacking.

To prevent this, PPO and RLHF introduce a KL penalty:

$$
\text{optimization objective} = \text{reward} - \beta D_{KL}(\pi_{new}\|\pi_{old}).
$$

Here $\beta$ controls the strength of the penalty. If the new policy moves too far from the old policy, the KL term grows, cancels out the reward gain, and forces policy updates to be smoother.

In RLHF, the KL constraint also has a special mission: it keeps the aligned model from moving too far away from the original language model. Without this constraint, the model may sacrifice general language ability in order to satisfy preference data, for example becoming a model that only gives pleasing responses and no longer answers normally.

---

## The Relationship Between Cross-Entropy, Entropy, and KL

So far, we have seen entropy, cross-entropy, and KL divergence separately. They look independent, but one equation ties them together. Understanding this equation explains why classification models, reward models, and language models all use cross-entropy loss.

$$
D_{KL}(P\|Q)=H(P,Q)-H(P).
$$

In words: **KL divergence equals the cost of encoding the true distribution $P$ with the wrong distribution $Q$, minus the cost of encoding $P$ with the correct distribution $P$ itself. The extra part is purely waste caused by inaccurate prediction**.

The derivation is direct. First write the definitions:

$$
H(P,Q)=-\sum_x P(x)\log Q(x),
$$

$$
H(P)=-\sum_x P(x)\log P(x),
$$

then subtract:

$$
H(P,Q)-H(P)
=
-\sum_xP(x)\log Q(x)
+\sum_xP(x)\log P(x).
$$

Combining the sums:

$$
\sum_xP(x)\log\frac{P(x)}{Q(x)}
=D_{KL}(P\|Q).
$$

This equality appears constantly in machine learning. During training, the true distribution $P$ is fixed, so minimizing cross-entropy $H(P,Q)$ is exactly equivalent to minimizing $D_{KL}(P\|Q)$. This is why classification models, reward models, and language models all use cross-entropy loss. On the surface they reduce cross-entropy; in substance they shrink the gap between the model distribution and the true distribution.

---

## DPO: Replacing the Reward Model with Log Probability Ratios

The RLHF pipeline first trains a reward model and then optimizes the policy with PPO. But this process is complex: it involves four models, the policy model, reference model, reward model, and critic, while also handling KL constraints and training instability. DPO, Direct Preference Optimization, starts from a question: **can we skip the reward model and PPO, and directly optimize the policy using preference data such as "answer A is better than answer B"?** DPO uses log probability ratios to implement KL regularization implicitly, completing in one step what RLHF needs several steps to do.

Its loss function looks intimidating:

$$
\mathcal{L}_{DPO}
=
-\mathbb{E}\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right)
\right].
$$

Do not rush to read the whole expression. Look first at one core component, the log probability ratio:

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

It compares how much the current model prefers answer $y$ relative to the reference model. Take a numerical example for a certain answer $y$:

| Model                                | Probability |
| ------------------------------------ | ----------- |
| Current model $\pi_\theta(y\mid x)$  | $0.20$      |
| Reference model $\pi_{ref}(y\mid x)$ | $0.05$      |

The probability ratio is $\frac{0.20}{0.05}=4$, and the logarithm gives $\log 4$. This means the current model prefers this answer more than the reference model does.

DPO compares the log probability ratios of the winner ($y_w$) and the loser ($y_l$) at the same time:

$$
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}.
$$

If the relative probability of the winner increases and the relative probability of the loser decreases, this difference becomes larger and the loss becomes smaller. The model is learning to prefer better answers more and worse answers less.

$\sigma$ is the sigmoid function, which compresses the difference into the interval $(0,1)$, and $\log\sigma$ turns it into a loss value. The leading minus sign ensures that the larger the difference, the smaller the loss.

DPO's relationship to KL is that the reference model $\pi_{ref}$ is not just a passive comparison target. It also acts like an anchor. The current model cannot pursue preference data alone; through the probability ratio, it is constrained by the reference model. Intuitively, this is the same kind of idea as the RLHF objective:

$$
J(\pi)=\mathbb{E}_\pi[r(x,y)]-\beta D_{KL}(\pi_\theta\|\pi_{ref})
$$

**Increase the probability of preferred answers, but do not move too far from the reference model**.

---

## Summary

This article applied information-theoretic tools to practical alignment training:

| Concept                   | Problem it solves                                      | Core formula/idea                                    | Role in RL                                            |
| ------------------------- | ------------------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------------- |
| KL constraint             | Preventing policy updates from becoming too aggressive | reward $- \beta D_{KL}$                              | Constrains policy drift in PPO/RLHF                   |
| Cross-entropy-KL identity | Explaining why classification uses cross-entropy       | $D_{KL}(P\|Q)=H(P,Q)-H(P)$                           | Minimizing cross-entropy = minimizing KL              |
| DPO log probability ratio | Preference learning without training a reward model    | $\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$ | Optimizes relative probabilities with preference data |

The three ideas form one narrative line: KL divergence limits policy drift -> the identity between cross-entropy and KL gives the training loss a theoretical basis -> DPO uses log probability ratios to implement KL regularization implicitly and skips the reward model.

> **Next**: [E.4.4 Mutual Information and Representation Learning](./information-mutual-info) -- using information theory to measure representation quality.
