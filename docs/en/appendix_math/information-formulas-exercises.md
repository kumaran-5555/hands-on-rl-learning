---
title: E.4.6 Formulas and Exercises
---

# E.4.6 Information Theory Formula Reference and Exercises

> **Prerequisites**: This page summarizes all formulas in module E.4. It is best reviewed after reading [E.4.1](./information-basics) through [E.4.5](./information-advanced-formulas). If this is your first pass, skip to the main articles first.

---

This page collects all formulas used in module E.4 for review. It is recommended that you read the previous articles first and then use this page as a reference table.

## Information Theory Formulas You Will Meet in This Book

| Concept                       | Formula                                                                                                                                                              | Meaning in reinforcement learning                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Self-information              | $I(x)=-\log p(x)$                                                                                                                                                    | Low-probability events contain more information          |
| Entropy                       | $H(P)=-\sum_x p(x)\log p(x)$                                                                                                                                         | Policy randomness and exploration                        |
| Entropy bonus                 | $J=\mathbb{E}[G]+\beta H(\pi)$                                                                                                                                       | Encourages exploration and avoids premature certainty    |
| Cross-entropy                 | $H(P,Q)=-\sum_x P(x)\log Q(x)$                                                                                                                                       | Classification training and reward model training        |
| KL divergence                 | $D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$                                                                                                                      | Measures differences between old and new policies        |
| Cross-entropy-KL relationship | $D_{KL}(P\|Q)=H(P,Q)-H(P)$                                                                                                                                           | KL is extra encoding cost                                |
| KL penalty                    | $\text{reward}-\beta D_{KL}$                                                                                                                                         | Constrains policy drift in PPO/RLHF                      |
| RLHF objective                | $J(\pi)=\mathbb{E}_\pi[r(x,y)]-\beta D_{KL}(\pi_\theta\|\pi_{ref})$                                                                                                  | Reward maximization with a reference model constraint    |
| DPO loss                      | $\mathcal{L}_{DPO}=-\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)})]$ | Uses preference data to optimize relative probabilities  |
| Mutual information            | $I(X;Y)=H(X)-H(X\mid Y)$                                                                                                                                             | Whether a representation keeps task-relevant information |

---

## Summary

The hierarchy on this page is: start from "smaller probability means more information" and "a more uniform policy has higher entropy," then extend to cross-entropy, KL, the RLHF regularized objective, and the DPO loss. When reading complex information-theoretic formulas, first ask: is this measuring randomness, prediction error, or the distance between two policy distributions?

---

## Common Mistakes

1. **Treating entropy as noise.** High entropy means the policy is more random and may help exploration, but it does not mean the policy is worse.
2. **Treating KL as an ordinary distance.** KL is asymmetric. $D_{KL}(P\|Q)$ and $D_{KL}(Q\|P)$ emphasize different errors.
3. **Thinking the KL constraint is only mathematical decoration.** In RLHF, the KL term directly determines how far the model can move from the reference model.

---

## Exercises

1. Compare $[0.5,0.5]$ and $[0.9,0.1]$. Which has higher entropy? Why?
2. If the old policy is $[0.5,0.5]$ and the new policy is $[0.8,0.2]$, write the expanded expression for $D_{KL}(\pi_{old}\|\pi_{new})$.
3. In the RLHF objective $\mathbb{E}[r]-\beta D_{KL}$, when $\beta$ becomes larger, does the policy update become more aggressive or more conservative?
