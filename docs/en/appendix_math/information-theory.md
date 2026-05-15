---
title: E.4 Information Theory and Distribution Distance
---

# E.4 Information Theory and Distribution Distance

If you have trained a language model, you have probably seen this situation: the model hesitates between two answers, or after one update step its style suddenly drifts. Behind these problems are two basic questions: "how do we measure how random a distribution is?" and "how do we measure how far apart two distributions are?" The tools that answer them come from information theory.

Information theory began as a foundation of communication, but it appears almost everywhere in reinforcement learning: policy exploration needs entropy, stable PPO updates need KL constraints, RLHF alignment training depends on cross-entropy and KL divergence, and DPO repackages these tools into an elegant preference optimization formula.

This section starts from the simplest probability events and builds up to the mathematical core of RLHF and DPO.

![Policy distributions, entropy, and KL](../../appendix_math/images/rl-policy-distribution.svg)

## Roadmap

| Article                                                                           | Mathematical rhythm                                           | Role in reinforcement learning                                   |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| [E.4.1 Self-Information, Entropy, and Exploration](./information-basics)          | Probability event -> self-information -> entropy              | Measures policy randomness and exploration                       |
| [E.4.2 Cross-Entropy and KL Divergence](./information-cross-entropy-kl)           | Encoding cost -> cross-entropy -> KL                          | Measures differences between prediction and policy distributions |
| [E.4.3 KL Constraints, RLHF, and DPO](./information-rlhf-dpo)                     | KL regularization -> log probability ratio -> preference loss | Understands policy constraints in alignment training             |
| [E.4.4 Mutual Information and Representation Learning](./information-mutual-info) | Reduction in conditional uncertainty -> mutual information    | Measures task-relevant information kept in representations       |
| [E.4.5 Complete Information Theory Formulas](./information-advanced-formulas)     | Full expressions for KL, RLHF, DPO, and mutual information    | Unifies distribution distance and preference optimization        |
| [E.4.6 Summary, Formulas, and Exercises](./information-formulas-exercises)        | Formula review -> common mistakes -> exercises                | Reviews and checks understanding                                 |
