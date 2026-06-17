---
title: C.4 GRPO and Reward Models
---

# C.4 GRPO and Reward Models

---

## GRPO Loss

### One-Line Memory

> No critic. Let the group decide who's good.

### Pseudocode

```
# Step 1: for one prompt, let the model generate G completions
completions = [generate(prompt) for _ in range(G)]

# Step 2: score every completion
rewards = [reward_fn(c) for c in completions]   # [G]

# Step 3: normalize inside the group (subtract mean, divide by std) -> treat as the advantage
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)  # [G]

# Step 4: PPO clipped loss (advantage comes from step 3, not a critic)
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
policy_loss = -min(surr1, surr2).mean()

# Step 5: add a KL penalty (pull back, don't drift too far from the reference model)
kl = kl_penalty(log_probs, ref_log_probs)

# Step 6: total loss
loss = policy_loss + kl_coeff * kl
```

### What To Say In An Interview

GRPO stands for **G**roup **R**elative **P**olicy **O**ptimization. Compared to PPO:

|                              | PPO                                              | GRPO                                         |
| ---------------------------- | ------------------------------------------------ | -------------------------------------------- |
| Where advantages come from   | critic value $V(s)$ + GAE                        | group-wise normalized rewards                |
| How many models are involved | often 4 (actor, critic, reference, reward model) | often 2-3 (actor, reference, RM or verifier) |
| KL penalty                   | optional                                         | almost always used                           |
| Sampling style               | single rollout per prompt                        | $G$ samples for the same prompt              |

Mnemonic: "PPO without the critic; replace advantages with group-wise z-score; everything else is basically copied."

### Python (NumPy) Implementation

```python
import numpy as np


def grpo_advantages(rewards):
    """
    rewards: [num_prompts, G]
    returns: [num_prompts, G] z-scored within each prompt group
    """
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + 1e-8)


def grpo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """Identical to PPO's clipped surrogate loss."""
    ratio = np.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch Implementation

```python
import torch


def grpo_loss(
    log_probs,
    old_log_probs,
    ref_log_probs,
    rewards,
    clip_eps=0.2,
    kl_coeff=0.05,
):
    """
    log_probs:     [B, G, seq_len] current policy
    old_log_probs: [B, G, seq_len] behavior policy (used during sampling)
    ref_log_probs: [B, G, seq_len] reference policy
    rewards:       [B, G]          group-wise rewards

    B = number of prompts, G = group size
    """
    B, G = rewards.shape

    # 1) Group-wise normalization
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)
    advantages = advantages.unsqueeze(-1)  # [B, G, 1], broadcast over seq_len if needed

    # 2) Sequence-level log-prob sums (one per completion)
    seq_logp = log_probs.sum(dim=-1)  # [B, G]
    seq_old = old_log_probs.sum(dim=-1)
    seq_ref = ref_log_probs.sum(dim=-1)

    # 3) Clipped policy loss
    ratio = torch.exp(seq_logp - seq_old)
    adv = advantages.squeeze(-1)  # [B, G]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # 4) KL penalty (k3 estimator: r = π_ref / π_θ, samples from π_θ)
    log_ratio = seq_ref - seq_logp
    kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()

    return policy_loss + kl_coeff * kl
```

---

## Reward Model (Bradley-Terry)

### One-Line Memory

> Train the reward model to score good answers higher than bad ones. One line: `-log_sigmoid(r_chosen - r_rejected)`.

### Pseudocode

```
# Step 1: reward model scores both answers
r_w = reward_model(chosen_input)      # score for the good answer
r_l = reward_model(rejected_input)    # score for the bad answer

# Step 2: we want the good score higher than the bad one; sigmoid + negative log
loss = -log(sigmoid(r_w - r_l))
```

### Intuition

The Bradley-Terry model assumes the probability that humans prefer $y_w$ over $y_l$ is:

$$P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$$

Maximizing the log probability is equivalent to minimizing `-log_sigmoid(diff)`.

Mnemonic: "Reward model training is pairwise cross-entropy."

### Python (NumPy) Implementation

```python
import numpy as np


def log_sigmoid(x):
    return -np.logaddexp(0, -x)


def reward_model_loss(r_chosen, r_rejected):
    """r_chosen, r_rejected: [B]"""
    return -log_sigmoid(r_chosen - r_rejected).mean()
```

### PyTorch Implementation

```python
import torch.nn.functional as F


def reward_model_loss(r_chosen, r_rejected):
    """
    r_chosen:   [B] reward scores for chosen
    r_rejected: [B] reward scores for rejected
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

---

## Interview Follow-Up: DPO vs RLHF-PPO

Interviewers often ask: "What are the pros/cons of DPO compared to PPO-based RLHF?" This table is a good answer:

| Dimension               | PPO-RLHF                      | DPO                                |
| ----------------------- | ----------------------------- | ---------------------------------- |
| Needs a reward model    | yes                           | no (implicit)                      |
| Needs a critic          | yes                           | no                                 |
| Needs a reference model | optional                      | required                           |
| Online vs offline       | online (requires sampling)    | offline (preference data only)     |
| Training cost           | high (often 4 models)         | lower (often 2 models)             |
| Reward hacking risk     | present (RM can be exploited) | lower (no explicit RM in training) |
| Theoretical optimum     | stronger (can keep exploring) | limited by offline data quality    |
| Best-fit use case       | large-scale online training   | when preference data is abundant   |

---

## Common Pitfalls

| Pitfall                                           | Explanation                                                                                             |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| GRPO advantages are group-wise                    | Not global normalization. Compare only the $G$ completions from the **same prompt**.                    |
| No value loss in GRPO                             | There is no critic, so there is no value loss. That is the key difference from PPO.                     |
| Reward model must be frozen during policy updates | When training the RM, you backprop through it; when training the policy, RM is usually frozen.          |
| KL is sequence-level in practice                  | Commonly compute KL after summing token log-probs over the completion, not per-token.                   |
| DPO implicitly learns a reward difference         | `log_ratio_w - log_ratio_l` acts like an implicit reward delta.                                         |
| Choosing $G$                                      | Typical range is 4-16. Too small gives noisy advantages; too large increases sampling cost.             |
| RLVR case                                         | Rewards may come from a rule-based verifier (code execution, math answer check), not from a learned RM. |
