---
title: C.8 DAPO
---

# C.8 DAPO

DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) is a GRPO improvement proposed by ByteDance in 2025, and its interview frequency has been rising quickly.

---

## DAPO vs GRPO: Three Improvements

| Improvement      | GRPO                                  | DAPO                                                                     |
| ---------------- | ------------------------------------- | ------------------------------------------------------------------------ |
| clipping         | symmetric `clip(ratio, 1-eps, 1+eps)` | **decoupled clipping**: clip positive/negative advantages separately     |
| sampling         | fixed prompts                         | **dynamic sampling**: filter prompts that are all-correct or all-wrong   |
| overlong penalty | binary (overlong -> reward = 0)       | **progressive penalty**: the longer the excess, the larger the deduction |

---

## Decoupled Clipping

### One-Line Memory

> Don't be greedy on the good, don't be vengeful on the bad: positive advantages clip the upper bound only, negative advantages clip the lower bound only.

### Pseudocode

```
# Step 1: compute the new/old policy ratio
ratio = exp(new_logp - old_logp)

# Step 2: positive advantage — clip only the upper bound (don't let the ratio run too high)
pos_surr = min(ratio, 1 + eps) * advantage    # advantage > 0

# Step 3: negative advantage — clip only the lower bound (let the ratio bounce back)
neg_surr = max(ratio, 1 - eps) * advantage    # advantage < 0

# Step 4: combine and average
loss = -mean(pos_surr + neg_surr)
```

### Intuition

Compare to symmetric clipping:

```
GRPO (symmetric):
  advantage > 0:  min(ratio, 1+eps) * A
  advantage < 0:  max(ratio, 1-eps) * A

DAPO (decoupled):
  advantage > 0:  min(ratio, 1+eps_high) * A
  advantage < 0:  max(ratio, 1-eps_low)  * A
```

This makes it possible to tune exploration differently in the positive and negative directions (for example, more aggressive improvements but more conservative punishment).

### Python (NumPy) Implementation

```python
import numpy as np


def dapo_policy_loss(new_logp, old_logp, advantages, clip_high=0.28, clip_low=0.28):
    """
    new_logp:     [T]
    old_logp:     [T]
    advantages:   [T]
    clip_high:    upper-bound clipping for positive advantages
    clip_low:     lower-bound clipping for negative advantages
    """
    ratio = np.exp(new_logp - old_logp)

    pos_mask = advantages >= 0
    neg_mask = ~pos_mask

    loss = np.zeros_like(advantages)

    # positive: clip only the upper bound
    if pos_mask.any():
        clipped_ratio = np.minimum(ratio[pos_mask], 1 + clip_high)
        loss[pos_mask] = -(clipped_ratio * advantages[pos_mask])

    # negative: clip only the lower bound
    if neg_mask.any():
        clipped_ratio = np.maximum(ratio[neg_mask], 1 - clip_low)
        loss[neg_mask] = -(clipped_ratio * advantages[neg_mask])

    return loss.mean()
```

### PyTorch Implementation

```python
import torch


def dapo_policy_loss(new_logps, old_logps, advantages, clip_high=0.28, clip_low=0.28):
    """
    new_logps:    [B, seq_len]
    old_logps:    [B, seq_len]
    advantages:   [B, seq_len]
    """
    ratio = torch.exp(new_logps - old_logps)

    pos_mask = advantages >= 0
    neg_mask = ~pos_mask

    loss = torch.zeros_like(advantages)

    # positive: min(ratio, 1 + clip_high) * advantage
    if pos_mask.any():
        clipped = torch.clamp(ratio[pos_mask], max=1 + clip_high)
        loss[pos_mask] = -(clipped * advantages[pos_mask])

    # negative: max(ratio, 1 - clip_low) * advantage
    if neg_mask.any():
        clipped = torch.clamp(ratio[neg_mask], min=1 - clip_low)
        loss[neg_mask] = -(clipped * advantages[neg_mask])

    return loss.mean()
```

---

## Dynamic Sampling

### One-Line Memory

> If a whole group of answers is all-correct or all-wrong, there is nothing to compare, so skip that question.

### Pseudocode

```
# Walk every prompt one at a time
for each prompt:
    # score each completion under this prompt
    rewards = [get_reward(completion) for completion in group]

    # if every reward is identical (all right or all wrong) there is no signal -> skip
    if all rewards are the same:
        skip this prompt
```

### PyTorch Implementation

```python
import torch


def dynamic_sampling_filter(rewards):
    """
    rewards: [B, G] where B prompts, each with G completions
    returns: bool mask [B], True means keep this prompt
    """
    reward_std = rewards.std(dim=1)
    return reward_std > 1e-6
```

### Intuition

GRPO uses group-wise z-score normalization. If all rewards in a group are identical, then `std=0` and advantages are undefined or all zeros. DAPO filters those samples at the data level instead of discovering the problem later in the loss computation.

---

## Overlong Reward Shaping

### One-Line Memory

> Don't zero out overlong answers all at once — deduct gradually based on how much they overflow.

### Pseudocode

```
# Step 1: only penalize responses that exceed the max length
if response_length > max_length:
    # Step 2: the more it overflows, the more we deduct (proportional to the overflow)
    penalty_ratio = (response_length - max_length) / max_length
    # Step 3: subtract from the original reward
    reward = reward - penalty_weight * penalty_ratio
```

### Python (NumPy) Implementation

```python
def overlong_reward_shaping(reward, response_length, max_length, penalty_weight=0.1):
    if response_length <= max_length:
        return reward
    penalty = penalty_weight * (response_length - max_length) / max_length
    return reward - penalty
```

### Intuition

Compare to GRPO:

- GRPO: overlong -> reward = 0 (binary, discontinuous)
- DAPO: overlong -> reward decreases linearly (smooth signal)

From an RL view, a binary reward provides little directional signal at the boundary. A linear penalty tells the policy: "shorter would be better."

---

## DAPO Total Loss (Sketch)

```
# 1) group-wise normalization (same as GRPO)
advantages = (rewards - mean) / (std + eps)

# 2) dynamic sampling filter
valid_mask = dynamic_sampling_filter(rewards)

# 3) decoupled-clipping policy loss
policy_loss = dapo_policy_loss(new_logp, old_logp, advantages, clip_high, clip_low)

# 4) KL penalty
kl = kl_penalty(log_probs, ref_log_probs)

# 5) total
loss = policy_loss[valid_mask].mean() + kl_coeff * kl
```

---

## Full Comparison: GRPO vs DAPO

| Dimension               | GRPO                              | DAPO                                                                          |
| ----------------------- | --------------------------------- | ----------------------------------------------------------------------------- |
| clipping                | symmetric `clip(r, 1-eps, 1+eps)` | decoupled; one epsilon for each sign of advantage                             |
| invalid data            | not handled (std=0 -> NaN)        | filtered via dynamic sampling                                                 |
| overlong reward         | binary (0/1 style)                | progressive linear penalty                                                    |
| exploration flexibility | fixed                             | can be more aggressive for positive direction, more conservative for negative |
| representative work     | DeepSeek-R1                       | ByteDance / Tsinghua DAPO                                                     |

---

## Common Pitfalls

| Pitfall                                     | Explanation                                                                          |
| ------------------------------------------- | ------------------------------------------------------------------------------------ |
| Decoupled clipping is not “no clipping”     | Clipping still exists; the positive/negative sides are tuned independently.          |
| Wrong condition for dynamic sampling        | It is not "reward below a threshold"; it is "group reward variance is (near) zero."  |
| Overlong shaping is linear, not exponential | Simple `(len - max_len) / max_len` is enough.                                        |
| Advantages are still group-wise normalized  | This part is exactly the same as GRPO.                                               |
| `clip_high` and `clip_low` can differ       | In interviews: "you can tune exploration strength separately in the two directions." |
