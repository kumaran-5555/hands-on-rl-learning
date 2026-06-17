---
title: C.2 PPO and GAE
---

# C.2 PPO and GAE

PPO is one of the most frequently tested algorithms in LLM RL interviews. Interviewers often ask you to write the **clipped policy loss**, and then follow up with the value loss and GAE.

---

## GAE (Generalized Advantage Estimation)

### One-Line Memory

> Sweep backward from the last step: today's reward plus the discounted future advantage.

GAE is prerequisite knowledge for PPO, and is often asked on its own.

### Pseudocode

```
# Step 1: compute the TD error delta_t
#   = current reward + discounted next-step value estimate - current value estimate
delta_t = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t

# Step 2: sweep backward — add this step's delta plus the discounted "future advantage"
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}

# Step 3: return = advantage + current value estimate (this trains the critic)
return_t = advantage_t + value_t
```

### Intuition

You can view GAE as an exponentially weighted moving average over TD errors:

- $\lambda = 0$: reduces to one-step TD (only $\delta_t$), low variance, higher bias
- $\lambda = 1$: reduces to Monte Carlo-style returns (sums many $\delta$), higher variance, lower bias

Mnemonic: "larger $\lambda$ means you dare to look further into the future."

### Python (NumPy) Implementation

```python
import numpy as np


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T+1] (the last element is the bootstrap value)
    dones:   [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values[:T]
    return advantages, returns
```

### PyTorch Implementation

```python
import torch


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [B, T]
    values:  [B, T+1]
    dones:   [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B)

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
        last_adv = delta + gamma * lam * (1 - dones[:, t]) * last_adv
        advantages[:, t] = last_adv

    returns = advantages + values[:, :T]
    return advantages, returns
```

---

## PPO Clipped Policy Loss

### One-Line Memory

> New policy / old policy = ratio. Ratio \* advantage = target. If the ratio leaves the allowed band, clip it; take the smaller target.

$$L^{CLIP} = -\min\big(r_t(\theta) \cdot A_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)$$

### Pseudocode

```
# Step 1: compute the new/old policy ratio (subtract logs then exp — more numerically stable)
ratio = exp(new_log_prob - old_log_prob)

# Step 2: build two surrogates — one unclipped, one clipped to the band
surr1 = ratio * advantage
surr2 = clip(ratio, 1-eps, 1+eps) * advantage

# Step 3: take the smaller (more conservative) one, negate it to minimize
loss = -min(surr1, surr2).mean()
```

### Intuition

Think of `ratio` on a number line:

- When `advantage > 0`: improving the action should be encouraged, but once `ratio > 1+eps` we stop being greedy.
- When `advantage < 0`: discouraging the action is fine, but once `ratio < 1-eps` we stop being vengeful.

Mnemonic: "positive advantage clips the top; negative advantage clips the bottom; `min` keeps it conservative."

### Python (NumPy) Implementation

```python
import numpy as np


def ppo_policy_loss(new_logp, old_logp, advantages, clip_eps=0.2):
    """
    new_logp:   [T] log-probs under the current policy
    old_logp:   [T] log-probs under the behavior (sampling) policy
    advantages: [T]
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch Implementation

```python
import torch


def ppo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """
    new_logps:  [B, T]
    old_logps:  [B, T]
    advantages: [B, T]
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()
```

---

## PPO Value Loss

### One-Line Memory

> How far the guess is from the target: square it, then average. Optional: don't let the new prediction stray too far from the old one.

### Pseudocode

```
# Step 1: critic produces a fresh prediction
value_pred = critic(state)

# Step 2: clip how far the prediction can move; this gives a "conservative" version
value_clipped = old_values + clip(value_pred - old_values, -eps, eps)

# Step 3: compute MSE for both versions and take the larger one (more conservative)
loss1 = (value_pred - returns)^2
loss2 = (value_clipped - returns)^2
loss = max(loss1, loss2).mean()
```

### PyTorch Implementation

```python
import torch


def ppo_value_loss(values, old_values, returns, clip_eps=0.2):
    loss1 = (values - returns) ** 2
    values_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    loss2 = (values_clipped - returns) ** 2
    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## Total PPO Loss

```
total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
```

If the interviewer asks for the full “three-piece set,” it is:

| Component           | Purpose                 | Typical coefficient |
| ------------------- | ----------------------- | ------------------- |
| clipped policy loss | update the actor/policy | weight 1.0          |
| value loss (MSE)    | update the critic       | `vf_coef=0.5`       |
| entropy bonus       | encourage exploration   | `ent_coef=0.01`     |

---

## Common Pitfalls

| Pitfall                       | Explanation                                                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| Using division for `ratio`    | Prefer `exp(logp_new - logp_old)`; it is more numerically stable.                             |
| Advantages not normalized     | In practice, advantages are often normalized within a batch.                                  |
| `min` vs `max` confusion      | Policy loss uses `min` (conservative). Value loss uses `max` (also conservative).             |
| Forgot to stop gradients      | `old_log_probs` and `old_values` should be `.detach()`'d.                                     |
| Missing `done` masking in GAE | When `done=1`, cut the recursion: multiply by `(1-done)`.                                     |
| Missing bootstrap value       | `values` should have length `T+1`; the last value is the bootstrap.                           |
| Wrong entropy sign            | Entropy is positive; use `- entropy_coeff * entropy` in the loss to encourage higher entropy. |
