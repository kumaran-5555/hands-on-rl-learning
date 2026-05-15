---
title: C.1 SFT and KL
---

# C.1 SFT and KL

## SFT Loss (Autoregressive Cross-Entropy)

### One-Line Memory

> Shift the input right by one token to form targets, and compute cross-entropy only on the answer region (`label != -100`).

### Pseudocode

```
logits = model(input_ids)          # [B, seq_len, vocab_size]
shift_logits = logits[:, :-1, :]   # drop the last prediction position
shift_labels = input_ids[:, 1:]    # drop the first token

loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

### Why Shift Right?

An autoregressive model predicts the token at position $t+1$ from the prefix up to position $t$. Therefore, the logits at index $t$ correspond to the labels at index $t+1$.

A simple mnemonic: "cut the tail of logits, cut the head of labels."

### Python (NumPy) Implementation

```python
import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [seq_len, vocab_size]
    labels: [seq_len] (unshifted; we shift inside)
    """
    shift_logits = logits[:-1]  # drop tail
    shift_labels = labels[1:]  # drop head

    probs = softmax(shift_logits, axis=-1)
    total, count = 0.0, 0

    for t in range(len(shift_labels)):
        if shift_labels[t] == ignore_index:
            continue
        total += -np.log(probs[t, shift_labels[t]] + 1e-12)
        count += 1

    return total / max(count, 1)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F


def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, seq_len, vocab_size]
    labels: [B, seq_len] (typically the original input_ids; we shift inside)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss
```

---

## KL Divergence Estimates

### One-Line Memory

> $\mathrm{KL}(p \| q) = \mathbb{E}_p[\log p - \log q]$. Two estimators are common: a biased `log_ratio` form, and an always-nonnegative “unbiased” form `ratio - 1 - log_ratio`.

Interview-style questions:

- How do you compute KL in PPO?
- How do you compute KL in GRPO?
- What is the difference between these estimates?

### Pseudocode

```
# Method 1: biased estimate (simple; common in PPO)
kl = (log_prob - ref_log_prob).mean()

# Method 2: nonnegative estimate (common in GRPO / trl)
ratio = exp(log_prob - ref_log_prob)
kl = (ratio - 1 - (log_prob - ref_log_prob)).mean()
```

### Python (NumPy) Implementation

```python
import numpy as np


def kl_biased(log_p, log_q):
    """Biased estimate: E_p[log p - log q]. Simple, but can be negative with few samples."""
    return np.mean(log_p - log_q)


def kl_unbiased(log_p, log_q):
    """Nonnegative estimate: E_p[exp(log p - log q) - 1 - (log p - log q)]."""
    log_ratio = log_p - log_q
    return np.mean(np.exp(log_ratio) - 1 - log_ratio)
```

### PyTorch Implementation

```python
import torch


def kl_penalty(log_probs, ref_log_probs, mode="unbiased"):
    """
    log_probs:     [B, seq_len] log-probabilities under the current policy
    ref_log_probs: [B, seq_len] log-probabilities under the reference policy
    """
    if mode == "biased":
        return (log_probs - ref_log_probs).mean()

    log_ratio = log_probs - ref_log_probs
    return (torch.exp(log_ratio) - 1 - log_ratio).mean()
```

### What Is the Difference?

| Estimator   | Formula                                            | Notes                                                |
| ----------- | -------------------------------------------------- | ---------------------------------------------------- |
| Biased      | $\mathbb{E}_p[\log \frac{p}{q}]$                   | simple, but may become negative with limited samples |
| Nonnegative | $\mathbb{E}_p[\frac{p}{q} - 1 - \log \frac{p}{q}]$ | guaranteed $\ge 0$, commonly used in GRPO            |

---

## Common Pitfalls

| Pitfall                  | Explanation                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Shift direction is wrong | Cut the **tail** of logits and the **head** of labels.                                                             |
| Forgot `ignore_index`    | Prompt tokens should not contribute to the loss; they are usually masked with `-100`.                              |
| KL arguments swapped     | In $\mathrm{KL}(p \| q)$, $p$ is the current policy and $q$ is the reference policy. Swapping them flips the sign. |
| Softmax overflow         | Subtract `max(x)` before `exp`. This is expected in interviews.                                                    |
| Missing `.contiguous()`  | In PyTorch, slicing can create non-contiguous tensors; `view` then fails unless you call `.contiguous()`.          |
