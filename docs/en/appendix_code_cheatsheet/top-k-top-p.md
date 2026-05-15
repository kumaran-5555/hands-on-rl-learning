---
title: C.6 Sampling Methods
---

# C.6 Sampling Methods

Decoding strategies are a frequent LLM interview topic, and they connect directly to RL: after RLHF, how do you sample from the policy? How does temperature change the action distribution?

---

## Temperature

### One-Line Memory

> Divide logits by $T$ and then apply softmax. Larger $T$ means more randomness; smaller $T$ means more determinism.

### Pseudocode

```
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
```

### Intuition

- $T \to 0$: approaches argmax (greedy), similar to a deterministic policy
- $T = 1$: original distribution
- $T \to \infty$: approaches uniform, similar to a highly random policy

From an RL view, temperature is a knob for exploration.

### PyTorch Implementation

```python
import torch


def sample_with_temperature(logits, temperature=1.0):
    if temperature < 1e-8:
        return logits.argmax(dim=-1)  # greedy
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-k Sampling

### One-Line Memory

> Keep only the top-k tokens by probability (or logits). Set the rest to `-inf`, then softmax and sample.

### Pseudocode

```
top_k_values, top_k_indices = topk(logits, k)
logits[not in top_k] = -inf
probs = softmax(logits)
sample from probs
```

### Python (NumPy) Implementation

```python
import numpy as np


def top_k_filtering(logits, k):
    """
    logits: [vocab_size]
    returns: filtered logits (non-top-k positions set to -inf)
    """
    if k >= len(logits):
        return logits
    threshold = np.sort(logits)[-k]  # kth largest value
    return np.where(logits >= threshold, logits, -np.inf)
```

### PyTorch Implementation

```python
import torch


def top_k_filtering(logits, k):
    """
    logits: [B, vocab_size] or [vocab_size]
    """
    if k <= 0:
        return logits
    top_k = min(k, logits.size(-1))
    # Use the kth largest value as threshold
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_k_sample(logits, k, temperature=1.0):
    logits = top_k_filtering(logits, k)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-p (Nucleus) Sampling

### One-Line Memory

> Sort tokens by probability from high to low, accumulate until the mass exceeds $p$, and keep only that prefix.

### Pseudocode

```
sorted_logits = sort_desc(logits)
sorted_probs = softmax(sorted_logits)
cumulative_probs = cumsum(sorted_probs)

# positions past the nucleus are set to -inf
cutoff_mask = cumulative_probs - sorted_probs > p
sorted_logits[cutoff_mask] = -inf

# restore order, softmax, sample
```

### Intuition

Top-k keeps a fixed number of tokens. Top-p keeps a variable number of tokens but a fixed probability mass. Top-p adapts to distribution sharpness:

- If the model is confident, a few tokens are enough to reach mass $p$.
- If the model is uncertain, it may need many tokens to reach mass $p$.

Common interview comparison:

|                | Top-k                       | Top-p                                       |
| -------------- | --------------------------- | ------------------------------------------- |
| Selection rule | keep exactly k tokens       | keep tokens whose cumulative mass reaches p |
| Adaptivity     | does not adapt to sharpness | adapts automatically                        |
| Extremes       | k=1 -> greedy               | p=0 -> greedy, p=1 -> no restriction        |

### Python (NumPy) Implementation

```python
import numpy as np


def top_p_filtering(logits, p):
    """
    logits: [vocab_size]
    """
    sorted_indices = np.argsort(logits)[::-1]  # descending
    sorted_logits = logits[sorted_indices]
    sorted_probs = np.exp(sorted_logits - sorted_logits.max())
    sorted_probs = sorted_probs / sorted_probs.sum()
    cumulative_probs = np.cumsum(sorted_probs)

    # Keep at least one token: use (cumsum - prob) > p
    cutoff = cumulative_probs - sorted_probs > p
    sorted_logits[cutoff] = -np.inf

    # restore original order
    result = np.empty_like(logits)
    result[sorted_indices] = sorted_logits
    return result
```

### PyTorch Implementation

```python
import torch


def top_p_filtering(logits, p):
    """
    logits: [B, vocab_size]
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens beyond nucleus (keep at least one)
    sorted_mask = (cumulative_probs - sorted_probs) > p
    sorted_logits[sorted_mask] = float("-inf")

    # Restore original order
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_p_sample(logits, p, temperature=1.0):
    logits = top_p_filtering(logits, p)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Typical Combined Usage

In practice, temperature + top-k + top-p are often combined:

```python
import torch


def generate_sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    # 1) temperature scaling
    logits = logits / max(temperature, 1e-8)
    # 2) top-k filtering
    logits = top_k_filtering(logits, top_k)
    # 3) top-p filtering
    logits = top_p_filtering(logits, top_p)
    # 4) sample
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Common Pitfalls

| Pitfall                          | Explanation                                                                         |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| Wrong cumsum direction for top-p | You must sort in descending order before cumsum. Ascending order is meaningless.    |
| Not keeping at least one token   | Use `cumsum - current_prob > p`, not `cumsum > p`, or you may remove the top token. |
| Wrong top-k threshold            | Use `topk().values[..., -1]` to get the kth largest value.                          |
| Forgot to restore order          | After sorting for top-p, scatter back to original positions.                        |
| `temperature=0` edge case        | Treat as argmax (greedy). Do not divide by zero.                                    |
