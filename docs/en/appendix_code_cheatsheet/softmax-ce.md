---
title: C.5 Softmax and Cross-Entropy
---

# C.5 Softmax and Cross-Entropy

This is a common warm-up question. Before you write DPO or PPO on the whiteboard, an interviewer may ask you to handwrite a numerically stable softmax and cross-entropy.

---

## Numerically Stable Softmax

### One-Line Memory

> Subtract `max` first, then `exp`. The denominator is the sum of all exponentials.

### Pseudocode

```
x_shifted = x - max(x)
exp_x = exp(x_shifted)
softmax = exp_x / sum(exp_x)
```

### Why This Works

You cannot safely compute `exp(x) / sum(exp(x))` when `x` can be large, because `exp(1000)` overflows to `inf`. After subtracting the maximum element, the largest exponent becomes `exp(0) = 1`, and the rest are in `(0, 1]`, avoiding overflow.

### Python (NumPy) Implementation

```python
import numpy as np


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

# Use the built-in version in real code
probs = F.softmax(logits, dim=-1)


# Handwritten version (interview)
def manual_softmax(x, dim=-1):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x_shifted)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

---

## The Log-Sum-Exp Trick

### One-Line Memory

> $\log\sum\exp(x) = \max(x) + \log\sum\exp(x - \max(x))$.

Follow-up question: how do you compute `log(softmax(x))` without overflow? Answer: do not compute softmax first and then take log; use log-softmax directly.

### Python (NumPy) Implementation

```python
import numpy as np


def log_softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
```

### PyTorch Implementation

```python
import torch.nn.functional as F

# Built-in and numerically stable
log_probs = F.log_softmax(logits, dim=-1)


def manual_log_softmax(x, dim=-1):
    max_val = x.max(dim=dim, keepdim=True).values
    return x - max_val - torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))
```

---

## Cross-Entropy Loss

### One-Line Memory

> Negative log-probability under a one-hot target: $-\sum_k y_k \log p_k$. With integer labels: $-\log p_y$.

### Pseudocode

```
log_probs = log_softmax(logits)
loss = -log_probs[target].mean()
```

### Intuition

Cross-entropy measures how much probability the model assigns to the correct class. If the model is confident and correct, $p_y$ is large, so $-\log p_y$ is small. Smaller loss means better predictions.

### Python (NumPy) Implementation

```python
import numpy as np


def cross_entropy(logits, targets, ignore_index=-100):
    """
    logits:  [N, C]
    targets: [N] integer class labels
    """
    log_probs = log_softmax(logits, axis=-1)
    total, count = 0.0, 0
    for i in range(len(targets)):
        if targets[i] == ignore_index:
            continue
        total += -log_probs[i, targets[i]]
        count += 1
    return total / max(count, 1)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F


def manual_cross_entropy(logits, targets, ignore_index=-100):
    """
    logits:  [B, C]
    targets: [B]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # gather selects log-prob at target index
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # mask out ignore_index
    mask = targets != ignore_index
    return -target_log_probs[mask].mean()
```

---

## Common Pitfalls

| Pitfall                           | Explanation                                                                                                     |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Forgot to subtract max in softmax | This is the first thing interviewers look for.                                                                  |
| Softmax then log                  | Numerically unstable. Use `log_softmax` directly.                                                               |
| Computing CE from probabilities   | Do not do `softmax -> log -> CE` manually in real code; use `F.cross_entropy(logits, targets)` which is stable. |
| `ignore_index` handling           | In SFT loss questions, interviewers often ask how you handle padding/prompt tokens.                             |
| Temperature scaling               | Do `logits / temperature` before softmax. Larger $T$ produces a flatter distribution.                           |
