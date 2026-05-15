---
title: C.7 Attention Mechanism
---

# C.7 Attention Mechanism

Strictly speaking, this is not an RL algorithm. But it is one of the top three most frequent “handwrite the code” questions in LLM interviews, and RL interviews often use it as prerequisite knowledge.

---

## Scaled Dot-Product Attention

### One-Line Memory

> Compute $QK^T$, divide by $\sqrt{d_k}$, apply mask, softmax, then multiply by $V$.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Symbols

- $Q$: queries
- $K$: keys
- $V$: values
- $d_k$: per-head key/query dimension

### Pseudocode

```
scores = Q @ K^T / sqrt(d_k)
scores = scores + mask    # causal: upper triangle -> -inf
attn_weights = softmax(scores, dim=-1)
output = attn_weights @ V
```

### Intuition

Three steps:

1. Score: dot product between $Q$ and $K$ measures similarity. Divide by $\sqrt{d_k}$ to prevent overly large dot products that saturate softmax.
2. Mask: a causal mask sets "future" positions to $-\infty$ (an autoregressive LM can only look left).
3. Weight: softmax weights are used to take a weighted sum of $V$.

### Python (NumPy) Implementation

```python
import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [seq_len, d_k]
    K: [seq_len, d_k]
    V: [seq_len, d_v]
    mask: [seq_len, seq_len] where 0=keep, -inf=mask
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return attn_weights @ V
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [B, heads, seq_len, d_k]
    K: [B, heads, seq_len, d_k]
    V: [B, heads, seq_len, d_v]
    mask: [1, 1, seq_len, seq_len] or [B, 1, 1, seq_len]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def causal_mask(seq_len):
    """Causal mask: lower triangle is 1 (keep), upper triangle is 0 (mask)."""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
```

---

## Multi-Head Attention (MHA)

### One-Line Memory

> Split `d_model` into `h` heads. Each head runs attention independently, then concatenate and apply an output projection.

### Pseudocode

```
Q = x @ W_Q   # [B, seq, d_model] -> [B, seq, d_model]
K = x @ W_K
V = x @ W_V

# split heads: [B, seq, d_model] -> [B, heads, seq, d_k]
Q = Q.view(B, seq, heads, d_k).transpose(1, 2)
K = K.view(B, seq, heads, d_k).transpose(1, 2)
V = V.view(B, seq, heads, d_k).transpose(1, 2)

attn_out = scaled_dot_product_attention(Q, K, V, mask)

# merge heads: [B, heads, seq, d_k] -> [B, seq, d_model]
attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq, d_model)
output = attn_out @ W_O
```

### Intuition

A single attention head can focus on one kind of relation pattern. Multiple heads let the model attend to different patterns and different subspaces in parallel.

Shape mnemonic: "view to split heads, transpose to move head dimension, attention, transpose back, view to merge."

### PyTorch Implementation

```python
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, seq_len, d_model = x.shape

        # linear projections + split heads
        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask)

        # merge heads + output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        return self.W_O(attn_out)
```

---

## MQA and GQA

### Quick Comparison

| Variant | # Q heads | # K/V heads       | K/V parameter size     | Example models     |
| ------- | --------- | ----------------- | ---------------------- | ------------------ |
| MHA     | $h$       | $h$               | $3 \times d_{model}^2$ | GPT-2, BERT        |
| MQA     | $h$       | 1                 | much smaller           | PaLM, StarCoder    |
| GQA     | $h$       | $g$ where $g < h$ | tradeoff               | LLaMA 2/3, Mistral |

### One-Line Memory

- **MQA**: all Q heads share a single K/V head. Best for KV cache, may lose expressiveness.
- **GQA**: split Q heads into `g` groups; heads within a group share K/V. A compromise between MHA and MQA.

### PyTorch Implementation (GQA)

```python
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        n_heads: number of query heads (e.g. 32)
        n_kv_heads: number of key/value heads (e.g. 8)
        n_heads must be divisible by n_kv_heads
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_O = nn.Linear(n_heads * self.d_k, d_model)

    def forward(self, x, mask=None):
        B, seq_len, _ = x.shape

        # Q: [B, seq, n_heads*d_k] -> [B, n_heads, seq, d_k]
        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # K/V: [B, seq, n_kv_heads*d_k] -> [B, n_kv_heads, seq, d_k]
        K = self.W_K(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Expand K/V to match Q head count: [B, n_kv_heads, seq, d_k] -> [B, n_heads, seq, d_k]
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.W_O(attn_out)
```

---

## Follow-Up: Complexity

| Component          | Complexity         | Notes                                      |
| ------------------ | ------------------ | ------------------------------------------ |
| self-attention     | $O(n^2 \cdot d)$   | $n$ is sequence length, $d$ is hidden size |
| linear projections | $O(n \cdot d^2)$   | per-token linear layers                    |
| total (MHA)        | $O(n^2 d + n d^2)$ | when $n$ is large, $n^2$ dominates         |

---

## Common Pitfalls

| Pitfall                                        | Explanation                                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------------- |
| Divide by $\sqrt{d_k}$, not $\sqrt{d_{model}}$ | Use per-head dimension, not full model dimension.                                |
| Causal mask direction                          | `tril` creates a lower triangle of ones (keep) and upper triangle (mask future). |
| `view` after `transpose`                       | `transpose` makes tensors non-contiguous; call `.contiguous()` before `view`.    |
| Using the wrong repeat API for GQA             | Use `repeat_interleave` so adjacent Q heads share the same K/V head.             |
| MQA is a special case of GQA                   | When `n_kv_heads=1`, GQA reduces to MQA.                                         |
