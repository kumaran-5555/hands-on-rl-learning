---
title: C.3 DPO Family
---

# C.3 DPO Family

In post-training interviews, DPO loss is the single most frequently requested “handwritten code” question. Many interviews ask it directly.

---

## DPO Loss

### One-Line Memory

> Push good answers up, push bad answers down — both compared against the reference model.

$$\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\big(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big)\Big)\Big]$$

### Symbols (Know What You Are Writing)

- $x$: prompt (input)
- $y_w$: preferred (winner / chosen) response
- $y_l$: dispreferred (loser / rejected) response
- $\pi_\theta$: current policy model
- $\pi_{ref}$: reference model (frozen)
- $\beta$: temperature-like scaling; larger $\beta$ means stronger sensitivity to preference gaps
- $\sigma(\cdot)$: sigmoid

### Pseudocode

```
# Step 1: grab 4 log-probs — two models x two answers
pi_chosen   = log_pi_theta(y_w | x)        # current model on the good answer
pi_rejected = log_pi_theta(y_l | x)        # current model on the bad answer
ref_chosen  = log_pi_ref(y_w | x)          # reference model on the good answer
ref_rejected = log_pi_ref(y_l | x)         # reference model on the bad answer

# Step 2: for each answer compute the "current vs reference" log ratio
log_ratio_w = pi_chosen  - ref_chosen      # good answer: how much higher is current vs ref
log_ratio_l = pi_rejected - ref_rejected   # bad answer: how much higher is current vs ref

# Step 3: want the good ratio to beat the bad ratio; pass through sigmoid and take negative log
loss = -log_sigmoid(beta * (log_ratio_w - log_ratio_l))
```

### Memory Trick

Break it into four steps:

1. Two models: current $\pi_\theta$ and reference $\pi_{ref}$
2. Two samples: chosen ($y_w$) and rejected ($y_l$)
3. For each sample, compute $\log \frac{\pi_\theta}{\pi_{ref}}$ (a log odds ratio)
4. Chosen minus rejected: push the chosen odds ratio above the rejected one

Whiteboard diagram:

```
pi_theta(chosen)  ──┐
                   ├── diff1 = log_theta_w - log_ref_w
pi_ref(chosen)    ──┘
                        diff1 - diff2 -> beta * -> sigmoid -> -log
pi_theta(rejected) ─┐
                   ├── diff2 = log_theta_l - log_ref_l
pi_ref(rejected)   ─┘
```

### Python (NumPy) Implementation

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def log_sigmoid(x):
    # numerically stable: log(sigmoid(x)) = -log(1 + exp(-x))
    return -np.logaddexp(0, -x)


def dpo_loss(
    logp_chosen,
    logp_rejected,
    logp_ref_chosen,
    logp_ref_rejected,
    beta=0.1,
):
    """
    All inputs can be scalars or shape [B].
    Returns a scalar loss.
    """
    log_ratio_w = logp_chosen - logp_ref_chosen
    log_ratio_l = logp_rejected - logp_ref_rejected
    loss = -log_sigmoid(beta * (log_ratio_w - log_ratio_l))
    return loss.mean()
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """
    All inputs: [B]
    """
    log_ratio_w = policy_chosen_logps - ref_chosen_logps
    log_ratio_l = policy_rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
```

---

## IPO (A Squared-Loss Alternative to DPO)

### One-Line Memory

> Swap DPO's sigmoid for a squared difference: the good-minus-bad gap should approach a fixed value.

IPO does not use log-sigmoid; it directly regresses to a margin around 0.5.

### Pseudocode

```
# Step 1: the good-vs-bad gap in log ratios
delta = log_ratio_chosen - log_ratio_rejected

# Step 2: push delta toward a fixed target 1/(2*beta); squared error does the work
loss = (delta - 1 / (2 * beta))^2
```

### PyTorch Implementation

```python
def ipo_loss(log_ratio_w, log_ratio_l, beta=0.1):
    delta = log_ratio_w - log_ratio_l
    return ((delta - 1.0 / (2 * beta)) ** 2).mean()
```

---

## KTO (No Pairing Required; Only Good/Bad Labels)

### One-Line Memory

> No pairing needed. Push good samples up, push bad samples down; each goes through its own sigmoid.

KTO does not require chosen/rejected pairs. You only need to know whether a single sample is desirable or undesirable.

### Pseudocode

```
# Step 1: for a single sample, compute the "current vs reference" log ratio
log_ratio = log_pi(y|x) - log_pi_ref(y|x)

# Step 2: good samples -> push log_ratio up (above the baseline z_ref)
loss_desirable = -log_sigmoid(beta * (log_ratio - z_ref))

# Step 3: bad samples -> push log_ratio down (below the baseline)
loss_undesirable = -log_sigmoid(-beta * (log_ratio - z_ref))

# Step 4: weighted sum over both classes
loss = w_desirable * loss_desirable + w_undesirable * loss_undesirable
```

Here `z_ref` is a baseline term derived from a KL estimate.

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F


def kto_loss(log_ratio, is_desirable, z_ref=0.0, beta=0.1):
    """
    log_ratio: [B] = log_pi(y|x) - log_ref(y|x)
    is_desirable: [B] bool; True means desirable
    """
    loss = torch.zeros_like(log_ratio)

    desirable = is_desirable
    undesirable = ~is_desirable

    if desirable.any():
        loss[desirable] = -F.logsigmoid(beta * (log_ratio[desirable] - z_ref))
    if undesirable.any():
        loss[undesirable] = -F.logsigmoid(-beta * (log_ratio[undesirable] - z_ref))

    return loss.mean()
```

---

## SimPO (No Reference Model Required)

### One-Line Memory

> DPO without a reference model. Divide log-prob by answer length, then subtract an offset gamma.

### Pseudocode

```
# Step 1: divide log-prob by answer length (so long answers aren't penalized)
logp_w = log_pi(chosen) / len(chosen)
logp_l = log_pi(rejected) / len(rejected)

# Step 2: good minus bad, scale by beta, then subtract a margin gamma
loss = -log_sigmoid(beta * (logp_w - logp_l) - gamma)
```

### PyTorch Implementation

```python
import torch.nn.functional as F


def simpo_loss(
    chosen_logps,
    rejected_logps,
    chosen_lengths,
    rejected_lengths,
    beta=2.0,
    gamma=0.5,
):
    logp_w = chosen_logps / chosen_lengths
    logp_l = rejected_logps / rejected_lengths
    logits = beta * (logp_w - logp_l) - gamma
    return -F.logsigmoid(logits).mean()
```

---

## Quick Comparison: DPO Family

| Method | Needs ref? | Needs pairing?        | Key difference                               |
| ------ | ---------- | --------------------- | -------------------------------------------- |
| DPO    | yes        | yes (chosen/rejected) | log-sigmoid, canonical form                  |
| IPO    | yes        | yes                   | squared loss instead of log-sigmoid          |
| KTO    | yes        | no (good/bad labels)  | single-sample optimization                   |
| SimPO  | no         | yes                   | length normalization + implicit reward shift |
| ORPO   | no         | yes                   | odds ratio; merges SFT + alignment           |

---

## Common Pitfalls

| Pitfall                           | Explanation                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| Mixing up the four log-probs      | Two models times two samples: (chosen, rejected) for both policy and reference.      |
| Numeric overflow in `log_sigmoid` | Use `F.logsigmoid` in PyTorch; in NumPy, use `logaddexp`.                            |
| Misunderstanding `beta`           | Larger `beta` increases sensitivity to preference gaps; common range is ~0.1 to 0.5. |
| Forgot to detach the reference    | `ref_*_logps` should be `.detach()`'d and should not receive gradients.              |
| Swapped chosen vs rejected        | Check the dataset: chosen is the human-preferred response.                           |
| Expecting sigmoid in IPO          | IPO uses a squared loss, not a sigmoid. That is the main difference from DPO.        |
