---
title: C. Code Cheatsheet
---

# C. Code Cheatsheet

> Skim this once in the 30 minutes before an interview. For each item, memorize one sentence plus one formula. That is usually enough.

This appendix covers the algorithms that are most frequently asked to be handwritten in LLM post-training / RLHF interviews, ordered roughly by how often they show up. Each topic is presented from four angles:

| View                | What It Is For                                               |
| ------------------- | ------------------------------------------------------------ |
| **One-line memory** | The short mantra you can recite before walking into the room |
| **Pseudocode**      | The whiteboard version                                       |
| **Python**          | Explaining the logic with NumPy / plain Python               |
| **PyTorch**         | The engineering version interviewers often probe             |

## Contents

| Section                                       | Topic                                                          | Frequency |
| --------------------------------------------- | -------------------------------------------------------------- | --------- |
| [C.1 SFT Loss and KL Divergence](./sft-kl)    | autoregressive SFT loss, shift-right, KL estimates             | 4/5       |
| [C.2 PPO Policy Loss and GAE](./ppo-gae)      | clipped surrogate, value loss, reverse-time GAE recursion      | 5/5       |
| [C.3 DPO and Variants](./dpo-family)          | DPO loss, IPO, KTO, SimPO                                      | 5/5       |
| [C.4 GRPO and Reward Models](./grpo-rlvr)     | group-wise normalization in GRPO, Bradley-Terry reward model   | 4/5       |
| [C.5 Softmax and Cross-Entropy](./softmax-ce) | numerically stable softmax, log-sum-exp, CE loss               | 4/5       |
| [C.6 Top-k / Top-p Sampling](./top-k-top-p)   | temperature, top-k, top-p (nucleus) decoding                   | 4/5       |
| [C.7 Attention / MHA / GQA](./attention-mha)  | scaled dot-product attention, multi-head attention, MQA, GQA   | 5/5       |
| [C.8 DAPO](./dapo)                            | decoupled clipping, dynamic sampling, overlong penalty shaping | 3/5       |

## How To Use This Appendix

1. Start by memorizing the one-line mantra. Each file opens with a short sentence that is enough to reconstruct the pseudocode.
2. Prioritize pseudocode. In a whiteboard interview, pseudocode plus clear variable definitions is often sufficient.
3. Use the PyTorch snippet for details. If the interviewer asks about implementation specifics (for example `ignore_index`, `log_sum_exp`, `clamp`), jump to the PyTorch section.
4. Review the “Common Pitfalls.” Each file ends with a short list of high-frequency mistakes. Read those the night before.
