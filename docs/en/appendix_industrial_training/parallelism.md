---
title: B.6 Parallelism
search: false
---

# B.6 Parallelism

# Legacy Page: Distributed Parallelism (Merged into B.1)

> This page is kept as an entry point for legacy links. The core content has been merged into the "Distributed Parallelism and Memory Optimization" part of [B.1 RL Training Infrastructure: Sampling, Asynchrony, and Distributed Systems](./rl-infrastructure). The original text is preserved below (in English translation) so readers coming from old links can map the content.

PPO training often needs to load four models at the same time (Actor, Critic, Reference, Reward Model). A 7B model in FP16 is roughly 14 GB just for parameters. Four of them is roughly 56 GB, which does not fit comfortably on a single A100 (80 GB) once you include optimizer states, gradients, and runtime overhead. Larger models only make this problem worse.

This section answers one question: **when the model does not fit on one GPU, how do you split it across multiple GPUs?** We start with the four core parallelism strategies, then discuss mixed precision choices, and finally explain why RL training has additional challenges compared with ordinary fine-tuning.

## Four Parallelism Strategies

### Data Parallelism (DP)

The simplest approach. Each GPU holds a full replica of the model, but processes a different data batch. Gradients are synchronized with AllReduce.

Limitation: a single GPU must fit the entire model. For a 7B model in FP16, you need roughly 14 GB for parameters plus optimizer states and gradients; in practice you can easily end up near ~56 GB, which exceeds many consumer GPUs.

### Tensor Parallelism (TP)

Split a single matrix operation across multiple GPUs. For example, for a 4096×4096 matrix multiplication, with 4 GPUs each GPU computes a 1024×4096 slice and the results are concatenated.

- Requires high-bandwidth interconnects such as NVLink (typically within one node)
- Megatron-LM is the industrial-standard implementation
- Suitable for multi-GPU within a node (for example, an 8×H100 machine)

### Pipeline Parallelism (PP)

Split the model by layers. GPU 0 handles layers 1-10, GPU 1 handles layers 11-20, and so on.

A drawback is pipeline "bubbles": later stages can only start after earlier stages produce activations. Micro-batching helps, but cannot eliminate bubbles entirely.

### Expert Parallelism (EP)

Designed for MoE (Mixture-of-Experts) models. Different experts live on different GPUs, and a router decides which expert each token is sent to. Models like Mixtral and DeepSeek-V3 rely on this strategy.

**Typical combinations**: Dense 70B models often use a 3D hybrid of DP+TP+PP. MoE models additionally require EP.

| Strategy | What Is Split        | Communication          | Typical Scope                       |
| -------- | -------------------- | ---------------------- | ----------------------------------- |
| DP       | data                 | gradient AllReduce     | when a single GPU can fit the model |
| TP       | intra-layer matrices | every forward/backward | within a node (needs NVLink)        |
| PP       | across layers        | activation transfers   | across nodes                        |
| EP       | expert networks      | router decisions       | MoE models                          |

## Two Memory-Optimization Schemes

The four strategies above are about _how computation is parallelized_. Another class of techniques focuses on _how memory is saved_:

**FSDP (Fully Sharded Data Parallel)**: native PyTorch. Shards parameters, gradients, and optimizer states across GPUs, and temporarily gathers them during computation. It is widely applicable and usually does not require changing model code.

**DeepSpeed ZeRO**: similar to FSDP, with three stages. ZeRO-1 shards optimizer states, ZeRO-2 also shards gradients, and ZeRO-3 shards everything. ZeRO-3 can train arbitrarily large models, but also has the largest communication overhead.

In practice, FSDP and ZeRO are often combined with TP/PP.

## Choosing Mixed Precision

| Precision | Memory     | Speed     | Stability     | Where It Is Used                                           |
| --------- | ---------- | --------- | ------------- | ---------------------------------------------------------- |
| BF16      | half       | fast      | good          | **preferred for training** (A100/H100 support it natively) |
| FP16      | half       | fast      | overflow risk | requires loss scaling                                      |
| FP32      | baseline   | slow      | best          | critical-precision scenarios                               |
| FP8       | 1/4        | fastest   | weaker        | experimental training                                      |
| INT8/4    | 1/4 to 1/8 | very fast | lossy         | **inference deployment**                                   |

Practical guidance: use BF16 for training, and quantization (INT4/INT8) for inference.

## Extra Challenges in RL Training

RL training (PPO/GRPO) is more complex than ordinary fine-tuning, because you must manage multiple models and two distinct phases.

**The rollout phase is inference-intensive**. For GRPO, a setting like k=16 means each prompt generates 16 answers. You need many GPUs for high-throughput generation. vLLM's PagedAttention can often improve batch inference throughput by 2-4×.

**The training phase is compute-intensive**. Backpropagation consumes substantial GPU compute.

**GPU demand fluctuates across the two phases**: during rollout, training GPUs can idle; during training, inference GPUs can idle. A standard solution is an asynchronous architecture where rollout and training use different GPU groups. See the async-training section in [B.1 RL Training Infrastructure](./rl-infrastructure).

**Common memory-saving tricks**:

| Trick                     | Why It Works                                                       | Savings                       |
| ------------------------- | ------------------------------------------------------------------ | ----------------------------- |
| Share the reference model | The reference model is frozen; it can share weights with the actor | ~25%                          |
| LoRA rollout              | Inference only loads LoRA adapters                                 | ~50%                          |
| Gradient checkpointing    | Recompute instead of storing activations                           | ~40% (trades time for memory) |

## Frontier Topics: New Issues from MoE and PRM

**MoE routing inconsistency**. MoE models have multiple experts. Training frameworks (such as Megatron) and inference frameworks (such as vLLM) can differ slightly in floating-point router computations, which may route the same token to different experts across training vs inference. That makes the gradient direction effectively wrong. One reported approach is called Keep Routing: record routing decisions during inference and force the same routing path during training.

**Extra compute for PRM**. A Process Reward Model scores each step of a reasoning chain, and its compute can be comparable to generation itself. In some systems this adds an additional GPU group for PRM scoring between the inference cluster and the training cluster. Only a few frameworks currently implement this pipeline end-to-end (for example, PRIME-RL).

## References

[^1]: HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026.

[^2]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.

[^3]: OpenRLHF, [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143), EMNLP 2025 Demo.

[^4]: DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
