---
title: B.2 Agentic RL Infrastructure
---

# B.2 Agentic RL Infrastructure: Sandboxes, Multi-Turn Trajectories, and Tool Scheduling

> B.1 focuses on the RL training-system substrate: rollout, buffers, trainers, weight synchronization, and distributed parallelism. This page draws a different boundary: an agent's "action" is no longer just token generation. It may call tools, run code, read and write files, browse the web, or change external state over multiple turns.

## Division of Labor with B.1

B.2 will not repeat how vLLM/SGLang achieve high-throughput generation, nor how FSDP, ZeRO, TP, PP, and EP split a model across multiple GPUs. Those are B.1 topics. B.2 only covers the additional engineering problems introduced by Agentic RL beyond ordinary LLM RL.

| Question                  | B.1 Training Infrastructure                                                               | B.2 Agentic RL Infrastructure                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Meaning of rollout        | prompts enter the model, producing a completion or trajectory                             | the model acts across multiple turns; each step may call tools, execute code, or change the environment           |
| Sample structure          | token, mask, logprob, reward, policy version, rollout batch                               | episode, tool call, tool result, environment snapshot, step-level reward, loss mask                               |
| Main waiting time         | wait for a rollout batch before training, or wait for weights to flow back after training | tool execution, file I/O, network responses, and sandbox startup all create GPU idle time                         |
| System focus              | buffer depth, async training, staleness, weight sync, model parallelism                   | sandbox isolation, multi-turn trajectory storage, intra-batch pipelining, environment interfaces, reproducibility |
| Representative frameworks | OpenRLHF, veRL, slime                                                                     | Relax, AReaL, Agent-R1, NeMo Gym                                                                                  |

One-sentence distinction: B.1 answers "how does the system feed samples into the model for training"; B.2 answers "when an agent's actions actually happen in the external world, how do we collect them safely, reproducibly, and with high throughput."

## From Single-Turn Generation to Multi-Turn Action

Consider GRPO training (Chapter 8) on math tasks: given a question, the model generates a full solution in one shot, and a verifier checks correctness. The entire process is single-turn. The model's only operation is text generation, and the computation is GPU-bound.

By contrast, training an agent that can fix bugs requires a different interaction pattern. Given buggy code, the model must read files, locate the issue, edit code, run tests, and iterate if tests fail. A single task may require five or six turns. Between turns, there is waiting: reading files depends on disk I/O, running tests depends on sandbox execution, and searching depends on network latency. These operations are not on the GPU, and their latency ranges from tens of milliseconds to seconds.

This creates a chain of coupled engineering problems. If the agent executes code, how do we ensure safety? That requires **sandbox isolation**. Once interaction becomes multi-turn, how do we store training data that is far more complex than a rollout batch? That requires **multi-turn trajectory storage**. When each turn introduces hundreds of milliseconds of waiting, how do we avoid wasting GPU compute? That requires **GPU scheduling optimization**. These are not problems you solve by simply adding more GPUs; they come from the environment-execution nature of agent actions. We will follow this chain step by step.

## Sandbox Isolation

One core capability of an agent is executing code, and that is also the largest safety risk. During training, the model will try many strategies to get higher reward. Without constraints, it might generate `os.system("rm -rf /")` and delete files on the training server, or read API keys from environment variables. These are not "malicious intent" in a human sense; they are the model exploring the action space. The outcome, however, is catastrophic: a running training job can be destroyed, files wiped, and data lost.

Therefore, agent code execution must happen inside an isolated environment (a sandbox). The isolation design must balance security, startup overhead, and resource utilization.

### Comparing Isolation Options

In practice, there are four mainstream sandbox options, suited to different scenarios:

| Option                       | Isolation Granularity            | Startup Latency | When It Fits                                     |
| ---------------------------- | -------------------------------- | --------------- | ------------------------------------------------ |
| subprocess + resource limits | process-level                    | ~10 ms          | prototyping, trusted environments                |
| Docker containers            | filesystem + network + resources | ~100 ms         | general training, full isolation                 |
| MicroVM (Firecracker)        | kernel-level                     | ~125 ms         | security-sensitive scenarios, stronger isolation |
| WebAssembly (Wasm)           | instruction-set level            | ~1 ms           | pure compute tasks, ultra-low latency            |

**subprocess + resource limits** is the lightest option. You can restrict CPU time and memory with `rlimit`, restrict filesystem access with `chroot`, and restrict network namespaces with `unshare`. Isolation strength is limited (processes still share the host kernel), but startup overhead is extremely low, making it suitable for early prototyping:

```python
import resource
import subprocess


def run_in_subprocess(code, timeout=10, max_memory=256 * 1024 * 1024):
    """Lightweight isolation: subprocess + resource limits."""

    def set_limits():
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))

    result = subprocess.run(
        ["python", "-c", code],
        timeout=timeout,
        preexec_fn=set_limits,
        capture_output=True,
        text=True,
    )
    return result
```

**Docker containers** are the most common option for industrial training. Linux cgroups and namespaces provide a triple isolation layer (filesystem, network, and resources), and the image ecosystem is mature (you can directly use base images like `python:3.11-slim` or `node:20`). The main cost is container startup (~100 ms), which can be mitigated with a warm container pool:

```python
container = client.containers.run(
    "python:3.11-slim",
    command=f"python -c '{code}'",
    detach=True,
    mem_limit="512m",  # memory limit
    cpu_quota=50000,  # CPU limit
    network_mode="none",  # block networking
    remove=True,
)
```

**MicroVMs (for example, Firecracker)** provide kernel-level isolation: each VM runs its own minimal Linux kernel, so even if one VM is compromised it cannot affect the host or other VMs. AWS Lambda and Fly.io sandboxes are built on similar ideas. Startup latency is roughly ~125 ms (similar to Docker), but the security boundary is stronger. This is a good fit when untrusted code execution is part of training.

**WebAssembly (Wasm)** provides an instruction-set-level sandbox via WASI (WebAssembly System Interface). After compiling code into Wasm bytecode, it can only call host functions that are explicitly exported; it cannot access the filesystem or network by default. Startup latency can be ~1 ms, but the ecosystem is still maturing and not all Python packages are supported.

### Network Policy

No matter which isolation option you choose, network access must be tightly controlled. Agents in training should not directly access the public internet. It is unsafe, and it breaks reproducibility: if the same agent re-runs the same task later, a search engine might return different results, making trajectories irreproducible. For scenarios that genuinely require networking (for example, web agents), use a proxy layer for request filtering and caching, rather than simply enabling full outbound access.

### Warm Container Pools

If you use Docker, container startup cost is non-trivial. If you run 1000 episodes and each creates a fresh container, startup alone can take about a minute. The industrial approach is to maintain a **warm container pool**: pre-create N containers, recycle and reset them after use rather than destroying them. Startup overhead can drop to about ~5 ms.

Sandboxes solve the problem of "agents can execute actions safely." Next, multi-turn interaction produces large amounts of structured training data, and that data must be stored and managed properly.

## Multi-Turn Trajectory Storage

### Data Structure Differences: LLM RL vs Agentic RL

In B.1, an LLM RL sample is not just raw text. Real systems record token ids, attention masks, response masks, old logprobs, policy version, reward, and more. But structurally, it is still close to a linear sequence:

`prompt -> completion -> reward`

Agentic RL data looks more like a stateful conversation tree. An episode may contain 7-8 turns. Each turn includes model outputs, tool-call arguments, tool results, environment state changes, and step-level rewards. In a "fix a Python bug" task, for example, the model reads code, edits it, runs tests and sees failures, edits again, and runs tests until they pass. All of those interactions need to be recorded faithfully.

### Storage Requirements

Compared with a linear rollout batch, an Agentic RL storage system typically needs three additional capabilities:

- **Index by task type**: for example, analyzing the pattern "good at math but bad at code"
- **Slice by step**: pinpoint exactly which decision step failed
- **Deduplication and expiration**: avoid repeatedly training the same task; old trajectories may become invalid as environments change

At small scale (under 10k trajectories), JSON files plus SQLite can work. At medium scale (10k to 1M), Redis can be used for indexing and S3 for data storage. Beyond 1M, you will likely need a distributed database (MongoDB or DynamoDB). For multimodal agents, trajectories may also include images and audio. In that case, store references (URLs) rather than raw blobs, and download on demand during training, keeping the trajectory index at the KB level.

Once storage is handled, the next bottleneck appears in training: multi-turn interaction introduces large amounts of waiting time, which can make GPUs sit idle.

## GPU Idle Time and Asynchronous Scheduling

### Quantifying the Problem

B.1 discussed GPU idling in LLM RL: generation and training run serially, and training GPUs spend large fractions of time waiting for rollout to finish. Agentic RL makes the problem worse, and the waiting moves inside the trajectory.

Consider a single agentic trajectory timeline: generating an action on the GPU might take ~3 ms, but executing a tool on CPU might take ~500 ms. During those 500 ms, the GPU is idle. The next turn is similar: GPU ~3 ms, CPU ~300 ms. After multiple turns, the GPU may be doing useful work less than 1% of the time. In LLM RL, idling is primarily between rollout and training; in Agentic RL, idling happens within each turn.

### A Simple Fix: Pipeline Multiple Trajectories Within a Batch

A practical approach is to run multiple trajectories concurrently. While trajectory A waits for a tool result, the GPU generates an action for trajectory B; while trajectory B waits, the GPU generates for trajectory C. With pipelined scheduling, the GPU stays busy. This can raise GPU utilization from around ~1% to 70-80%, and increase throughput by 50-100×.

### Two-Level Asynchrony

The approach above fixes concurrency _within a batch_. But across batches, you still have the B.1 problem: rollout and training can be serialized. A complete solution uses two levels of asynchrony: intra-batch concurrency across trajectories (GPU and tools alternate), and inter-batch decoupling of rollout and training via a data queue (rollout continuously produces, training continuously consumes). The first level is specific to Agentic RL; the second level reuses the training-system substrate discussed in B.1.

At this point, we have covered the three core engineering problems for Agentic RL: safe execution, data storage, and GPU scheduling. How are these solutions organized in real industrial systems? The Relax case study below provides a concrete reference implementation.

## An Industrial-Grade Implementation: Relax

Relax is an open-source omni-modal Agentic RL post-training framework released by the Xiaohongshu AI Infra team. It is one of the few engines that supports omni-modal (text, image, and audio) Agentic RL training. We analyze it from four angles: architecture, data flow, execution modes, and engineering details.

### Disaggregated Architecture

Relax's central design choice is to deploy each role in the training pipeline as an independent Ray Serve service: Actor, Rollout, Critic, Reference, Advantages, and GenRM run independently. This reflects the heterogeneity of Agentic RL components: inference needs GPUs, tool execution needs CPUs, and orchestration needs CPU and memory. Independent services allow each component to scale elastically and recover independently, avoiding resource contention.

```
┌───────────────────────────────────────────────────────────────┐
│  Entrypoints:  train.py                                        │
├───────────────────────────────────────────────────────────────┤
│  Orchestration:  Controller (training loop) │ Service │ Registry│
├───────────────────────────────────────────────────────────────┤
│  Components:  Actor │ Rollout │ Critic │ ActorFwd │ GenRM      │
├───────────────────────────────────────────────────────────────┤
│  Engine:  SGLang inference │ reward library │ routing │ filters │
├───────────────────────────────────────────────────────────────┤
│  Backends:  Megatron-LM (training) │ SGLang (inference)         │
├───────────────────────────────────────────────────────────────┤
│  Distributed:  Ray Actor Groups │ DCS (weight synchronization)  │
└───────────────────────────────────────────────────────────────┘
```

The training backend is Megatron-LM, supporting the full set of parallelism strategies introduced in B.1 (TP/PP/CP/EP). The inference backend is SGLang. A Megatron Bridge handles weight-format conversion between the two.

### TransferQueue: A Streaming Data Channel

Recall the asynchronous mechanism in B.1: rollout writes data into a buffer; training reads from the buffer to update parameters. Traditional buffers are batch-based: rollout writes after generating an entire batch; training reads only when the batch is available. This creates waiting: if rollout writes too fast the buffer overflows; if training reads too fast the buffer runs empty and GPUs idle. In Agentic RL, this is compounded by tool-execution waits, so it is better to support finer-grained streaming.

TransferQueue makes the pipeline streaming: rollout pushes each sample as soon as it is produced, and the training side begins processing as soon as it receives a sample, without waiting for a full batch. In Agentic RL, a "sample" may be a multi-turn episode including tool results, not just a completion. Weight synchronization is handled by DCS (Distributed Checkpoint Service): after each training step, DCS broadcasts weights (via NCCL) to rollout and other components, overlapping with subsequent computation so it does not add extra wall time.

This design shrinks waiting from batch-level to sample-level, reducing idle time by roughly an order of magnitude.

### Two Execution Modes

Relax provides two modes to match different hardware constraints.

In **Collocate mode**, Actor and Rollout share the same GPU group and alternate. After Rollout finishes a batch, it yields GPUs to Training. This is suitable when GPU capacity is limited, and it can achieve strict on-policy behavior: there is no parameter lag, and training always consumes data generated by the latest policy.

In **Fully Async mode**, each role runs on an independent GPU cluster. Data is exchanged via TransferQueue, and weights are synchronized asynchronously via DCS. The parameter `--max-staleness` controls how much "old" data is allowed to participate in training: `0` means strict on-policy, and higher values allow more asynchrony for higher throughput. This is the same underlying issue as B.1's "how to handle old data," but in Agentic RL, "staleness" can also come from environment state changes, tool-version changes, or external data changes. That makes environment snapshots and reproducibility metadata even more important.

### Engineering Details

**Loss masks.** A common implementation mistake in Agentic RL is to include _all tokens_ from multi-turn trajectories in the loss. Tool outputs are not produced by the model; the model should not be trained to "predict tool outputs." What the model needs to learn is _when to call which tool and how to interpret tool results_. Relax uses a **loss mask**: tokens generated by the model have mask=1 (included in training), while tool-return tokens have mask=0 (excluded).

**Decoupled environment interfaces.** `BaseInteractionEnv` only exposes `reset`, `step`, and `format_observation`. Environment implementations are fully separated from rollout logic, so swapping a tool environment does not require touching training code. This sounds obvious, but in real projects coupling between environment and training logic is common and painful.

**Multimodal context continuity.** In multi-turn dialogs, an image provided by the user in turn 1 must still be visible to the model at turn 3. Relax maintains `image_data` on the rollout side and `multimodal_train_inputs` on the training side, and merges them automatically each round.

**Elastic scaling.** In RL post-training, 60-70% of wall time is often spent on rollout. When rollout becomes the bottleneck, Relax can add inference engines dynamically without stopping training:

```bash
# Add engines in the current cluster
curl -X POST http://controller:8000/scale \
  -d '{"target_engine_count": 4, "mode": "ray_native"}'

# Or register engines from another cluster (federated inference)
curl -X POST http://controller:8000/scale \
  -d '{"engine_urls": ["gpu-cluster-2:8000"], "mode": "external"}'
```

The `external` mode is worth highlighting: it can use idle capacity or preemptible instances from other GPU clusters to accelerate rollout, without migrating those resources into the current cluster.

### Algorithms, Models, and Operations

**Algorithm support.** Relax includes four algorithms: GRPO (see [Sections 8.1-8.2](/chapter09_grpo_rlvr/grpo-practice-and-mechanism)), GSPO, SAPO, and OPD (see [Section 8.5](/chapter09_grpo_rlvr/on-policy-distillation)). Adding a new algorithm requires implementing a Service class and registering it in the `ALGOS` dictionary.

**Model support.** The Qwen3 family (4B, 30B-A3B MoE), Qwen3-VL (vision-language), Qwen3-Omni (omni-modal), and Qwen3.5.

**Operational system.** HealthManager monitors heartbeats and provides two-level auto-recovery (restart in place, then global restart). Metrics Service ships training metrics to TensorBoard / WandB / ClearML. Apprise pushes alerts to Slack, email, and other channels. The real challenge in large-scale RL training is not launching a job, but keeping it running for days or weeks. GPU failures, network jitter, and OOMs are normal. Without auto-recovery, operators must intervene frequently, which severely reduces throughput.

### Comparison with Other Frameworks

| Framework | Organization       | Key Features                                        | Multimodal | Async                 |
| --------- | ------------------ | --------------------------------------------------- | ---------- | --------------------- |
| AReaL     | Tsinghua & Ant     | fully async, 2.77× speedup                          | no         | fully async           |
| Seer      | Moonshot AI (Kimi) | extreme synchronous; +74-97% rollout throughput     | no         | synchronous           |
| Agent-R1  | USTC               | MDP extension; separates process vs outcome rewards | no         | partially async       |
| NeMo Gym  | NVIDIA             | scientific agent environments                       | no         | mostly synchronous    |
| slime     | Tsinghua / Zhipu   | Megatron + SGLang; MoE-native optimizations         | no         | supports async        |
| Relax     | Xiaohongshu        | TransferQueue + elastic scaling + omni-modal        | yes        | fully async streaming |

Relax is currently one of the few Agentic RL engines that supports both omni-modal training and fully async elastic scaling. Seer represents another direction: it does not move toward async, but removes rollout tail latency within a synchronous framework via online context learning (divided rollout, context-aware scheduling, adaptive grouped speculative decoding), improving throughput by 74-97% while preserving strict on-policy guarantees (see [arXiv:2511.14617](https://arxiv.org/abs/2511.14617)). slime uses SGLang as the native inference layer and Megatron as the training backend, with dedicated optimizations for MoE post-training (fp8 rollout, DeepEP communication), making it a strong choice for large-scale MoE post-training (see [THUDM/slime](https://github.com/THUDM/slime)). The Relax paper is [arXiv:2604.11554](https://arxiv.org/abs/2604.11554), and the code is [redai-infra/Relax](https://github.com/redai-infra/Relax).

## Selection Guidance

Below is a practical set of choices. In the prototyping stage, TRL plus subprocess isolation is usually enough: first verify the training workflow and reward signal correctness. If you want an out-of-the-box full pipeline (SFT → DPO/GRPO → deployment), [ms-swift](https://github.com/modelscope/ms-swift) provides an integrated ModelScope ecosystem solution that is convenient for Chinese model adaptation. For medium scale (hundreds of concurrent trajectories), veRL or OpenRLHF plus Docker sandboxes and asyncio-based concurrency can work. For large-scale Agentic training, you generally need fully async systems like Relax or AReaL. For multimodal agents, Relax is currently one of the most direct options.

A good principle is progressive architecture evolution: validate feasibility first, optimize performance second, and productionize last.

## Hands-On: nanoRLHF, A From-Scratch LLM RL Training Framework

The framework analysis above is from a user's perspective: what each component does, how data flows, and how GPUs are scheduled. To truly understand the internal structure of an RL training framework, the fastest way is to build one. [hyunwoongko/nanoRLHF](https://github.com/hyunwoongko/nanoRLHF) is exactly such a project: it implements all essential LLM RLHF components from scratch using pure PyTorch and Triton, including a training engine, an inference engine, distributed scheduling, and RL orchestration.

nanoRLHF plays a role similar to nanoGPT: it strips a production system down to its load-bearing structure. Its directory structure maps directly onto the system layers discussed in B.1:

```
nanorlhf/
├── nanotron/     # training engine (3D parallelism, grad accumulation, checkpointing)
├── nanovllm/     # inference engine (PagedAttention, KV cache, continuous batching)
├── nanoverl/     # RL orchestration (PPO trainer, reward, dataset, configs)
├── nanoray/      # distributed scheduling (process mgmt, resource allocation)
├── nanosets/     # dataset utilities
├── kernels/      # Triton kernels (fusion, optimized ops)
└── eval/         # evaluation tools
```

### Training Engine: nanotron

nanotron corresponds to the lower layer under B.1's "training/orchestration layer": it is responsible for training large models across multiple GPUs. It implements 3D parallelism (data + pipeline + tensor), gradient accumulation, mixed-precision training, and checkpoint management.

Entry point: the `nanotron/` directory. Focus on:

- how tensor parallelism splits a linear layer across GPUs (`nanotron/parallel`)
- how pipeline parallelism places layers across devices (`nanotron/pipeline`)
- how gradient accumulation and gradient synchronization are coordinated in distributed training

### Inference Engine: nanovllm

nanovllm corresponds to the "inference/rollout layer" in B.1: it generates tokens with high throughput. It implements PagedAttention (vLLM's key technique), KV cache management, and continuous batching.

Entry point: the `nanovllm/` directory. Focus on:

- how PagedAttention avoids KV cache memory waste
- how continuous batching allows variable-length requests to share a GPU efficiently
- how weights are bridged between the inference engine and the training engine

### RL Orchestration: nanoverl

nanoverl is the orchestration layer that connects the two engines, analogous to the role OpenRLHF/veRL play in B.1. It implements a PPO training loop:

rollout (via nanovllm) → reward computation → advantage estimation → PPO clipped loss → gradient updates (via nanotron)

Entry point: the `nanoverl/trainer/` directory. Focus on:

- how the PPO `fit()` loop orchestrates the actor, reference, and rollout roles
- how KL penalties are implemented (with the reference model as an anchor)
- how reward functions are integrated (for example, a math-verification scenario)

### Suggested Reading Order

Read the project bottom-up:

1. `nanotron/`: understand distributed training first (the foundation)
2. `nanovllm/`: then study high-throughput generation and rollout-side constraints
3. `nanoverl/`: finally see how RL orchestration turns them into a PPO loop
4. `nanoray/`: if you care about scheduling, study process management and resource allocation

### Hands-On Exercises

```bash
# Clone the project
git clone https://github.com/hyunwoongko/nanoRLHF.git
cd nanoRLHF

# Install deps (requires a CUDA GPU)
pip install -e .
```

Recommended exercises:

1. Run SFT training: `bash ./scripts/train_sft.sh`, and inspect loss/lr/throughput metrics.
2. Read the PPO trainer: open `nanoverl/trainer/` and draw the dataflow graph: rollout → reward → advantage → train.
3. Compare against the B.1 framework table: map nanoRLHF modules to equivalent components in OpenRLHF / veRL / slime.
4. Modify the reward function: replace logic under `nanoverl/reward/` (for example, string matching or regex extraction), and run a custom-reward RL loop end-to-end.

nanoRLHF is not meant for production use. Its value is that it turns concepts like "rollout engine," "training backend," "weight sync," and "policy version" into readable code. After reading it, the source code of veRL or OpenRLHF becomes much easier to navigate.

## References

[^relax_paper]: Zhang L, Ning B, Yang R, et al. "[Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/abs/2604.11554)." arXiv:2604.11554, 2026. [GitHub](https://github.com/redai-infra/Relax)

[^1]: HuggingFace Blog, "[Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape)", 2026.

[^2]: PyTorch Blog, "[A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/)", 2025.

[^3]: AReaL Team. "[AReaL: Async RL for Language Reasoning](https://arxiv.org/abs/2505.24298)." arXiv:2505.24298, 2025. [GitHub](https://github.com/inclusionAI/AReaL)

[^4]: Hou L et al. "[Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning](https://arxiv.org/abs/2511.14617)." arXiv:2511.14617, 2025.

[^5]: Ko H. "[nanoRLHF: From-scratch journey into how LLMs and RLHF really work](https://github.com/hyunwoongko/nanoRLHF)." GitHub, 2025.
