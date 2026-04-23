# B.4 Agentic RL 基础设施

> 前面 B.1-B.3 讨论了 LLM RL 的基础设施，其前提是模型仅在 GPU 上进行文本生成。当训练对象变为具备工具调用、代码执行和多轮交互能力的 Agent 时，基础设施的需求将发生显著变化。

## 从单轮生成到多轮行动

以第 8 章的 GRPO 训练为例：给定一道数学题，模型一次性生成完整解答，随后检查答案是否正确。整个过程在单轮内完成，模型唯一的操作是文本生成，所有计算均在 GPU 上进行。

相比之下，训练一个具备 bug 修复能力的 Agent 需要完全不同的交互模式。模型拿到一段有错误的代码后，需要先读取代码、定位问题、修改代码、运行测试验证结果。若测试未通过，还需继续修改。一个任务可能需要五六个回合，而每一步之间都存在等待——读文件依赖磁盘 IO，跑测试依赖沙箱执行，搜索依赖网络响应。这些操作不在 GPU 上运行，延迟从几十毫秒到几秒不等。

由此产生了一组环环相扣的工程问题。当 Agent 需要执行代码时，如何确保安全性？这需要**沙箱隔离**。当交互变成多轮之后，如何存储结构远比三元组复杂的训练数据？这需要**多轮轨迹存储**。当每一轮交互都让 GPU 空闲数百毫秒时，如何避免算力浪费？这需要**GPU 调度优化**。这些问题沿着训练流程依次出现，每一层的解决方案又为下一层创造了条件。下面的讨论将沿着这条链条逐一展开。

## 沙箱隔离

Agent 的核心能力之一是执行代码，这也带来了最大的安全隐患。在训练过程中，模型会尝试各种策略以获取更高分数。如果不加限制，它可能生成 `os.system("rm -rf /")` 来删除训练服务器上的文件，或读取环境变量中的 API key。这些行为并非恶意——模型只是在探索动作空间。但后果是灾难性的：一个运行中的训练任务可能被破坏，导致文件系统被清空、训练数据丢失。

因此，Agent 执行代码必须在*隔离环境*（sandbox）中运行。隔离方案的选择需要在安全性、启动开销和资源利用率之间取得平衡。

### 隔离方案对比

实践中主要有三种隔离方案，适用于不同的场景：

| 方案 | 隔离粒度 | 启动延迟 | 适用场景 |
| ---- | -------- | -------- | -------- |
| subprocess + 资源限制 | 进程级 | ~10 ms | 原型验证，信任环境 |
| Docker 容器 | 文件系统 + 网络 + 资源 | ~100 ms | 通用训练，需要完整隔离 |
| MicroVM（Firecracker） | 内核级 | ~125 ms | 安全敏感场景，需要硬件级隔离 |
| WebAssembly（Wasm） | 指令集级 | ~1 ms | 纯计算任务，追求极低延迟 |

**subprocess + 资源限制**是最轻量的方案。通过 `rlimit` 限制 CPU 时间和内存、通过 `chroot` 限制文件系统访问、通过 `unshare` 限制网络命名空间。隔离强度有限（进程仍共享宿主内核），但启动开销极低，适合早期原型验证阶段：

```python
import subprocess, resource

def run_in_subprocess(code, timeout=10, max_memory=256 * 1024 * 1024):
    """最轻量的隔离：subprocess + 资源限制"""
    def set_limits():
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))

    result = subprocess.run(
        ["python", "-c", code],
        timeout=timeout,
        preexec_fn=set_limits,
        capture_output=True, text=True,
    )
    return result
```

**Docker 容器**是工业级训练中最常用的方案。通过 Linux cgroups 和 namespace 提供文件系统、网络和资源的三重隔离，且拥有完整的镜像生态（可直接使用 `python:3.11-slim`、`node:20` 等基础镜像）。主要开销在于容器启动（约 100 毫秒），可通过预热容器池优化：

```python
container = client.containers.run(
    "python:3.11-slim",
    command=f"python -c '{code}'",
    detach=True,
    mem_limit="512m",       # 内存限制
    cpu_quota=50000,         # CPU 限制
    network_mode="none",     # 禁止联网
    remove=True,
)
```

**MicroVM（如 Firecracker）** 提供内核级隔离——每个 VM 运行独立的精简 Linux 内核，即使一个 VM 被攻破也无法影响宿主机或其他 VM。AWS Lambda 和 Fly.io 的沙箱即基于此技术。启动延迟约 125 毫秒，与 Docker 相当，但安全性显著更高。适合训练中存在不可信代码执行的场景。

**WebAssembly（Wasm）** 通过 WASI（WebAssembly System Interface）提供一种指令集级别的沙箱。代码被编译为 Wasm 字节码后，只能调用宿主显式导出的函数，无法访问文件系统或网络。启动延迟仅约 1 毫秒，但生态仍在发展中，不支持所有 Python 包。

### 网络策略

无论选择哪种方案，网络访问都需要严格控制。训练中的 Agent 不应直接访问外网——这既不安全，也不可复现（同一个 Agent 第二次运行相同任务时，搜索引擎返回的结果可能已经变化，导致训练轨迹无法重现）。对于确实需要联网的场景（如 Web Agent），应通过代理做请求过滤和缓存，而非简单地断网。

### 预热容器池

当采用 Docker 方案时，容器的启动开销不容忽视。如果同时运行 1000 个 episode，每个都创建新容器，仅启动就需约一分钟。工业级的做法是维护一个*预热容器池*（warm container pool）——提前创建好 N 个容器，用完回收重置而非销毁，启动开销可降至约 5 毫秒。

沙箱解决了"Agent 能安全地执行动作"的问题。接下来，Agent 的多轮交互会产生大量结构化的训练数据，这些数据需要被妥善存储和管理。

## 多轮轨迹存储

### LLM RL vs. Agentic RL 的数据结构差异

LLM RL 的训练数据很简单：每个样本就是一个三元组 `(prompt, completion, reward)`——题目、回答、分数，一行 JSON 即可完成。

Agentic RL 的训练数据则更像一棵对话树。一个 episode 可能包含七八轮对话，每轮包含模型输出和工具返回。以"修复 Python bug"任务为例：模型先读代码，然后修改，跑测试发现失败，继续修改，再跑测试通过——这些交互过程都需要完整记录。

### 存储需求

与 LLM RL 的扁平三元组相比，Agentic RL 的存储系统还需要支持三项额外能力：

- **按任务类型检索**：如分析"数学做得好但代码做得差"的模式
- **按步骤切片**：定位具体哪一步决策出错
- **去重和过期处理**：同一任务不重复训练，旧轨迹可能因环境变化而失效

规模较小时（不到一万条轨迹），JSON 文件加 SQLite 即可满足需求。中等规模（一万到一百万条）可以使用 Redis 做索引、S3 存储数据。超过一百万条则需要分布式数据库（MongoDB 或 DynamoDB）。对于多模态 Agent，轨迹中还包含图片和音频——此时应存储引用（URL）而非原始数据，训练时按需下载，保持轨迹索引在 KB 级别。

存储问题解决后，训练过程中的下一个瓶颈出现了：多轮交互中大量的等待时间导致 GPU 严重空等。

## GPU 空等与异步调度

### 问题量化

B.2 节讨论过 LLM RL 的 GPU 空等问题：生成和训练串行执行，训练 GPU 有 99% 的时间在等待生成完成。Agentic RL 将这一问题进一步加剧。

以单条 Agentic 轨迹的时间线为例：GPU 生成一个动作约需 3 毫秒，随后 CPU 执行工具约需 500 毫秒。在这 500 毫秒内 GPU 处于空闲状态。下一轮类似：GPU 3 毫秒，CPU 300 毫秒。几轮交互后，GPU 实际工作时间不到 1%。与 LLM RL 相比，LLM RL 的空等发生在 rollout 和 training 之间，而 Agentic RL 的空等发生在每一轮交互内部。

### 批次内并发：流水线调度

解法与 B.2 一脉相承：并发运行多条轨迹。轨迹 A 等待工具返回时，GPU 为轨迹 B 生成动作；轨迹 B 等待工具时，GPU 为轨迹 C 生成动作。通过流水线调度，GPU 持续保持工作状态。这种设计可将 GPU 利用率从约 1% 提升至 70-80%，吞吐量提升 50-100 倍。

### 两级异步

上述方案解决的是"批次内"的并发问题。批次之间仍存在 B.2 节讨论的 Rollout 和 Training 串行问题。完整方案采用两级异步：批次内多条轨迹并发（GPU 和工具交替工作），批次间 Rollout 和 Training 通过数据队列解耦（Rollout 持续生成，Training 持续训练）。

至此，Agentic RL 的三个基础工程问题——安全执行、数据存储、GPU 调度——已逐一讨论。这些解决方案在真实的工业系统中如何组织？下面的 Relax 案例提供了一个完整的参考实现。

## 工业级实现：Relax

Relax 是小红书 AI Infra 团队开源的多模态 Agentic RL 后训练框架，也是目前少数支持全模态（文本、图像、音频）Agentic RL 训练的引擎之一。以下从架构、数据流、执行模式和工程细节四个层面进行分析。

### 分离式架构

Relax 的核心设计选择是将训练流程中的每个角色部署为独立的 Ray Serve 服务：Actor、Rollout、Critic、Reference、Advantages、GenRM 各自独立运行。这一设计源于 Agentic RL 组件的异构性——推理需要 GPU、工具执行需要 CPU、编排需要 CPU 和内存。独立部署使每个组件可以按需扩缩、独立容错，避免资源争用。

```
┌───────────────────────────────────────────────────────────────┐
│  Entrypoints:  train.py                                        │
├───────────────────────────────────────────────────────────────┤
│  Orchestration:  Controller (训练循环) │ Service │ Registry    │
├───────────────────────────────────────────────────────────────┤
│  Components:  Actor │ Rollout │ Critic │ ActorFwd │ GenRM     │
├───────────────────────────────────────────────────────────────┤
│  Engine:  SGLang 推理 │ 奖励函数库 │ 路由 │ 过滤器             │
├───────────────────────────────────────────────────────────────┤
│  Backends:  Megatron-LM (训练) │ SGLang (推理)                 │
├───────────────────────────────────────────────────────────────┤
│  Distributed:  Ray Actor Groups │ DCS (权重同步)               │
└───────────────────────────────────────────────────────────────┘
```

训练后端采用 Megatron-LM，支持 B.3 节介绍过的 TP/PP/CP/EP 全套并行策略。推理后端采用 SGLang，两者之间通过 Megatron Bridge 自动完成权重格式转换。

### TransferQueue：流式数据通道

回顾 B.2 节的异步训练机制：Rollout 生成数据写入 Buffer，Training 从 Buffer 读取数据训练。传统 Buffer 是批量的——Rollout 生成完整个 batch 才写入，Training 等有数据后才读取。这导致一侧始终在等待：Rollout 写入过快时 Buffer 溢出，Training 读取过快时 Buffer 为空导致 GPU 空闲。

TransferQueue 将这一交互改为流式：Rollout 每生成一个样本即写入队列，Training 端每拿到一个样本即开始处理，无需等待整个 batch 生成完毕。配合 DCS（Distributed Checkpoint Service）做权重同步——Training 每更新一步参数，DCS 通过 NCCL 广播给 Rollout 等组件，与下一次训练计算重叠进行，不占额外时间。

这一设计将异步训练中 Batch 级别的等待缩短为 Sample 级别的等待，等待时间降低了一个数量级。

### 两种执行模式

Relax 提供两种模式以适应不同的硬件条件。

*Collocate 模式*下，Actor 和 Rollout 共享同一组 GPU，轮替使用。Rollout 生成完一个 batch，让出 GPU 给 Training。这适合 GPU 数量有限的情况，而且可以做到严格的 on-policy——模型参数没有任何延迟，Training 永远在用最新版本的模型生成的数据。

*Fully Async 模式*下，各角色跑在独立的 GPU 集群上，通过 TransferQueue 交换数据，通过 DCS 异步同步权重。参数 `--max-staleness` 控制允许多"旧"的数据参与训练——设 0 即为严格 on-policy，设大则允许更多异步以换取吞吐。这和 B.2 讨论的"旧数据怎么处理"是同一个问题，Relax 将其暴露为一个可调参数。

### 工程细节

**Loss mask。** 训练 Agentic RL 时有一个常见的实现误区：将多轮轨迹中的所有 token 都纳入 loss 计算。实际上，工具返回的结果并非模型生成，模型不应为此负责。模型需要学习的是"何时调用何种工具、如何理解工具结果"，而非"如何输出工具结果"。Relax 通过 *loss mask* 处理这一问题：模型生成的 token 标记为 mask=1 参与训练，工具返回的 token 标记为 mask=0 不参与训练。

**环境接口解耦。** `BaseInteractionEnv` 仅提供 `reset` / `step` / `format_observation` 三个方法，环境实现与 Rollout 逻辑完全分离。更换工具环境无需修改训练代码。虽然这一设计看似理所当然，但在实际项目中，环境与训练逻辑的耦合是非常常见的问题。

**多模态上下文保持。** 多轮对话里，第一轮用户发的图片，到第三轮模型仍需看到。Relax 在 Rollout 端维护 `image_data`，在 Training 端维护 `multimodal_train_inputs`，每轮自动合并。

**弹性扩展。** RL 训练的 60-70% 时间花在 Rollout 上。若训练过程中发现 Rollout 速度成为瓶颈，Relax 支持在不中断训练的情况下动态增加推理引擎：

```bash
# 在当前集群里加引擎
curl -X POST http://controller:8000/scale \
  -d '{"target_engine_count": 4, "mode": "ray_native"}'

# 或者注册其他集群已有的引擎（跨集群联邦推理）
curl -X POST http://controller:8000/scale \
  -d '{"engine_urls": ["gpu-cluster-2:8000"], "mode": "external"}'
```

`external` 模式值得注意——它可以利用其他 GPU 集群上的空闲资源或抢占式实例来加速 Rollout，无需将它们迁移到当前集群。

### 算法、模型和运维

**算法支持。** Relax 内置了四种算法：GRPO（见 [8.1-8.2 节](/chapter08_grpo_rlvr/grpo-practice-and-mechanism)）、GSPO、SAPO 和 OPD（见 [8.5 节](/chapter08_grpo_rlvr/on-policy-distillation)）。添加新算法只需实现一个 Service 类并注册到 `ALGOS` 字典。

**模型支持。** Qwen3 全系列（4B、30B-A3B MoE）、Qwen3-VL（视觉语言）、Qwen3-Omni（全模态）和 Qwen3.5。

**运维体系。** HealthManager 负责心跳监控和两级自动恢复（先尝试原地重启，失败后全局重启）；Metrics Service 将训练指标分发到 TensorBoard / WandB / ClearML；Apprise 负责推送告警到 Slack、微信、邮件。大规模 RL 训练的挑战不在于启动，而在于持续稳定运行——一个训练任务可能持续数天甚至数周，期间 GPU 故障、网络抖动、OOM 都是常态。缺乏自动恢复机制将导致运维人员频繁手动介入，严重影响训练效率。

### 与其他框架对比

| 框架     | 出品方      | 特点                             | 多模态 | 异步       |
| -------- | ----------- | -------------------------------- | ------ | ---------- |
| AReaL    | 清华 & 蚂蚁 | 全异步，2.77x 提速               | 否     | 全异步     |
| Agent-R1 | 中科大      | MDP 扩展，过程/结果奖励分离      | 否     | 部分异步   |
| NeMo Gym | NVIDIA      | 科学 Agent 环境                  | 否     | 同步为主   |
| Relax    | 小红书      | TransferQueue + 弹性扩展 + 全模态 | 是     | 全异步流式 |

Relax 是目前唯一同时支持全模态和全异步弹性扩展的 Agentic RL 引擎。论文见 [arxiv.org/abs/2604.11554](https://arxiv.org/abs/2604.11554)，代码见 [github.com/redai-infra/Relax](https://github.com/redai-infra/Relax)。

## 选型建议

以下是实践层面的选型建议。原型验证阶段，TRL 配合 subprocess 即可满足需求——先验证训练流程的可行性和 reward 信号的正确性。中等规模（如数百条轨迹并发）可以采用 veRL 或 OpenRLHF，配合 Docker 沙箱和 asyncio 实现异步并发。大规模 Agentic 训练则需要 Relax 或 AReaL 等全异步框架。多模态 Agent 场景下，Relax 是目前唯一的选择。

建议遵循渐进式架构演进原则：先验证流程可行性，再做性能优化，最后进行生产化改造。

## 参考文献

[^relax_paper]: Zhang L, Ning B, Yang R, et al. "[Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/abs/2604.11554)." arXiv:2604.11554, 2026. [GitHub](https://github.com/redai-infra/Relax)

[^1]: HuggingFace Blog, "[Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape)", 2026.

[^2]: PyTorch Blog, "[A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/)", 2025.

[^3]: AReaL Team. "[AReaL: Async RL for Language Reasoning](https://arxiv.org/abs/2505.24298)." arXiv:2505.24298, 2025. [GitHub](https://github.com/inclusionAI/AReaL)
