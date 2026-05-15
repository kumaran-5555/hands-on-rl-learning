---
search: false
---

# 旧页：工业实践（已并入 10.3）

> 这一页保留为旧链接入口。核心内容已经合并到 [10.3 工业实践、评测与 Badcase](./industrial-evaluation)。下面保留原文，方便从旧链接进入的读者对照。

# 12.5 工业界实践：Agentic RL 训练中的常见问题与解决方案

前面几节介绍了 Agentic RL 的通用工程原则和框架设计。然而，在实际训练过程中，研究者往往会遇到一系列工程问题——训练不稳定、输出长度失控、奖励指标与实际质量脱节等。这些问题在学术论文中通常不会详细讨论，但对于工程实践至关重要。

2025–2026 年间，多家团队（包括 Alibaba、Moonshot、LinkedIn、Bespoke Labs 等）陆续公开了他们在 Agentic RL 训练中的实践经验。本节不再按团队逐一介绍，而是**按照实际训练中可能遇到的问题场景**进行组织，将不同团队的发现和解决方案汇总在一起。

> **核心要点**：在 Agentic RL 中，训练的稳定性往往比算法选择更为重要。数据质量和环境的一致性是决定训练效果的关键因素。

---

## 场景一：训练数据的获取与环境构建

许多研究者在开始 Agentic RL 训练时，首先面临的问题是：如何为模型提供稳定且可复现的交互环境？

### 真实 API 的局限性：不可复现性

如果直接接入真实的搜索引擎或代码执行环境进行训练，会遇到一个根本性的问题：**外部环境的输出是不可复现的**。

> **Moonshot AI** 在训练 Kimi-Researcher 时指出，Agent 所面对的环境是动态的——即使输入相同的查询，搜索引擎也可能返回不同的结果。他们在训练中主要采用了 **REINFORCE** 算法，并强调严格 On-policy 数据生成对训练稳定性的重要性 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

### 合成环境的构建

一个可行的替代方案是构建确定性的合成环境，让模型在受控的条件下进行训练。

> **Alibaba 通义团队** (Tongyi DeepResearch) 摒弃了充满噪声且不可控的在线 API，构建了一个以离线 Wikipedia 数据库和稳定工具沙盒为核心的合成训练环境。
>
> **核心方法与具体内容**：
>
> 1. **数据与环境合成 (WebShaper & AgentFounder)**：由于真实网页经常变动，导致相同 Query 在不同时间的搜索结果不一致，这严重破坏了强化学习的马尔可夫决策过程（MDP）假设。为此，他们开发了 **WebShaper**，将海量 Wikipedia 转化为静态且结构化的离线搜索环境；同时利用 **AgentFounder** 自动生成具有极高难度（PhD-level）的合成查询和基准答案。这种合成环境的**确定性**使得模型在多次 Rollout 时的动作与奖励映射关系绝对稳定。
> 2. **异步计算架构 (rLLM)**：Agentic RL 的 Rollout 阶段（即与环境交互生成长达几十步的动作轨迹）耗时极长。如果采用传统的同步 RL 架构（即训练和推理在同一批 GPU 上交替进行），由于环境交互的延迟，训练节点（GPU）将长时间处于空闲状态（GPU Idling）。他们开发的 **rLLM (Ray-based LLM)** 异步 Rollout 服务，在物理层面隔离了推理和训练：多个 Worker 节点利用高吞吐推理引擎（如 vLLM）不断与环境交互生成 Trajectory 并存入共享的 Replay Buffer，而专门的 Trainer 节点（基于 Megatron/FSDP）则持续从 Buffer 中采样并计算梯度更新模型。
>
> **深层原因与工程意义**：
> 实验证明，在高度受控、无噪声的合成环境中进行强化学习，其最终产出的模型在真实互联网环境下的泛化能力，反而**全面优于**直接使用带噪声的人类专家标注数据进行训练。其根本原因在于：模型在 RL 阶段真正需要学习的是“如何搜索、如何根据结果反思重试”的通用决策逻辑，而不是过拟合某些特定的搜索返回结果。稳定的环境信号是 RL 算法收敛的基石 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### 小规模数据的有效性

对于资源有限的研究者而言，高质量的小规模数据同样可以取得显著效果。

> **Amazon Science** 在复杂的 AppWorld 基准测试上验证了“极少样本定制”的可行性：他们并没有盲目收集几万条带噪声的人类交互轨迹，而是精心构建了仅 **72 个高质量训练样本**（覆盖了核心的工具调用模式、依赖关系，以及遇到 API 错误时的重试逻辑）。通过 RL 训练，成功将 Qwen-2.5-32B 的任务完成率从 39.2% 提升至 72%，一举超越了当时的最强闭源模型 Claude Sonnet 3.7/4.0。
>
> **核心方法与深层原因**：
> 这一反直觉的结果揭示了现代 Agentic RL 的一个核心洞察：对于 32B 以上参数量的基座模型，它们在预训练阶段已经具备了强大的世界知识和逻辑推理能力。此时，RL 的作用并非“向模型中注入新知识”，而是“激活（Elicit）并对齐”模型在特定环境下的交互范式与工具语法。只要这区区 72 个高质量样本能够作为“引子”，成功触发模型在环境中的有效探索（Exploration），RL 算法（如 PPO/GRPO）就能通过环境反馈的奖励信号，让模型在数万次的自我试错（Self-Play）中自行完善策略。这证明了**在基础能力达标的模型上，强化学习具有极高的数据效率，"少而精的数据 + RL 自主探索" 远胜于海量低质量的 SFT 数据** [\[参考\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

---

## 场景二：训练初期的梯度爆炸问题

在解决了数据和环境的准备问题后，训练启动阶段的梯度爆炸是另一个常见问题。在排查超参数之前，应当首先检查底层实现的正确性。

### 推理引擎与训练引擎的实现差异

Agentic RL 的训练过程包含两个阶段：**推理（Rollout）** 阶段生成动作序列，**训练（Backward）** 阶段更新模型权重。这两个阶段通常由不同的引擎负责执行，而引擎之间的实现差异可能导致梯度计算不一致。

> **LinkedIn 团队** 在使用 GPT-OSS（一个 MoE 架构的开源模型）进行 RL 训练时，遇到了梯度爆炸和奖励不增长的问题。经过排查，他们发现根本原因是训练框架中 **Attention Sink 参数的反向传播未被实现**：推理引擎（SGLang 使用的 Triton kernel）支持 Attention Sink 的前向计算，但训练框架（FSDP 使用的 FlashAttention-v2）完全缺少对应的支持。他们从 vLLM 的 FlashAttention 分支中获取了前向实现，并自行编写了反向传播代码来计算 Sink 参数的梯度。修复该问题后，训练才恢复稳定 [\[参考\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

**实践建议**：在使用复杂模型架构时，建议先在简单的单轮任务（如 GSM8K）上验证训练流程的正确性，确认 Loss 正常下降后，再切换到多轮 Agent 任务。

---

## 场景三：输出长度失控与格式坍塌

这是 Agentic RL 训练中最常见的问题之一：模型未能学会正确使用工具，反而开始生成大量无意义的 token，最终退化为重复的乱码输出。这种现象被称为**格式坍塌（Format Collapse）**：

```json
// 期望的输出格式：
{"action": "search", "query": "AAPL stock"}

// 格式坍塌后的输出：
{"action": "searchsearchAAPL stockAAAAA"
```

下面分析导致这一问题的三个主要原因及其对应的解决方案。

### 原因一：奖励函数设计过于复杂

直觉上，研究者可能会设计多维度的奖励信号：工具调用成功给 +1，输出格式正确给 +1，最终答案正确给 +5。然而，这种细粒度的奖励设计可能适得其反。

**奖励博弈（Reward Hacking）** 是其中的核心问题。当奖励函数包含多个可被模型独立优化的子项时，模型可能找到只满足部分条件就能获得高奖励的策略。

> **Bespoke Labs** 的实验表明，包含工具调用次数奖励、格式检查奖励和正确性奖励的复合奖励函数，反而导致训练稳定性下降，推测原因正是奖励博弈。此外，他们还观察到输出长度持续膨胀、最终退化为无意义的乱码字符。他们最终采用的做法是：**仅保留"任务是否完成"这一个二值奖励信号**（通过 BFCL 的评估检查即为 1，否则为 0），删除所有中间过程的奖励项，训练稳定性反而显著提升 [\[参考\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

这一发现背后的逻辑是：二值的最终结果奖励不提供任何中间步骤的"捷径"，模型必须在整体上完成任务才能获得正向奖励，从而避免了针对单个奖励项的投机行为。此外，Bespoke Labs 还观察到复合奖励下输出长度持续膨胀并最终退化为乱码的现象，简化奖励设计后这一问题也得到了缓解。

### 原因二：负样本处理不当

在训练过程中，并非所有未能完成任务的样本质量都相同。例如，模型可能因为交互步数达到上限而被环境截断，此时并未产生最终答案，但在此之前的输出可能是合理的。如果将这类样本不加区分地作为负样本给予惩罚，可能会损害模型已经习得的输出能力。

> **Alibaba 通义团队** 观察到，如果不加过滤地将所有未完成任务的轨迹视为负样本进行惩罚，在长时间训练后会导致严重的**格式坍塌**——模型为了规避因任务失败带来的整体惩罚，开始产生乱码，或者完全拒绝使用工具（因为多做多错）。
>
> **核心方法与深层原因**：
> 为解决这一长程信用分配（Credit Assignment）难题，他们在定制的 **On-policy GRPO（Group Relative Policy Optimization）算法**中，采取了以下两项核心设计：
>
> 1. **Token-level loss 与 Leave-one-out 优势估计**：相比于传统 PPO 将整个轨迹的奖励均摊到每一个动作上，GRPO 通过组内生成多个候选轨迹，计算每个动作相对于组内其他动作的相对优势（Relative Advantage），并在 Token 级别施加更精细的梯度更新，这大幅降低了奖励评估的方差。
> 2. **保守的负样本过滤策略（Conservative Negative Filtering）**：Agent 的动作具有极强的因果序列性。在长达 30 步的交互中，很多轨迹最终失败（例如超时或达到最大交互步数截断），往往只是因为最后几步的逻辑判断失误，而其前 20 步的思维链（CoT）和工具调用格式是完全正确的。如果对这类截断样本强行给予全局负奖励（如 `-1`），RL 优化器就会“倒洗澡水连孩子一起倒掉”，错误地惩罚了原本正确的格式输出。因此，他们**选择性地将这类截断样本从损失计算中剔除（Mask out）**，使得它们不贡献负向梯度。这一策略极为有效地保护了模型的基础对齐（Alignment）能力，维持了格式输出的长期稳定性 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### 原因三：KL 散度约束的配置不当

在 RLHF/GRPO 中，通常使用 KL 惩罚项来限制当前策略模型与初始参考模型之间的偏离程度。KL 约束的作用是防止策略在训练过程中偏离初始模型太远，从而维持输出的基本质量。

这一约束的配置需要在"允许策略探索"和"维持稳定性"之间取得平衡：

- **KL 惩罚过小**：约束力不足，策略可能偏离初始模型太远，导致输出质量退化。
- **KL 惩罚过大**：约束过强，策略难以学到新的行为，训练效果受限。

> **Bespoke Labs** 在训练 Qwen2.5-7B-Instruct 时发现，将 KL 惩罚设为 0 时，模型在约 300 步后即出现输出退化。他们采用的策略是：
>
> 1. **设置微小的 KL 权重**（如 0.001），提供最小程度的约束。
> 2. **定期更新参考模型**：每隔一定步数（如 100 步），将当前策略模型复制为新的参考模型。这样，KL 约束的目标会随训练推进而动态调整，避免策略被"锚定"在过远的初始状态上 [\[参考\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

### 输出长度控制：Gamma-decay 奖励

为了鼓励模型以更少的步数完成任务，可以引入基于步数衰减的奖励机制。

> **Moonshot** 提出了 **Gamma-decay Reward**。当模型正确完成任务时，奖励值随所用步数指数衰减：
>
> $$r_i = r \times \gamma^{T-i}, \quad \gamma < 1$$
>
> 其中 $T$ 是总步数，$i$ 是当前步数。这意味着：完成相同任务时，使用更少的步数会获得更高的奖励，从而引导模型学会更高效地执行任务 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## 场景四：长程交互中的上下文管理

Agentic RL 与传统 RL 的一个重要区别在于交互轮数可能非常长。在文献检索、代码编写、调试等复杂任务中，交互轮数可能超过 50 轮，此时上下文窗口会被大量历史信息填满，模型可能丢失对初始任务的关注。

### 上下文管理机制

> **Moonshot** 的 Kimi-Researcher 引入了 **上下文管理（Context Management）** 机制，这是解决长程任务（Long-horizon tasks）中注意力稀释（Attention Dilution）和“迷失在中间（Lost in the Middle）”问题的关键工程实践。
>
> **核心方法与深层原因**：
> 在长达几十轮的 Agent 交互中，如果不加控制，网页的冗余 HTML 标签、成百上千行的代码执行日志等 Observation 文本，会迅速填满模型十几万 Token 的上下文窗口。随着上下文长度的急剧增加，LLM 的信噪比（SNR）会显著下降，导致模型在第 40 轮时“忘记”了用户在第 1 轮提出的原始需求。
>
> 为此，Kimi 引入了一个独立的 `context_manager` 机制。在每执行完一步（Step）后，系统会动态评估并**压缩上下文**：
>
> 1. **保留核心逻辑（Working Memory）**：将模型自身的思维链（Thought）、历史动作（Action）以及从网页中提取的关键事实保留在上下文的核心区。
> 2. **摘要或丢弃噪音**：将长篇累牍的原始网页替换为一两句话的摘要，或者直接丢弃那些已经被证明是死胡同（Dead-end）的无效搜索记录。
>
> 上下文管理本质上是在维护 Agent 的动态“工作记忆”，确保模型每一步决策的输入都是高密度的有效信息。消融实验显示，启用该机制后，不仅避免了灾难性的遗忘，还将单次 Rollout 的安全交互轮次延伸至 50 轮以上，模型能够获取更多线索，最终在复杂研究任务上取得显著更高的得分 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## 场景五：Agent 幻觉及其控制

在解决了训练稳定性和输出格式的问题之后，另一个需要关注的问题是 **Agent 幻觉（Hallucination）**：模型可能在搜索结果中引用不存在的文献，或者错误地使用 API 参数，却对后续推理表现出不恰当的"自信"。Agent 场景中的幻觉比纯对话场景更为复杂，因为模型不仅生成文本，还生成动作。

### Agent 幻觉的四种类型

**工具选择幻觉。** 模型调用了一个不存在的工具，或者在不该调用工具时强制调用。例如用户询问天气信息，模型却调用了 `execute_sql`。

**参数幻觉。** 工具选择正确，但参数填写错误——编造了不存在的 API 端点、拼错了数据库名、或使用了格式不正确的参数值。最值得警惕的是：参数格式可能看起来"合理"，但实际值是虚构的。

**结果幻觉。** 这是最隐蔽的幻觉类型。模型调用了正确的工具并获得了真实的返回结果，但在解读结果时引入了偏差——将搜索结果中的无关信息当作支持自己论点的证据，或忽略了与假设矛盾的内容。

**引用幻觉。** 模型声称"根据某文献/某网站"得出某个结论，但该引用实际上不存在，或引用内容与原文不符。这在 Deep Research Agent 中尤为常见——模型可能编造论文标题、URL 和统计数据来使输出"看起来有据可查"。

### Agent 幻觉的级联效应

在纯对话场景中，幻觉的后果通常限于提供错误信息。但在 Agent 场景中，幻觉会在多轮交互中**级联传播并自我强化**：

1. 第 3 轮：模型产生参数幻觉，调用了一个不存在的 API 参数 → 调用失败
2. 第 4 轮：模型未能识别幻觉，反而认为"该 API 存在缺陷" → 切换到另一个工具
3. 第 5 轮：新工具缺少关键功能 → 模型编造了一个看似合理的结论
4. 最终输出：一份表面上完整但建立在幻觉基础上的报告

更值得关注的是，如果 RL 的奖励仅基于最终输出质量（即 Outcome Reward），理论上模型可能发现"编造一个看似可信的答案"比"承认不确定"获得更高的奖励——这意味着 RL 训练反而可能**强化幻觉行为**。这一推断在逻辑上成立，但在已公开的工业界实践中尚未被明确报告为观察到的现象。

### RL 训练中的幻觉惩罚机制

**引用感知评分奖励。** 清华大学与智谱 AI 联合提出的 CaRR[^carr_industrial]（Citation-aware Rubric Rewards）设计了一种细粒度的奖励机制来引导模型正确引用证据。其核心思路是将多跳问题分解为一系列原子事实陈述（Rubrics），然后通过三步流程计算奖励：（1）检查模型输出是否识别了关键实体；（2）提取输出中引用的 URL，获取网页内容，判断每条 Rubric 是否被引用内容所支持；（3）通过图上的广度优先搜索验证各 Rubric 是否在逻辑上与最终答案相连通。最终奖励为被满足且逻辑连通的 Rubric 数量占总 Rubric 数量的比率。这一机制鼓励模型为每个论断提供可验证的、逻辑连贯的引用证据。

**工具结果忠实度奖励。** 鼓励模型在解读工具返回结果时忠实于原始内容。如果模型的总结与工具实际返回的信息存在偏差（通过 NLI 模型或交叉验证检测），则给予惩罚。

**不确定性奖励。** 鼓励模型在不确定时主动表达"需要更多信息"或"该结果不确定"，而非编造答案。综合上述三种策略，可以设计一个幻觉感知的奖励函数作为示例：

> **注意**：以下代码为说明性示例，综合了多种惩罚思路，并非直接来自某一篇论文的具体实现。

```python
def hallucination_aware_reward(answer, tool_results, citations):
    """幻觉感知的奖励函数"""
    reward = base_task_reward(answer)

    # 1. 引用真实性检查
    for citation in citations:
        if not verify_citation_exists(citation):
            reward -= 0.5  # 虚假引用，惩罚
        elif not verify_citation_supports(citation, answer):
            reward -= 0.3  # 引用与论断不符

    # 2. 工具结果忠实度
    for claim in extract_claims(answer):
        if has_supporting_evidence(claim, tool_results):
            reward += 0.1  # 有据可查的论断
        elif claim_is_verifiable(claim) and not has_supporting_evidence(claim, tool_results):
            reward -= 0.2  # 可验证但无证据的论断

    # 3. 鼓励不确定性表达（诚实奖励）
    if is_complex_question and ("不确定" in answer or "需要更多信息" in answer):
        if not all_claims_supported(answer, tool_results):
            reward += 0.15  # 在确实缺乏证据时，承认不确定性是合理行为

    return reward
```

### 基于验证的幻觉过滤

除了在奖励函数中惩罚幻觉外，还可以在**推理阶段**通过验证机制进行过滤：

**Self-RAG[^selfrag_industrial]** 提出了"自适应检索 + 自我评估"的框架。与传统 RAG 对每个查询都检索不同，Self-RAG 让模型在生成每个文本段**之前**，先通过特殊的反思 token（Reflection Token）判断是否需要检索外部信息。如果需要，则检索若干相关段落，为每条段落分别生成续写，并通过 [IsRel]（相关性）、[IsSup]（支撑度）、[IsUse]（有用性）等反思 token 对各候选续写进行打分，最终通过分段束搜索（Beam Search）选择综合得分最高的输出。该框架的核心特点是模型通过反思 token 实现了对自身输出的结构化自评估。

**CRITIC[^critic_industrial]** 提出了"工具辅助纠错"的幻觉过滤机制。模型生成初始回答后，主动调用外部工具（如搜索引擎、代码执行器）来验证关键论断，并基于工具反馈生成结构化的批评意见。如果批评意见表明回答存在问题，则模型基于批评意见重新生成修正后的回答。这一"验证→修正→验证"的循环可以迭代多轮，直到回答通过验证或达到最大迭代次数。与纯粹依赖模型自我评估的方法不同，CRITIC 引入了外部工具的客观反馈作为纠错依据。

### 幻觉控制实践总结

| 幻觉类型     | 检测方法                | RL 惩罚策略                 |
| ------------ | ----------------------- | --------------------------- |
| 工具选择幻觉 | 工具白名单校验          | 调用不存在工具 → reward = 0 |
| 参数幻觉     | Schema 校验 + 类型检查  | 参数格式错误 → 负向 reward  |
| 结果幻觉     | NLI 模型 + 交叉验证     | 论断与工具结果矛盾 → 惩罚   |
| 引用幻觉     | URL 可达性 + 内容相关性 | 虚假引用 → 惩罚             |

一个重要的实践原则是：**幻觉惩罚应在训练早期即引入**。一旦幻觉行为通过 RL 被强化，后续消除将非常困难。

---

## 场景六：特定模型架构的注意事项

前面的场景是大多数 Agentic RL 训练中都会遇到的共性问题。此外，使用特定的模型架构（如 MoE）或在较小参数量的模型上进行训练时，还可能出现一些额外的问题。

### MoE 模型的路由不确定性

MoE 模型（如 Mixtral、DeepSeek-V3）因推理成本较低而受到关注，但其路由机制可能破坏 RL 训练的基本假设。

PPO 等算法假设当前生成数据的模型与正在被训练的模型是同一个（即 On-policy），这在数学上表现为重要性采样比率等于 1。

> **LinkedIn 团队** 在使用 GPT-OSS 进行 RL 训练时发现，MoE 模型的路由网络（Gating Network）在两次前向传播中，可能为同一个 Token 选择不同的专家（Expert），导致 $\log \pi(a|s) \neq \log \pi_{\text{old}}(a|s)$，即 On-policy 假设被破坏。在排查过程中，他们曾尝试通过 `old_log_prob = log_prob.detach()` 的方式将两次概率强制对齐来验证这一假设。需要指出的是，该路由不一致问题虽真实存在，但在他们的调试中并非梯度爆炸的根本原因——根本原因在于上节所述的 Attention Sink 反向传播缺失 [\[参考\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

### MoE 模型的负载均衡问题

MoE 模型在 RL 训练中不仅面临上述路由一致性问题，还存在专家负载不均衡导致的 GPU 利用率低下。不同 Token 可能集中选择少数"热门"专家，导致负责这些专家的 GPU 成为瓶颈，而其他 GPU 则处于空闲状态。

> **Salesforce** 在其 SFR-RL 系统中提出了 **流水线同步 RL（Pipelined Synchronous）** 方案：所有 GPU 在 Rollout 和 Training 两个阶段之间交替切换，而非将 GPU 固定分配给某一阶段。此外，针对 MoE 模型，他们引入了 **Least-Loaded Expert Parallelism** 来优化专家的负载均衡。整体系统在内存效率上相比 VERL（FSDP + Context Parallelism）提升了约 250 倍，仅用 16 块 H200 即可训练 120B 参数的 MoE 模型 [\[参考\]](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)。

### 小模型的推理能力上限

需要注意的是，RL 的本质是**激发模型已有的能力**，而非注入新的知识。模型的基础能力决定了 RL 能够达到的效果上限。

> **Amazon Science** 的实验显示：32B 参数量的模型从 RL 中获益显著，因为模型本身能够生成高质量的交互轨迹（Rollout），形成正反馈循环。但较小的模型面临基础推理能力的限制，例如无法识别不可回答的问题或从相关上下文中提取答案——这种能力的缺失，RL 训练难以弥补。对于基础能力不足的小模型，研究者的建议是通过蒸馏（Distillation）从更强的模型中获取能力，而非单纯增加 RL 训练强度 [\[参考\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

### 分阶段训练管线

考虑到不同规模模型的特点，一个更稳健的训练策略是采用分阶段管线，而非直接进行 RL 训练。

目前工业界对于是否需要 SFT 存在两种并行的训练范式：**SFT-RL 范式**与**Pure-RL 范式**。

> **SFT-RL 范式（主流路径）**：**Alibaba 通义团队**在 Tongyi DeepResearch 中设计了 **CPT → SFT → RL** 的三阶段训练管线。在预训练（CPT）阶段将工具调用的轨迹以文本形式融入；在 SFT 阶段利用人类或高质量合成数据培养模型的基本推理和工具使用能力；最后在 RL 阶段进行探索与优化。这种范式的核心在于：在非推理对齐（如复杂 API 调用、长程探索）场景下，SFT/RM 仍是**降低探索空间和克服冷启动难度**的最有效手段。如果模型在起步阶段不具备基本的工具使用格式，直接进行 RL 训练往往会在庞大的动作空间中迷失 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

> **Pure-RL 范式（前沿突破）**：与上述思路形成鲜明对比的是 **DeepSeek-R1-Zero** 带来的范式革命。它证明了在具备明确对错反馈（如数学、代码测试、客观推理）的场景下，**完全放弃 SFT 冷启动**，直接基于 Base Model 进行大规模强化学习是绝对可行的。在纯 RL 驱动下，模型能够自发涌现出长思维链（CoT）、自我验证、甚至自我反思等高级推理能力。这种无 SFT 偏见（Bias-free）的训练方式突破了人类标注数据的上限，但对奖励信号的客观性和环境的防作弊能力提出了极高要求。

这两种范式在 Agentic RL 中并不互斥，研究者应根据**环境是否具备完全确定性的客观奖励**来选择合适的管线。

---

## 实践总结 {#tricks}

下表汇总了各问题的对应解决方案：

| 问题                   | 解决方案                                                             | 参考             |
| ---------------------- | -------------------------------------------------------------------- | ---------------- |
| 训练环境不可复现       | 构建确定性的合成环境                                                 | Alibaba          |
| 小规模数据定制         | 高质量的小数据（如 72 条）结合 RL 也能取得显著效果                   | Amazon           |
| 训练初期梯度爆炸       | 检查推理引擎与训练引擎的底层实现一致性（如 Attention Sink 反向传播） | LinkedIn         |
| 输出退化为重复乱码     | 采用极简奖励设计（仅奖惩任务成败）；对过长的输出进行过滤             | Bespoke Labs     |
| 策略偏离初始模型       | 设置较小的 KL 惩罚（如 0.001）；定期将当前模型设为新的参考模型       | Bespoke Labs     |
| 输出效率低（步数过多） | 使用 Gamma-decay 衰减奖励，鼓励以更少步数完成任务                    | Moonshot         |
| 格式坍塌               | 采用保守的负样本处理策略，排除因超长截断而未产生最终答案的轨迹       | Alibaba          |
| 长任务上下文溢出       | 引入上下文管理机制，主动摘要或丢弃无用历史信息                       | Moonshot         |
| MoE 训练资源利用率低   | 流水线同步 RL + Expert Parallelism；16 块 H200 即可训练 120B MoE     | Salesforce       |
| MoE 路由不一致         | 注意 MoE 路由非确定性可能破坏 On-policy 假设；排查时需区分根因与表象 | LinkedIn         |
| 小模型训练效果不佳     | 通过蒸馏提升基础能力后再进行 RL；采用 CPT → SFT → RL 三阶段管线      | Amazon / Alibaba |

## 参考资料 {#references}

- Zhu J, Sang H, et al. "[Unlocking Agentic RL Training for GPT-OSS: A Practical Retrospective](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)." Hugging Face Blog, 2026.
- Zhuang R, Vu T, et al. "[Improving Multi-Turn Tool Use with Reinforcement Learning](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)." Bespoke Labs Blog, 2025.
- Moonshot AI. "[Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)." 2025.
- Tongyi DeepResearch Team. "[Tongyi DeepResearch: From Chatbot to Autonomous Agent](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)." 2025. [GitHub](https://github.com/Alibaba-NLP/DeepResearch)
- Salesforce AI Research. "[Building Efficient RL Training for the Agentic Era](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)." 2026.
- Subramanian S, Xu P, Wang Y. "[Customizing Multiturn AI Agents with Reinforcement Learning](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)." Amazon Science Blog, 2026.

[^carr_industrial]: Zhang J, Lv X, Feng L, Hou L, Li J. "[Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards](https://arxiv.org/abs/2601.06021)." arXiv, 2026.

[^selfrag_industrial]: Asai A, et al. "[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)." ICLR 2024.

[^critic_industrial]: Gou Z, et al. "[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)." ICLR 2024.

---

本节梳理了 Agentic RL 训练中的常见工程问题及工业界的解决方案。下一节将进入 [10.3 节：工业实践、评测与 Badcase](./industrial-evaluation)，看看怎么用 benchmark、评测管线和 badcase 归因判断 Agent 到底有没有变好。
