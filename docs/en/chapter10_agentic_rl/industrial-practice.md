---
search: false
---

# Legacy Page: Industrial Practice (Merged into 10.3)

> This page is kept as an entry point for legacy links. The core content has already been merged into [10.3 Industrial Practice, Evaluation, and Badcases](./industrial-evaluation). The original content is retained below so readers arriving through old links can compare it with the new chapter.

# 12.5 Industrial Practice: Common Problems and Solutions in Agentic RL Training

The previous sections introduced the general engineering principles and framework design of Agentic RL. In real training, however, researchers often run into a series of engineering problems: unstable training, uncontrolled output length, reward metrics that drift away from actual quality, and so on. These issues are usually not discussed in detail in academic papers, but they are crucial in engineering practice.

From 2025 to 2026, several teams, including Alibaba, Moonshot, LinkedIn, and Bespoke Labs, publicly shared their experience with Agentic RL training. This section is not organized team by team. Instead, it is organized **by the problem scenarios that are likely to appear in real training**, bringing together findings and solutions from different teams.

> **Core takeaway**: In Agentic RL, training stability is often more important than algorithm choice. Data quality and environmental consistency are key factors that determine training effectiveness.

---

## Scenario 1: Obtaining Training Data and Building the Environment

When researchers begin Agentic RL training, the first question is often this: how can we provide the model with a stable and reproducible interaction environment?

### The Limits of Real APIs: Lack of Reproducibility

If training directly connects to a real search engine or code execution environment, it immediately encounters a fundamental problem: **the outputs of the external environment are not reproducible**.

> When training Kimi-Researcher, **Moonshot AI** pointed out that the environment faced by an agent is dynamic. Even for the same input query, a search engine may return different results. Their training mainly used the **REINFORCE** algorithm, and they emphasized the importance of strict on-policy data generation for training stability [\[Reference\]](https://moonshotai.github.io/Kimi-Researcher/).

### Building Synthetic Environments

A practical alternative is to build deterministic synthetic environments, so the model can train under controlled conditions.

> **Alibaba's Tongyi team** (Tongyi DeepResearch) abandoned noisy and uncontrollable online APIs and built a synthetic training environment centered on an offline Wikipedia database and stable tool sandboxes.
>
> **Core method and concrete design**:
>
> 1. **Data and environment synthesis (WebShaper & AgentFounder)**: Because real web pages change frequently, the same query may produce inconsistent search results at different times. This seriously violates the Markov decision process (MDP) assumption in reinforcement learning. To address this, they developed **WebShaper**, which converts massive Wikipedia content into a static, structured offline search environment. They also used **AgentFounder** to automatically generate extremely difficult, PhD-level synthetic queries and reference answers. The **determinism** of this synthetic environment makes the mapping between actions and rewards absolutely stable across multiple rollouts.
> 2. **Asynchronous compute architecture (rLLM)**: The rollout phase of Agentic RL, where the model interacts with the environment to generate trajectories that may span dozens of steps, is extremely time-consuming. If a traditional synchronous RL architecture is used, with training and inference alternating on the same group of GPUs, training nodes will remain idle for long periods because of environment-interaction latency. Their **rLLM (Ray-based LLM)** asynchronous rollout service physically separates inference from training. Multiple worker nodes continuously interact with the environment through high-throughput inference engines such as vLLM, generate trajectories, and write them into a shared replay buffer. Dedicated trainer nodes, based on Megatron/FSDP, continuously sample from the buffer and compute gradient updates.
>
> **Deeper cause and engineering significance**:
> Experiments show that reinforcement learning in a highly controlled, noise-free synthetic environment can produce a model whose final generalization ability on the real internet is actually **comprehensively better** than training directly on noisy human expert annotations. The root reason is that what the model truly needs to learn during RL is the general decision logic of "how to search" and "how to reflect and retry based on results," not overfitting to particular search returns. Stable environment signals are the foundation of RL convergence [\[Reference\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/).

### The Effectiveness of Small-Scale Data

For researchers with limited resources, a small amount of high-quality data can still produce substantial gains.

> **Amazon Science** validated the feasibility of "few-shot customization" on the complex AppWorld benchmark. Instead of blindly collecting tens of thousands of noisy human interaction trajectories, they carefully constructed only **72 high-quality training examples** covering core tool-calling patterns, dependencies, and retry logic when API errors occur. Through RL training, they raised the task completion rate of Qwen-2.5-32B from 39.2% to 72%, surpassing the strongest closed-source models at the time, Claude Sonnet 3.7/4.0.
>
> **Core method and deeper cause**:
> This counterintuitive result reveals a central insight of modern Agentic RL: for base models above 32B parameters, the model has already acquired strong world knowledge and logical reasoning ability during pretraining. At that point, RL is not mainly "injecting new knowledge into the model," but rather "eliciting and aligning" the model's interaction paradigm and tool syntax in a specific environment. As long as those 72 high-quality examples can serve as a primer and successfully trigger effective exploration in the environment, RL algorithms such as PPO/GRPO can use reward signals from the environment to let the model improve its policy through tens of thousands of self-play trials. This demonstrates that **when the base capability is sufficient, reinforcement learning can be extremely data-efficient: small but refined data plus autonomous RL exploration is far better than massive low-quality SFT data** [\[Reference\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning).

---

## Scenario 2: Gradient Explosion at the Beginning of Training

After the data and environment have been prepared, gradient explosion at the start of training is another common problem. Before tuning hyperparameters, first check whether the underlying implementation is correct.

### Implementation Differences Between the Inference Engine and the Training Engine

Agentic RL training contains two phases: the **inference (rollout)** phase generates action sequences, and the **training (backward)** phase updates model weights. These two phases are usually handled by different engines, and implementation differences between engines can make gradient computation inconsistent.

> When the **LinkedIn team** used GPT-OSS, an open-source model with an MoE architecture, for RL training, they encountered gradient explosion and stagnant rewards. After investigation, they found that the root cause was that **backpropagation for the Attention Sink parameters had not been implemented** in the training framework. The inference engine, a Triton kernel used by SGLang, supported the forward computation of Attention Sink, but the training framework, FlashAttention-v2 used by FSDP, completely lacked the corresponding support. They took the forward implementation from vLLM's FlashAttention branch and wrote the backward code themselves to compute gradients for the sink parameters. Only after fixing this issue did training become stable again [\[Reference\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl).

**Practical advice**: When using complex model architectures, first validate the training pipeline on a simple single-turn task, such as GSM8K. Confirm that the loss decreases normally before switching to multi-turn agent tasks.

---

## Scenario 3: Uncontrolled Output Length and Format Collapse

This is one of the most common problems in Agentic RL training: the model fails to learn correct tool use and instead begins to generate large amounts of meaningless tokens, eventually degenerating into repeated garbled output. This phenomenon is called **format collapse**:

```json
// Expected output format:
{"action": "search", "query": "AAPL stock"}

// Output after format collapse:
{"action": "searchsearchAAPL stockAAAAA"
```

Below we analyze the three main causes of this problem and their corresponding solutions.

### Cause 1: The Reward Function Is Too Complex

Intuitively, researchers may design multidimensional reward signals: +1 for successful tool calls, +1 for correct output format, and +5 for a correct final answer. However, this kind of fine-grained reward design can backfire.

**Reward hacking** is the core problem. When a reward function contains multiple subitems that the model can optimize independently, the model may discover strategies that satisfy only some conditions while still obtaining high reward.

> Experiments from **Bespoke Labs** showed that a composite reward function containing rewards for tool-call count, format checks, and correctness actually reduced training stability, likely because of reward hacking. They also observed output length continually expanding and eventually degenerating into meaningless garbled characters. Their final approach was to **keep only one binary reward signal: whether the task was completed**. Passing the BFCL evaluation check gives 1; otherwise the reward is 0. After deleting all intermediate process rewards, training stability improved significantly [\[Reference\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning).

The logic behind this finding is that a binary final-outcome reward provides no "shortcut" in the intermediate steps. The model must complete the task as a whole to obtain positive reward, which prevents opportunistic behavior aimed at individual reward terms. Bespoke Labs also observed that under composite rewards, output length kept expanding and eventually degenerated into garbled text; simplifying the reward design also alleviated this issue.

### Cause 2: Improper Handling of Negative Samples

During training, not all samples that fail to complete the task have the same quality. For example, the model may be truncated by the environment because it reaches the interaction-step limit. In that case, it has not produced a final answer, but its previous outputs may still be reasonable. If such samples are indiscriminately treated as negative examples and penalized, the model's already learned output ability may be damaged.

> **Alibaba's Tongyi team** observed that if all incomplete trajectories are treated as negative samples and penalized without filtering, long training can lead to severe **format collapse**. To avoid the global penalty caused by task failure, the model starts producing garbled text or completely refuses to use tools, because more actions create more opportunities for mistakes.
>
> **Core method and deeper cause**:
> To solve this long-horizon credit assignment problem, they adopted two core designs in a customized **on-policy GRPO (Group Relative Policy Optimization) algorithm**:
>
> 1. **Token-level loss and leave-one-out advantage estimation**: Compared with traditional PPO, which spreads the reward of an entire trajectory over every action, GRPO generates multiple candidate trajectories within a group, computes the relative advantage of each action compared with other actions in the group, and applies finer-grained gradient updates at the token level. This greatly reduces the variance of reward evaluation.
> 2. **Conservative negative filtering**: Agent actions have a strong causal sequence structure. In interactions lasting up to 30 steps, many trajectories ultimately fail, for example because they time out or reach the maximum number of interaction steps, but often only because of a logic error in the last few steps. The first 20 steps of the chain of thought (CoT) and tool-call format may be entirely correct. If this type of truncated sample is forced to receive a global negative reward such as `-1`, the RL optimizer will throw away the useful behavior along with the failed ending, incorrectly punishing format outputs that were originally correct. Therefore, they **selectively mask out these truncated samples from loss computation**, so they do not contribute negative gradients. This strategy effectively protects the model's basic alignment ability and preserves long-term stability in formatted output [\[Reference\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/).

### Cause 3: Misconfigured KL-Divergence Constraint

In RLHF/GRPO, a KL penalty is usually used to limit how far the current policy model deviates from the initial reference model. The role of the KL constraint is to prevent the policy from drifting too far from the initial model during training, thereby preserving basic output quality.

This constraint must balance "allowing the policy to explore" and "maintaining stability":

- **KL penalty too small**: The constraint is too weak, and the policy may drift too far from the initial model, causing output quality to degrade.
- **KL penalty too large**: The constraint is too strong, and the policy struggles to learn new behaviors, limiting the training effect.

> When **Bespoke Labs** trained Qwen2.5-7B-Instruct, they found that when the KL penalty was set to 0, the model's output degraded after roughly 300 steps. Their strategy was:
>
> 1. **Set a very small KL weight**, such as 0.001, to provide a minimal constraint.
> 2. **Periodically update the reference model**: every fixed number of steps, such as 100, copy the current policy model as the new reference model. In this way, the target of the KL constraint changes dynamically as training progresses, preventing the policy from being anchored to an initial state that is now too far away [\[Reference\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning).

### Output-Length Control: Gamma-Decay Reward

To encourage the model to complete tasks in fewer steps, we can introduce a reward mechanism that decays with the number of steps used.

> **Moonshot** proposed **Gamma-decay Reward**. When the model completes a task correctly, the reward value decays exponentially with the number of steps used:
>
> $$r_i = r \times \gamma^{T-i}, \quad \gamma < 1$$
>
> Here, $T$ is the total number of steps and $i$ is the current step. This means that when completing the same task, using fewer steps yields a higher reward, guiding the model to execute tasks more efficiently [\[Reference\]](https://moonshotai.github.io/Kimi-Researcher/).

---

## Scenario 4: Context Management in Long-Horizon Interaction

One important difference between Agentic RL and traditional RL is that the number of interaction turns can become very large. In complex tasks such as literature search, code writing, and debugging, the number of turns may exceed 50. At that point, the context window is filled with a large amount of historical information, and the model may lose focus on the original task.

### Context Management Mechanism

> **Moonshot's** Kimi-Researcher introduced a **context management** mechanism. This is a key engineering practice for solving attention dilution and "lost in the middle" problems in long-horizon tasks.
>
> **Core method and deeper cause**:
> In agent interactions lasting dozens of turns, if there is no control mechanism, redundant HTML tags from web pages and hundreds or thousands of lines of code-execution logs quickly fill a context window that may contain hundreds of thousands of tokens. As context length grows sharply, the signal-to-noise ratio (SNR) of the LLM drops significantly, causing the model at turn 40 to "forget" the original user request from turn 1.
>
> To address this, Kimi introduced an independent `context_manager` mechanism. After each step, the system dynamically evaluates and **compresses the context**:
>
> 1. **Preserve core logic (working memory)**: Keep the model's own thoughts, historical actions, and key facts extracted from web pages in the core context.
> 2. **Summarize or discard noise**: Replace long raw web pages with one- or two-sentence summaries, or directly discard invalid search records that have already proven to be dead ends.
>
> Context management is essentially the maintenance of the agent's dynamic "working memory," ensuring that the input to each model decision is dense with useful information. Ablation experiments showed that enabling this mechanism not only avoided catastrophic forgetting, but also extended safe single-rollout interactions beyond 50 turns. The model could gather more clues and ultimately achieved significantly higher scores on complex research tasks [\[Reference\]](https://moonshotai.github.io/Kimi-Researcher/).

---

## Scenario 5: Agent Hallucination and Its Control

After solving training stability and output-format problems, another issue deserves attention: **agent hallucination**. The model may cite nonexistent papers from search results, or use API parameters incorrectly while showing inappropriate "confidence" in later reasoning. Hallucination in agent settings is more complex than in pure dialogue settings, because the model generates not only text but also actions.

### Four Types of Agent Hallucination

**Tool-selection hallucination.** The model calls a nonexistent tool, or forces a tool call when it should not call a tool. For example, the user asks about the weather, but the model calls `execute_sql`.

**Parameter hallucination.** The tool choice is correct, but the parameters are wrong: the model fabricates a nonexistent API endpoint, misspells a database name, or uses an incorrectly formatted parameter value. The most dangerous case is that the parameter format may look "reasonable," while the actual value is fabricated.

**Result hallucination.** This is the most hidden type of hallucination. The model calls the correct tool and obtains real returned results, but introduces bias while interpreting them, for example treating irrelevant information in the search results as evidence for its own claim, or ignoring content that contradicts its hypothesis.

**Citation hallucination.** The model claims that a conclusion is based on "a certain paper" or "a certain website," but the citation does not actually exist, or the cited content does not match the original source. This is especially common in Deep Research agents. The model may fabricate paper titles, URLs, and statistics to make the output "look well supported."

### The Cascading Effect of Agent Hallucination

In pure dialogue settings, the consequences of hallucination are usually limited to providing incorrect information. In agent settings, however, hallucination can **cascade and reinforce itself** across multiple turns:

1. Turn 3: The model produces a parameter hallucination and calls a nonexistent API parameter -> the call fails.
2. Turn 4: The model fails to recognize the hallucination and instead believes "this API is defective" -> it switches to another tool.
3. Turn 5: The new tool lacks a key function -> the model fabricates a seemingly plausible conclusion.
4. Final output: a report that looks complete on the surface but is built on hallucinated foundations.

More importantly, if the RL reward is based only on final output quality, that is, an outcome reward, then in theory the model may discover that "fabricating a credible-looking answer" receives higher reward than "admitting uncertainty." This means RL training may actually **reinforce hallucination behavior**. This inference is logically sound, but it has not yet been explicitly reported as an observed phenomenon in publicly available industrial practice.

### Hallucination-Penalty Mechanisms in RL Training

**Citation-aware scoring rewards.** CaRR[^carr_industrial] (Citation-aware Rubric Rewards), proposed jointly by Tsinghua University and Zhipu AI, designs a fine-grained reward mechanism to guide the model toward correct evidence citation. Its core idea is to decompose multi-hop questions into a series of atomic factual statements, or rubrics, and then compute reward through a three-step process: (1) check whether the model output identifies the key entities; (2) extract the URLs cited in the output, retrieve the web-page content, and determine whether each rubric is supported by the cited content; (3) use breadth-first search on a graph to verify whether the rubrics are logically connected to the final answer. The final reward is the ratio of satisfied and logically connected rubrics to the total number of rubrics. This mechanism encourages the model to provide verifiable and logically coherent citation evidence for each claim.

**Tool-result faithfulness rewards.** Encourage the model to remain faithful to the original content when interpreting tool returns. If the model's summary deviates from the information actually returned by the tool, as detected through an NLI model or cross-checking, it receives a penalty.

**Uncertainty rewards.** Encourage the model to explicitly say "more information is needed" or "this result is uncertain" when it is uncertain, rather than fabricating an answer. Combining the three strategies above, we can design the following hallucination-aware reward function as an example:

> **Note**: The following code is an illustrative example that combines several penalty ideas. It is not a concrete implementation taken directly from any single paper.

```python
def hallucination_aware_reward(answer, tool_results, citations):
    """Hallucination-aware reward function"""
    reward = base_task_reward(answer)

    # 1. Citation authenticity check
    for citation in citations:
        if not verify_citation_exists(citation):
            reward -= 0.5  # False citation, penalty
        elif not verify_citation_supports(citation, answer):
            reward -= 0.3  # Citation does not support the claim

    # 2. Tool-result faithfulness
    for claim in extract_claims(answer):
        if has_supporting_evidence(claim, tool_results):
            reward += 0.1  # Evidence-backed claim
        elif claim_is_verifiable(claim) and not has_supporting_evidence(claim, tool_results):
            reward -= 0.2  # Verifiable but unsupported claim

    # 3. Encourage uncertainty expression (honesty reward)
    if is_complex_question and ("uncertain" in answer or "need more information" in answer):
        if not all_claims_supported(answer, tool_results):
            reward += 0.15  # Admitting uncertainty is reasonable when evidence is truly missing

    return reward
```

### Verification-Based Hallucination Filtering

In addition to penalizing hallucination in the reward function, we can also filter it through verification mechanisms during the **inference phase**:

**Self-RAG[^selfrag_industrial]** proposed a framework of "adaptive retrieval plus self-evaluation." Unlike traditional RAG, which retrieves for every query, Self-RAG lets the model decide whether external information is needed **before** generating each text segment, using special reflection tokens. If retrieval is needed, the system retrieves several relevant passages, generates a continuation for each passage, and scores each candidate continuation using reflection tokens such as [IsRel] for relevance, [IsSup] for support, and [IsUse] for usefulness. It then uses segment-level beam search to select the output with the highest overall score. The key feature of this framework is that the model performs structured self-evaluation of its own output through reflection tokens.

**CRITIC[^critic_industrial]** proposed a hallucination-filtering mechanism based on "tool-assisted correction." After the model generates an initial answer, it proactively calls external tools such as search engines or code executors to verify key claims, and then generates structured critiques based on tool feedback. If the critique indicates that the answer has problems, the model regenerates a corrected answer based on the critique. This "verify -> revise -> verify" loop can iterate multiple times until the answer passes verification or reaches the maximum number of iterations. Unlike methods that rely purely on the model's self-evaluation, CRITIC introduces objective feedback from external tools as the basis for correction.

### Practical Summary of Hallucination Control

| Hallucination Type           | Detection Method                     | RL Penalty Strategy                           |
| ---------------------------- | ------------------------------------ | --------------------------------------------- |
| Tool-selection hallucination | Tool whitelist validation            | Calling nonexistent tool -> reward = 0        |
| Parameter hallucination      | Schema validation + type checking    | Incorrect parameter format -> negative reward |
| Result hallucination         | NLI model + cross-checking           | Claim contradicts tool result -> penalty      |
| Citation hallucination       | URL reachability + content relevance | False citation -> penalty                     |

One important practical principle is: **hallucination penalties should be introduced early in training**. Once hallucination behavior has been reinforced through RL, it becomes very difficult to eliminate later.

---

## Scenario 6: Considerations for Specific Model Architectures

The previous scenarios are common problems that appear in most Agentic RL training. In addition, when using specific model architectures such as MoE, or when training smaller models, several additional issues may arise.

### Routing Uncertainty in MoE Models

MoE models, such as Mixtral and DeepSeek-V3, have attracted attention because of their lower inference cost, but their routing mechanism may break a basic assumption of RL training.

Algorithms such as PPO assume that the model generating the current data and the model being trained are the same model, that is, training is on-policy. Mathematically, this appears as an importance-sampling ratio equal to 1.

> When the **LinkedIn team** used GPT-OSS for RL training, they found that the routing network, or gating network, of the MoE model could choose different experts for the same token across two forward passes. This caused $\log \pi(a|s) \neq \log \pi_{\text{old}}(a|s)$, meaning that the on-policy assumption was broken. During debugging, they tried setting `old_log_prob = log_prob.detach()` to force the two probabilities to align and verify this hypothesis. It should be noted that although this routing inconsistency is real, it was not the root cause of gradient explosion in their debugging. The root cause was the missing Attention Sink backpropagation described in the previous section [\[Reference\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl).

### Load-Balancing Problems in MoE Models

In RL training, MoE models face not only the routing-consistency problem above, but also low GPU utilization caused by imbalanced expert loads. Different tokens may concentrate on a few "popular" experts, making the GPUs responsible for those experts the bottleneck while other GPUs remain idle.

> **Salesforce** proposed a **pipelined synchronous RL** design in its SFR-RL system. All GPUs alternate between the rollout and training phases, instead of being permanently assigned to one phase. In addition, for MoE models, they introduced **Least-Loaded Expert Parallelism** to optimize expert load balancing. Compared with VERL (FSDP + Context Parallelism), the overall system improves memory efficiency by about 250x and can train a 120B-parameter MoE model using only 16 H200 GPUs [\[Reference\]](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/).

### The Reasoning-Capability Ceiling of Small Models

It is important to remember that the essence of RL is to **elicit capabilities the model already has**, not to inject new knowledge. The model's base capability determines the upper bound of what RL can achieve.

> Experiments from **Amazon Science** showed that 32B-parameter models benefit significantly from RL, because the model itself can generate high-quality interaction trajectories, forming a positive feedback loop. Smaller models, however, are limited by their base reasoning ability. For example, they may fail to recognize unanswerable questions or extract answers from relevant context. RL training has difficulty making up for the absence of these capabilities. For small models with insufficient base capability, the researchers recommend using distillation from stronger models to acquire ability, rather than simply increasing the intensity of RL training [\[Reference\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning).

### Staged Training Pipelines

Given the characteristics of models at different scales, a more robust training strategy is to use a staged pipeline instead of going directly into RL training.

At present, industry practice contains two parallel training paradigms regarding whether SFT is necessary: the **SFT-RL paradigm** and the **pure-RL paradigm**.

> **SFT-RL paradigm (mainstream path)**: In Tongyi DeepResearch, **Alibaba's Tongyi team** designed a three-stage **CPT -> SFT -> RL** training pipeline. In the continual pretraining (CPT) stage, tool-calling trajectories are incorporated in text form. In the SFT stage, human or high-quality synthetic data is used to cultivate the model's basic reasoning and tool-use ability. Finally, the RL stage performs exploration and optimization. The core of this paradigm is that in non-reasoning alignment settings, such as complex API calls and long-horizon exploration, SFT/RM remains the most effective way to **reduce the exploration space and overcome cold-start difficulty**. If the model does not possess the basic tool-use format at the starting point, direct RL training often gets lost in the enormous action space [\[Reference\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/).

> **Pure-RL paradigm (frontier breakthrough)**: In sharp contrast, **DeepSeek-R1-Zero** brought a paradigm shift. It demonstrated that in settings with clear right-or-wrong feedback, such as mathematics, code tests, and objective reasoning, it is entirely feasible to **discard the SFT cold start completely** and conduct large-scale reinforcement learning directly from the base model. Driven by pure RL, the model can spontaneously develop advanced reasoning abilities such as long chain of thought (CoT), self-verification, and even self-reflection. This SFT-free, bias-free training approach breaks through the ceiling imposed by human annotation data, but it places extremely high demands on the objectivity of reward signals and the environment's resistance to cheating.

These two paradigms are not mutually exclusive in Agentic RL. Researchers should choose an appropriate pipeline based on whether the environment provides fully deterministic objective rewards.

---

## Practical Summary {#tricks}

The table below summarizes the corresponding solutions for each problem:

| Problem                                       | Solution                                                                                                                                   | Reference        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| Training environment is not reproducible      | Build a deterministic synthetic environment                                                                                                | Alibaba          |
| Small-scale data customization                | Small amounts of high-quality data, such as 72 examples, can work well when combined with RL                                               | Amazon           |
| Gradient explosion early in training          | Check low-level consistency between the inference and training engines, such as Attention Sink backpropagation                             | LinkedIn         |
| Output degenerates into repeated garbled text | Use a minimal reward design, rewarding only task success or failure; filter overly long outputs                                            | Bespoke Labs     |
| Policy drifts away from the initial model     | Set a small KL penalty, such as 0.001; periodically use the current model as the new reference model                                       | Bespoke Labs     |
| Low output efficiency (too many steps)        | Use Gamma-decay rewards to encourage task completion in fewer steps                                                                        | Moonshot         |
| Format collapse                               | Use conservative negative-sample handling and exclude trajectories that were truncated by excessive length before producing a final answer | Alibaba          |
| Context overflow in long tasks                | Introduce a context management mechanism to actively summarize or discard useless historical information                                   | Moonshot         |
| Low resource utilization in MoE training      | Pipelined synchronous RL + Expert Parallelism; 16 H200 GPUs can train a 120B MoE model                                                     | Salesforce       |
| Inconsistent MoE routing                      | Note that MoE routing nondeterminism may break the on-policy assumption; distinguish root causes from symptoms during debugging            | LinkedIn         |
| Poor training results for small models        | Improve base capability through distillation before RL; use a CPT -> SFT -> RL three-stage pipeline                                        | Amazon / Alibaba |

## References {#references}

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

This section reviewed common engineering problems in Agentic RL training and the corresponding solutions used in industry. The next section moves to [Section 10.3: Industrial Practice, Evaluation, and Badcases](./industrial-evaluation), where we examine how to use benchmarks, evaluation pipelines, and badcase attribution to judge whether an agent has truly improved.
