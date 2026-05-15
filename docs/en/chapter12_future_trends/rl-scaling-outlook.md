---
title: 12.6 RL Scaling and Future Outlook
---

# 12.6 RL Scaling and Future Outlook: Where Is the Ceiling?

In the previous three sections, we traced the evolution of RL training from PPO to GRPO to DAPO/RLVR. On the strategy side, the Critic was removed; on the reward side, the RM was removed; training costs decreased step by step. But a more fundamental question remains unanswered: **Is there a ceiling to the benefits of RL training? Can we continue improving by investing more compute?**

One of the most exciting discoveries of 2025 is that **the returns from RL training have not yet saturated**. Continuing to increase RL training scale still yields steady improvements in reasoning ability. This section discusses the future of RL from three dimensions: training paradigm choice (Online vs. Offline), three directions of RL Scaling, and Test-time Scaling with process reward models.

## Comparing Three Training Paradigms

After learning DPO (Offline) and GRPO/DAPO (Online), a natural practical question is: **Which paradigm should you choose?**

|                            | Offline (DPO)                      | Online (PPO/GRPO)                       | Semi-Online                     |
| -------------------------- | ---------------------------------- | --------------------------------------- | ------------------------------- |
| **Data source**            | Fixed offline preference dataset   | Generated in real-time by current model | Offline data + periodic updates |
| **Exploration**            | None (limited by dataset)          | Yes (model explores new strategies)     | Partial                         |
| **Theoretical ceiling**    | Limited by data quality            | Higher in principle                     | Compromise                      |
| **Engineering complexity** | Low (standard supervised learning) | High (online sampling loop)             | Medium                          |
| **Memory requirements**    | Low                                | High                                    | Medium                          |
| **Representative methods** | DPO, KTO, SimPO, IPO               | PPO, GRPO, DAPO                         | Iterative DPO, RLOO             |
| **Analogy**                | Learning to drive from videos      | Learning by actually driving            | Videos + occasional practice    |

### Practical Recommendations

A workflow validated by extensive practice goes like this:

1. **Step 1: DPO for rapid validation**. First use DPO to verify data quality and model baselines. DPO is the simplest and fastest; if even DPO cannot train well, the data has issues, and switching to PPO/GRPO will not help.
2. **Step 2: GRPO to raise the ceiling**. After DPO validation passes, switch to GRPO for online optimization. GRPO's online exploration capability can break through DPO's data limitations.
3. **Step 3: DAPO for fine-tuning**. If compute budget allows, use DAPO's dynamic sampling and token-level loss to further improve efficiency.

```python
# ==========================================
# Typical training code comparison for three paradigms (pseudocode)
# ==========================================

# ---- Offline (DPO) ----
# Feature: simplest, only needs preference dataset
# dpo_trainer = DPOTrainer(model, ref_model, dataset=preference_pairs)
# dpo_trainer.train()

# ---- Online (GRPO) ----
# Feature: online sampling, no Critic needed
# grpo_trainer = GRPOTrainer(model, reward_fn=rule_based_reward, k=8)
# grpo_trainer.train()

# ---- Semi-Online (Iterative DPO) ----
# Feature: periodically generate new data with current model, then train with DPO
# for iteration in range(num_iterations):
#     new_data = model.generate_and_label(prompts)  # generate + label
#     dpo_trainer.train_on(new_data)                 # DPO training
#     model = dpo_trainer.get_updated_model()        # update model

print("Training paradigm decision tree:")
print("  Data quality uncertain? → DPO first for validation")
print("  DPO validation passed? → GRPO to raise ceiling")
print("  Limited compute? → Iterative DPO (semi-online)")
print("  Pursuing maximum performance? → DAPO (dynamic sampling + token-level loss)")
```

## RLMT: Moving "Thinking" from Math to General Chat

The three paradigms discussed above (DPO/GRPO/DAPO) and this chapter's RLVR all focus on one question: **how to make models reason better on math and code**. But a natural follow-up is — can this "think before answering" capability also be applied to general chat, creative writing, and other open-ended scenarios?

A 2025 paper, "Language Models that Think, Chat Better," answers yes, proposing **RLMT (Reinforcement Learning with Model-rewarded Thinking)**.

### The Dilemma of Existing Methods

| Method | Chain of Thought | Reward Source                     | Applicable Domain | Shortcoming                          |
| ------ | ---------------- | --------------------------------- | ----------------- | ------------------------------------ |
| RLHF   | None             | Human preference reward model     | General chat      | No thinking, insufficient depth      |
| RLVR   | Yes              | Rules / ground truth              | Math/Code         | Cannot generalize to open-ended chat |
| RLMT   | **Yes**          | **Human preference reward model** | **General chat**  | Reward model quality is critical     |

RLHF has the model directly output answers without deep reasoning; RLVR forces the model to write long chains of thought, but the reward signal (answer correctness) only applies to tasks with ground-truth answers. RLMT's core insight is: **retain RLVR's "think before answering" structure, but use RLHF's preference reward model for scoring** — so the chain of thought can serve general chat.

### RLMT Training Methods

RLMT has two routes, similar to DeepSeek-R1's SFT route and Zero route:

**Route 1: SFT warm-up + RLMT**

1. First use Gemini/GPT-4 to generate "thinking process + final answer" data for supervised fine-tuning, teaching the model "what a chain of thought looks like"
2. Then use GRPO online reinforcement learning for optimization, with reward signals from the preference reward model

**Route 2: RLMT-Zero (train directly from base model)**

No SFT at all; apply RLMT training directly to the base model. The results are surprising:

- Only **7K real conversation prompts**
- Llama-3.1-8B base + RLMT-Zero
- Results **exceed** Llama-3.1-8B-Instruct trained with 25 million samples in multiple stages

This shows that the "think before answering" capability does not need to be taught via SFT — RL training itself can elicit it, consistent with DeepSeek-R1's findings.

### Results: Thinking Small Models > Non-Thinking Large Models

Comprehensive validation on Llama-3.1-8B and Qwen-2.5-7B:

- **Chat benchmarks** (AlpacaEval2 / WildBench / ArenaHardV2) improved by 3–7 points on average
- **Creative writing, commonsense, instruction following** improved by 1–3 points consistently
- Llama-3.1-8B-Instruct + RLMT **exceeds GPT-4o** in chat and creative writing, approaching Claude 3.7 Sonnet
- Significantly better than 10x larger Llama-3.1-70B and Qwen2.5-72B

This result again validates a key finding: **the "thinking ability" brought by RL training can compensate for differences in model scale.**

### What Kind of "Thinking" Does the Model Learn?

The paper analyzes changes in the model's thinking patterns before and after RLMT training:

| Phase      | Thinking Characteristics                                    | Analogy                                             |
| ---------- | ----------------------------------------------------------- | --------------------------------------------------- |
| SFT phase  | Linear listing, bullet points, rigid planning               | A clerk filling out a form                          |
| After RLMT | Organize constraints → group → weigh perspectives → iterate | An experienced consultant reasoning at a whiteboard |

Meanwhile, the model automatically increases chain-of-thought and answer length — not artificially set, but naturally emerging during RL optimization: longer thinking → better answers → higher rewards.

### RLMT Practical Points

```python
# ==========================================
# Key differences between RLMT and RLVR (pseudocode)
# ==========================================

# ---- RLVR (learned in this chapter) ----
# Reward = whether the answer is correct (rule verification)
# def rlvr_reward(response, question):
#     answer = extract_answer(response)
#     return 1.0 if answer == ground_truth else 0.0

# ---- RLMT (new in this section) ----
# Reward = preference reward model score (general chat quality)
# def rlmt_reward(response, question):
#     # response contains <think thinking process</think + final answer
#     return preference_reward_model(question, response)

# Key differences:
# 1. RLMT response structure = <think thinking</think + answer
# 2. Reward signal comes from preference RM, not rule verification
# 3. Training prompts must be close to real user chat; too many math problems actually hurts

print("RLMT practical points:")
print("  Reward model quality is critical — a weak RM will ruin performance")
print("  Training prompts must be close to real user chat scenarios")
print("  GRPO significantly outperforms PPO and DPO; best suited for thinking-style training")
print("  Base model can be directly aligned with RLMT, overturning traditional three-stage training")
```

### RLMT's Connection to Previous Chapters

RLMT stands at the intersection of Chapter 8 RLVR and Chapter 7 RLHF:

| Concept Source                       | Role in RLMT                                         |
| ------------------------------------ | ---------------------------------------------------- |
| RLVR's long chain of thought (Ch8)   | Retain the "think before answering" output structure |
| RLHF's preference reward (Ch7)       | Replace rule verification with preference RM         |
| GRPO's within-group comparison (Ch8) | The most effective online training method for RLMT   |
| DeepSeek-R1-Zero (Ch8)               | Direct inspiration for RLMT-Zero                     |

The significance of RLMT is: **it proves that "thinking" is not exclusive to mathematical reasoning; general chat also benefits from deep thinking.** This opens a new direction for RL training — instead of having the model think only on math problems, have it "think carefully before speaking" in all scenarios.

<details>
<summary>Discussion Question: Why can't RLVR's chain of thought be directly transferred to general chat, while RLMT can?</summary>

The core difference lies in **matching the reward signal**. RLVR's chain of thought is trained under the "answer correctness" reward signal — the model learns "how to think to get the correct answer." But general chat has no ground-truth answers; the "correct/incorrect" reward signal does not exist, so this thinking strategy fails.

RLMT's key insight is to replace the reward signal with a preference reward model. The preference RM can judge "whether this answer is good" (helpful, harmless, honest), not just "whether the answer is correct." The chain of thought trained under this reward signal naturally applies to general scenarios — the model learns "how to think to write better answers," not "how to think to calculate the correct answer."

This also explains why reward model quality is critical: if the preference RM itself has poor judgment, the chain of thought trained under its guidance will also go astray.

</details>

## RL Scaling: More Compute for Stronger Reasoning

One of the most exciting discoveries of 2025: **RL training returns have not yet saturated**. DeepSeek-R1's experiments show that for mathematical reasoning, RL training's scaling curve is steeper than SFT's. Continuing to increase training steps, the model's pass@1 continues to improve without clear saturation.

### Three Dimensions of RL Scaling

| Dimension          | Meaning                                        | Practical Method                             | Key Finding                                                       |
| ------------------ | ---------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------- |
| **Data scale**     | More training prompts of varying difficulty    | Auto-generate + filter for quality           | Diversity matters more than quantity                              |
| **Sampling scale** | More sampled answers per prompt (increasing k) | k from 4 to 16 or even 64                    | Within-group comparisons are more stable, but diminishing returns |
| **Training steps** | Longer RL training                             | Monitor KL divergence and evaluation metrics | Pass@1 continues to improve, not yet saturated                    |

```mermaid
flowchart LR
    subgraph scaling ["RL Scaling Three Dimensions"]
        D["Data Scale ↑\nMore problems\nMore difficulty levels"]
        S["Sampling Scale ↑\nk: 4 → 16 → 64\nMore stable group statistics"]
        T["Training Steps ↑\nLonger training\nContinued improvement"]
    end

    D --> Result["Model reasoning ability\nContinued improvement\nNot yet saturated"]
    S --> Result
    T --> Result

    subgraph caveat ["Key Prerequisites"]
        C1["Prompt diversity\nPrevent overfitting to training set"]
        C2["KL monitoring\nPrevent policy collapse"]
        C3["Validation set evaluation\nPrevent inflated metrics"]
    end

    style D fill:#e3f2fd,stroke:#1976d2
    style S fill:#fff3e0,stroke:#f57c00
    style T fill:#e8f5e9,stroke:#2e7d32
    style Result fill:#fce4ec,stroke:#c62828
```

The key prerequisite is sufficiently diverse prompt data. If the training data types are too narrow, the model will overfit to the training set — high scores on the training set but poor performance on different problems. DeepSeek-R1's solution is to use automated methods to generate and filter training problems, ensuring coverage across different difficulty levels and types.

### Agentic RL Scaling Laws

The three dimensions above focus on standard RL scaling. When RL enters Agentic scenarios (discussed in detail in Chapter 9), scaling takes new forms. The ZeroTIR method enables models to spontaneously learn to generate and execute code to assist reasoning **without supervised examples**, and discovers a predictable relationship: there is a **power-law relationship** between training steps, code execution frequency, and final accuracy. This means you can predict final performance early in training — if code execution frequency is still rising after 100 training steps, the model is still learning; if the frequency stabilizes, learning is approaching saturation. This finding gives practitioners a **free training progress indicator**: just monitor code execution frequency to determine "whether to continue training." ZeroTIR was accepted at NeurIPS 2025 and will be discussed in more detail in the Code Agent section of Chapter 9.

## Test-time Scaling: More Compute at Inference Time Too

Complementary to RL Scaling (investing more compute at training time) is another approach: **also let the model "think more" at inference time**.

Standard inference is "Prompt → model directly outputs answer." Test-time Scaling's approach is "Prompt → generate multiple candidates → verify/vote/search → select the best."

| Method                     | Principle                                                | Additional Cost             | Applicable Scenarios                  |
| -------------------------- | -------------------------------------------------------- | --------------------------- | ------------------------------------- |
| **Best-of-N sampling**     | Generate N answers, select the one with highest reward   | Grows linearly with N       | Simple, direct, general               |
| **Majority voting**        | Generate N answers, select the most frequent answer      | Grows linearly with N       | Math/code (has deterministic answers) |
| **MCTS / Tree of Thought** | Tree search in reasoning space, backtrack wrong branches | Exponential (needs pruning) | Complex reasoning tasks               |
| **Verifier-guided**        | Use a verifier to dynamically prune during reasoning     | Medium                      | Code/math                             |

### The Relationship Between RL and Test-time Scaling

An open frontier debate is: **Does RLVR only improve test-time search efficiency, rather than injecting genuine reasoning ability?**

- Supporters argue: Models trained with RL perform better with the same sampling budget, indicating that RL genuinely changes the model's internal policy, not just search efficiency.
- Skeptics argue: A base model without RL training, given enough sampling attempts (N → ∞), could theoretically achieve similar results — RL only makes the model "search more efficiently."
- The reality is: In practice, the efficiency improvement from RL training is enormous — enabling the model to produce high-quality answers with minimal sampling. Even if RL's essence is "improving search efficiency," this efficiency improvement is extremely valuable in engineering.

## PRM vs ORM: Process Supervision vs. Outcome Supervision

In reasoning scenarios, the credit assignment problem takes a concrete form: **should we only look at the final outcome (whether the answer is correct), or evaluate each reasoning step (whether intermediate steps are correct)?** This is the distinction between PRM (Process Reward Model) and ORM (Outcome Reward Model).

### ORM (Outcome Reward Model)

ORM only looks at the final outcome: correct answer gets positive reward, incorrect gets zero. Its advantage is simple annotation — you only need to know whether the final answer is correct. The disadvantage is sparse signal — out of 7 reasoning steps, only the last step has feedback; the correctness of intermediate steps is unknown.

### PRM (Process Reward Model)

PRM evaluates each reasoning step: is step 1 correct? Is step 2 correct? ... Each step has feedback. The advantage is dense learning signals that can precisely guide the improvement direction of each step. The disadvantage is extremely high annotation cost — requiring human experts to judge the correctness of each reasoning step.

### PRM's Practical Effect

| Method   | GSM8K Accuracy | MATH Accuracy | Annotation Cost |
| -------- | -------------- | ------------- | --------------- |
| ORM only | ~82%           | ~40%          | Low             |
| PRM only | ~85%           | ~45%          | Extremely high  |
| ORM + RL | ~88%           | ~50%          | Low             |
| PRM + RL | ~90%           | ~55%          | Extremely high  |

PRM's improvement is real (5 percentage points higher than ORM on MATH), but so is its cost. OpenAI's PRM800K dataset required math experts to annotate each reasoning step — a cost not every team can bear.

### Exploring Automated PRM

Since manual step-by-step annotation is too expensive, researchers have begun exploring automated process supervision:

```python
# ==========================================
# Auto PRM: estimating per-step correctness probability via Monte Carlo sampling
# ==========================================
def auto_prm(model, prompt, reasoning_steps, num_samples=32):
    """
    Estimate each reasoning step's correctness probability via Monte Carlo sampling

    Idea: starting from step i, re-sample subsequent reasoning N times
    Check the proportion of correct final answers → this is step i's "quality score"
    """
    step_scores = []

    for i in range(len(reasoning_steps)):
        # Keep the first i steps, re-generate subsequent steps
        correct_count = 0
        for _ in range(num_samples):
            # Re-sample starting from step i
            new_completion = model.generate(
                prompt + reasoning_steps[:i+1],
                temperature=0.7  # high temperature sampling, increase diversity
            )
            # Check if final answer is correct
            if check_answer_correct(new_completion):
                correct_count += 1

        # Step i's quality score = probability of getting correct answer from here
        step_scores.append(correct_count / num_samples)

    return step_scores

# Example output
# reasoning_steps = ["Let x = number of apples", "x = 15 - 3 - 5", "x = 7"]
# step_scores = [0.85, 0.90, 1.00]
# Step 1 has an 85% probability of eventually getting the correct answer, indicating a good start
```

The core idea of automated PRM is: **no need for humans to annotate each step; use Monte Carlo sampling to automatically estimate each step's "correctness probability."** Starting from step $i$, re-sample subsequent reasoning $N$ times and check the proportion of correct final answers — this is step $i$'s "quality score." The cost of this method is computational (each step requires $N$ samples), but it does not rely on manual annotation and has better scalability.

## Section Summary

In Chapter 8, we completed a full evolution path of RL training:

```mermaid
flowchart TD
    subgraph strategy ["Strategy Evolution"]
        S1["PPO\nActor + Critic + Ref + RM\n4 models"]
        S2["GRPO\nActor + Ref\n2 models (Critic removed)"]
        S3["DAPO\nDynamic sampling + Token-level loss\nMore efficient GRPO"]
    end

    subgraph reward ["Reward Evolution"]
        R1["RLHF\nManual annotation + RM training\nExtremely high cost"]
        R2["RLVR\nRule verification\nExtremely low cost"]
    end

    subgraph paradigm ["Paradigm Evolution"]
        P1["Offline (DPO)\nSimple but limited by data"]
        P2["Online (GRPO)\nComplex but higher ceiling"]
        P3["Semi-Online\nCompromise"]
    end

    S1 --> S2 --> S3
    R1 --> R2

    style S1 fill:#fce4ec,stroke:#c62828
    style S2 fill:#fff3e0,stroke:#f57c00
    style S3 fill:#e8f5e9,stroke:#2e7d32
    style R1 fill:#fce4ec,stroke:#c62828
    style R2 fill:#e8f5e9,stroke:#2e7d32
```

Three core dimensions can be independently chosen and flexibly combined:

- **Strategy side**: PPO → GRPO → DAPO (gradual simplification)
- **Reward side**: RLHF → RLVR (gradual automation)
- **Paradigm choice**: Offline → Online → Semi-Online (trade-off based on scenario)

The future of RL training has two clear directions: **RL Scaling** (investing more compute at training time) and **Test-time Scaling** (investing more compute at inference time). These two are complementary, jointly pushing improvements in model reasoning ability. The development of PRM and automated process supervision is expected to provide finer-grained training signals in the future, further accelerating RL training efficiency.

<details>
<summary>Discussion Question: Should you prioritize RL Scaling or Test-time Scaling?</summary>

This depends on your application scenario and resource constraints:

- **If inference cost is the bottleneck** (e.g., an online service serving millions of users), prioritize RL Scaling — train a stronger model so it can produce high-quality answers in a single inference pass without multiple sampling. Training cost is high but a one-time investment; inference cost is ongoing, and the savings far exceed the training investment.

- **If answer quality is the bottleneck** (e.g., math competitions, coding contests, where inference time doesn't matter), prioritize Test-time Scaling — use Best-of-N or MCTS to let the model "think more," spending more compute at inference time for better results.

- **If neither is a bottleneck** (e.g., internal research, small-scale deployment), investing a small amount in both is sufficient. RL Scaling's "data diversification" is the safest investment direction — more diverse training data is almost always beneficial.

It is worth noting that these two directions are not mutually exclusive. The most advanced systems (like DeepSeek-R1) use both RL Scaling (large-scale GRPO training) and Test-time Scaling (Best-of-N sampling at inference); the combined effect far exceeds either alone.

</details>

Here, the three directions of RL Scaling have been covered. But there is one important post-training approach we have not yet discussed — **knowledge distillation**: using the teacher model's log-prob as training signals to achieve results comparable to RL at 1/10 the compute. Let us move to the next section — [Knowledge Distillation and On-Policy Distillation](../chapter09_grpo_rlvr/on-policy-distillation).
