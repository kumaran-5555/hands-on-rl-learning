---
title: B.4 Training Metrics Glossary
---

# B.4 Training Metrics Glossary

The first time you open an RL training log or monitoring dashboard, the number of metrics can be intimidating:
`actor/pg_clipfrac`, `critic/advantages/mean`, `timing_s/gen`, `perf/mfu/actor` ...

Do not panic. Think about watching a student prepare for an exam:

- the **final exam score** is what you care about most,
- **daily homework accuracy** tells you whether they are improving,
- **study time** tells you whether they are efficient,
- the **distribution of mistakes** tells you where they are weak.

Training metrics work the same way. Each metric describes one slice of the training process.

This glossary organizes common metrics in the order of "what you should look at first." It is broadly applicable to veRL, OpenRLHF, TRL, and related stacks, though exact field names may differ.

---

## 1. Validation Metrics: "What Score Did You Get on the Final Exam?"

Regardless of algorithm and hyperparameters, the only question that ultimately matters is:

**did the model get better?**

Validation metrics answer that question. You run evaluation periodically during training, on held-out tasks, to estimate generalization.

| Metric                    | What It Means                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `val-core/*/acc/mean@1`   | validation accuracy: how often the model is correct; the number you should watch first |
| `val-aux/*/reward/mean@1` | mean validation reward; for rule-based rewards it often correlates with accuracy       |
| `val-aux/num_turns/*`     | turn-count stats during validation (for multi-turn tasks)                              |

Read `val-core` first, then use `val-aux` to interpret why it moved.

If validation accuracy starts to drop, do not immediately tune the optimizer. First ask: is training unstable (Section 2/3), or is data/reward mis-specified (Section 6/7)?

---

## 2. Actor Metrics: "Daily Homework and Practice"

Once you know whether validation is improving, the next step is to understand whether the training process is healthy. Actor metrics record the policy optimization dynamics.

### 2.1 Loss: What the Actor Is Actually Optimizing

| Metric               | What It Means                                                  |
| -------------------- | -------------------------------------------------------------- |
| `actor/loss`         | total actor loss (often `pg_loss + regularizers`)              |
| `actor/pg_loss`      | core policy-gradient loss                                      |
| `actor/entropy_loss` | entropy regularization loss (prevents over-confident collapse) |
| `actor/kl_loss`      | KL penalty loss term (only when enabled)                       |
| `actor/kl_coef`      | weight of the KL penalty                                       |

Reading guidance: `actor/loss` should trend down overall. Sudden jumps often indicate too-large learning rates, abrupt distribution shifts, or oscillatory updates. `pg_loss` is noisy by nature, but sustained blow-ups are a red flag.

### 2.2 Gradient Norm: Is the Update Magnitude Exploding?

| Metric            | What It Means                                                |
| ----------------- | ------------------------------------------------------------ |
| `actor/grad_norm` | L2 norm of gradients; "how hard the parameters want to move" |

If `grad_norm` spikes to 10× its normal value, suspect gradient explosion, bad batches, or numerical issues.

### 2.3 Entropy: How "Uncertain" the Policy Is

| Metric          | What It Means                           |
| --------------- | --------------------------------------- |
| `actor/entropy` | mean entropy of the policy distribution |

High entropy early is normal (exploration). Low entropy later is normal (commitment). But if entropy collapses too early, you often get premature convergence (for example, "stand still" behavior in continuous control, or narrow pattern copying in LLM RL).

---

### 2.4 KL Divergence: How Far Has the Policy Moved from the Starting Point?

PPO-style methods are designed to keep policy updates conservative. KL divergence is the main signal for how far the current policy has moved away from the reference or old policy.

| Metric                         | What It Means                                        |
| ------------------------------ | ---------------------------------------------------- |
| `actor/approx_kl` (or similar) | approximate KL divergence between old and new policy |
| `actor/kl_loss`                | KL penalty term when an explicit KL loss is enabled  |
| `actor/kl_coef`                | coefficient applied to the KL penalty                |

Reading guidance: KL spikes often precede reward collapse. If KL remains near zero for a long time, the model may not be learning new behavior.

### 2.5 Clip Fraction: How Many Updates Were "Speed-Limited"?

Clip fraction records how often PPO-style clipping is activated. It is a direct signal of whether the policy update is too aggressive.

| Metric                           | What It Means                                |
| -------------------------------- | -------------------------------------------- |
| `actor/pg_clipfrac` (or similar) | clip fraction: how often updates are clipped |

Reading guidance: sustained high clip fraction suggests the learning rate is too high or batches are too small/noisy.

### 2.6 Learning Rate

Learning rate controls the size of each optimizer step. It is not a performance metric by itself, but it explains many other metrics.

| Metric           | What It Means                               |
| ---------------- | ------------------------------------------- |
| `actor/lr`       | actor learning rate                         |
| `critic/lr`      | critic learning rate, if present            |
| `lr_scheduler/*` | scheduler state, if logged by the framework |

Reading guidance: if loss, KL, and gradient norm all spike together, first check whether the learning rate schedule changed at the same time.

---

## 3. Critic / Reward Statistics: "Scores from the Teacher"

The critic matters because advantage estimates depend on it. If the critic fails, the actor receives garbage learning signals.

| Metric                   | What It Means                                                |
| ------------------------ | ------------------------------------------------------------ |
| `critic/loss`            | value-function regression loss                               |
| `critic/value/mean`      | mean predicted value; track for drift or saturation          |
| `critic/advantages/mean` | mean advantage; track for sign flips or collapse             |
| `critic/advantages/std`  | advantage variance; extreme variance often means instability |

Reading guidance: value loss should generally decrease over time. If it never decreases, suspect incorrect reward scaling, incorrect masks, or insufficient critic capacity.

---

## 4. Length Metrics: "Is the Response Too Long or Too Short?"

For LLM RL, rollout throughput and length distribution are first-class metrics. They determine both cost and stability.

| Metric                       | What It Means                                            |
| ---------------------------- | -------------------------------------------------------- |
| `response_length/mean`       | mean completion length                                   |
| `response_length/max`        | maximum completion length                                |
| `response_length/clip_ratio` | fraction of completions clipped by `max_response_length` |
| `prompt_length/mean`         | mean prompt length                                       |
| `prompt_length/clip_ratio`   | fraction of prompts clipped by `max_prompt_length`       |
| `response/aborted_ratio`     | fraction of generation failures/aborts                   |

Common interpretations:

- high clip ratio means truncation is frequent; adjust max lengths or shorten inputs
- reward rises while response length collapses can indicate reward hacking via length bias
- high aborted ratio often indicates sampling/runtime issues

---

## 5. Preference Optimization Metrics (DPO/KTO/SimPO): "Preference Learning Report Card"

DPO-family methods learn from preference data rather than on-policy rollouts, so metric sets differ.

| Metric             | What It Means                                                |
| ------------------ | ------------------------------------------------------------ |
| `loss/dpo`         | total DPO loss; should decrease                              |
| `rewards/chosen`   | implicit reward for preferred responses                      |
| `rewards/rejected` | implicit reward for rejected responses                       |
| `rewards/margins`  | chosen minus rejected; key signal of discrimination strength |
| `rewards/accuracy` | fraction where the model prefers the chosen response         |

If margins increase but validation does not improve, suspect overfitting to training pairs.

---

## 6. Reward Model (RM) Training Metrics: "Training the Judge"

If you rely on an RM to score outputs, RM quality is a critical dependency.

| Metric             | What It Means                                                      |
| ------------------ | ------------------------------------------------------------------ |
| `rm/loss`          | RM preference loss (often cross-entropy)                           |
| `rm/accuracy`      | how often RM picks the better response                             |
| `rm/reward_margin` | score gap between chosen and rejected; "how decisive the judge is" |
| `rm/grad_norm`     | RM gradient norm                                                   |

Common failure signs:

- high RM accuracy but downstream PPO does not improve: RM may be overfitting or misaligned with real quality
- margin approaches zero: RM becomes non-discriminative
- RM score behavior changes drastically across RM versions for the same samples: RM may be contaminated by policy outputs

---

## 7. Multi-Turn Interaction Metrics: "How Many Turns Did It Take?"

For agent tasks, turn-count distribution is important:

| Metric           | What It Means |
| ---------------- | ------------- |
| `num_turns/min`  | minimum turns |
| `num_turns/max`  | maximum turns |
| `num_turns/mean` | mean turns    |

If `num_turns/max` suddenly becomes very large, investigate potential loops or tool failures that cause repeated retries.

---

## 8. Load Balancing Metrics: "Is Work Evenly Distributed?"

Previous sections focus on "is the model learning well." From here, we focus on "is training running fast."

`global_seqlen/*` metrics measure whether workload is balanced across GPUs in multi-GPU training. They represent total token workload per GPU (partition / rank), not the length of any single sequence.

| Metric                       | What It Means                            |
| ---------------------------- | ---------------------------------------- |
| `global_seqlen/min`          | tokens assigned to the lightest GPU      |
| `global_seqlen/max`          | tokens assigned to the heaviest GPU      |
| `global_seqlen/minmax_diff`  | `max - min`; smaller means more balanced |
| `global_seqlen/balanced_min` | lightest GPU after load balancing        |
| `global_seqlen/balanced_max` | heaviest GPU after load balancing        |
| `global_seqlen/mean`         | average workload per GPU                 |

Reading guidance: `minmax_diff` is the most important. Imagine a team project where one person does 80% of the work while others idle — progress is bottlenecked by that one person. Multi-GPU training works the same way. Large `minmax_diff` means unbalanced load, where fast GPUs wait for slow ones.

---

## 9. Timing Metrics: "Where Does Time Go Per Step?"

| Metric                    | What It Means                                                                |
| ------------------------- | ---------------------------------------------------------------------------- |
| `timing_s/step`           | total wall time for one step                                                 |
| `timing_s/gen`            | seconds for generation; usually the largest component                        |
| `timing_s/reward`         | seconds for reward computation; rule rewards are fast, model scoring is slow |
| `timing_s/old_log_prob`   | seconds for old-policy log probability and entropy                           |
| `timing_s/ref`            | seconds for reference model log probability                                  |
| `timing_s/adv`            | seconds for advantage computation                                            |
| `timing_s/update_actor`   | seconds for actor parameter update                                           |
| `timing_s/update_critic`  | seconds for critic parameter update (PPO only, not GRPO)                     |
| `timing_s/update_weights` | seconds for syncing updated weights to the generation engine                 |

Reading guidance: First check `timing_s/step` for overall pace, then drill into each component. In most cases, `timing_s/gen` is the bottleneck. If you want to optimize training speed, start with the slowest component.

### Finer-Grained Timing

For Agentic scenarios, there are finer timing breakdowns:

- `timing_s/agent_loop/generate_sequences/min|max|mean`: per-sample generation time distribution
- `timing_s/agent_loop/tool_calls/min|max|mean`: tool call timing
- `timing_s/agent_loop/slowest/*`: details of the slowest sample

### Per-Token Normalized Timing

`timing_per_token_ms/*` divides time by token count, giving "milliseconds per token." This is better for **cross-run comparisons** — e.g., PPO vs GRPO generation efficiency — since different runs may generate different token counts.

Common fields: `timing_per_token_ms/gen`, `timing_per_token_ms/ref`, `timing_per_token_ms/adv`, `timing_per_token_ms/update_actor`.

---

## 10. Performance Metrics: "Is GPU Utilization High Enough?"

| Metric                  | What It Means                                          |
| ----------------------- | ------------------------------------------------------ |
| `perf/total_num_tokens` | total tokens processed this step                       |
| `perf/time_per_step`    | total wall time per step                               |
| `perf/throughput`       | tokens per second per GPU — the core efficiency metric |
| `perf/mfu/actor_infer`  | MFU (Model FLOPs Utilization) during actor inference   |
| `perf/mfu/actor`        | MFU during actor training update                       |

### What Is MFU

MFU measures "how much of the GPU's compute capacity is actually used." Think of it like factory capacity utilization — if the factory can produce 100 units/hour but only produces 40, utilization is 40%.

During normal training, MFU is typically 50%–60%. Below 30% means GPUs are spending a lot of time waiting (communication bottlenecks, data loading bottlenecks, or load imbalance), with room for optimization.

---

## 11. Framework Differences Note

Different frameworks may use different field names for the same metric. The core meaning is the same; only the naming differs.

| Meaning         | veRL / OpenRLHF        | TRL                                 |
| --------------- | ---------------------- | ----------------------------------- |
| Policy entropy  | `actor/entropy`        | `policy/approx_kl` or `entropy`     |
| KL divergence   | `actor/ppo_kl`         | `objective/kl`                      |
| Gradient norm   | `actor/grad_norm`      | gradient monitoring in `loss/total` |
| Response length | `response_length/mean` | `response_length`                   |
| Generation time | `timing_s/gen`         | usually not logged separately       |
| Reward mean     | `critic/rewards/mean`  | `rewards`                           |
| DPO margin      | `rewards/margins`      | `rewards/margins` (consistent)      |

If you are using TRL's `SFTTrainer` + `DPOTrainer`, metrics are automatically logged to wandb/tensorboard. See TRL documentation's Logging section for field names.

---

## 12. Quick Reference: What to Watch by Scenario

### PPO / GRPO Training

1. **Effectiveness first**: `val-core/*/acc/mean@1` (accuracy), `critic/rewards/mean` (reward trend)
2. **Stability next**: `actor/loss`, `actor/grad_norm`, `actor/ppo_kl`, `actor/pg_clipfrac`
3. **Efficiency**: `perf/throughput`, `timing_s/gen`
4. **Data**: `response_length/mean`, `response_length/clip_ratio`, `global_seqlen/minmax_diff`

### DPO / KTO Training

1. **Effectiveness first**: `rewards/margins` (discrimination), `rewards/accuracy` (preference accuracy)
2. **Stability next**: `loss/dpo` (loss trend)
3. **Overfitting check**: `rewards/accuracy` hits 99%+ but evaluation doesn't improve → overfitting

### Reward Model Training

1. **Effectiveness first**: `rm/accuracy` (preference prediction accuracy), `rm/reward_margin` (discrimination)
2. **Generalization**: does validation accuracy track training accuracy?

### Training Problems

| Symptom                      | Check First                         | Then Check                                     |
| ---------------------------- | ----------------------------------- | ---------------------------------------------- |
| Reward crashes               | did `actor/entropy` drop suddenly?  | is `actor/pg_clipfrac` spiking?                |
| KL spikes                    | `actor/ppo_kl` trend                | is the learning rate schedule correct?         |
| Eval stalls but reward rises | is `response_length/mean` changing? | is `response/aborted_ratio` high?              |
| Training is slow             | `timing_s/step` sub-items           | `perf/throughput`, `global_seqlen/minmax_diff` |
| OOM                          | is `global_seqlen/max` too large?   | `response_length/max`, `prompt_length/max`     |
