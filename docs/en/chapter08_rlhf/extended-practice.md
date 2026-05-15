---
title: 8.8 Extended Practice
---

# 8.8 Extended Practice: Reward Hacking and the Data Flywheel

## Reading Guide

**Core points**

- Observe how reward hacking occurs by using an intentionally flawed reward function.
- Learn to monitor reward together with length, repetition rate, KL, and manual quality inspection — not just a single curve.
- Organize badcase collection, error clustering, data supplementation, retraining, and regression evaluation into a data flywheel.

**Core formulas**

$$
R_{bad}(x,y)=0.01|y|+0.5\cdot\mathbb{1}[\text{has\_list}(y)]
+0.3\cdot\mathbb{1}[\text{has\_polite\_phrase}(y)]
\quad \text{(flawed reward: mistaking "good" for "long and template-like")}
$$

$$
R_{safe}(x,y)=
0.4R_{helpful}+0.3R_{correct}+0.2R_{format}
-0.05R_{length}-0.05R_{repeat}
\quad \text{(safer mixed reward: multiple dimensions counterbalance each other)}
$$

> **Keep one sentence in mind**
>
> Reward hacking is not the model being "bad." It is the model seriously optimizing the objective you wrote incorrectly. Fixing it is not about scolding the model — it is about fixing the reward, supplementing data, and adding evaluation gates.

Sections 8.1–8.7 covered the main RLHF pipeline: SFT provides the starting policy, the RM provides preference signal, PPO optimizes under the reward, and evaluation confirms real improvement. This section adds two topics that do not fit cleanly into the main narrative: a controlled reward-hacking experiment and an engineering template for the data flywheel.

## Deliberately Creating a Bad Reward

Chapter 3 explained that the reward function defines the objective in the agent's eyes. The same principle applies to LLM RLHF — except the "environment reward" here often comes from an RM, a judge, rule checks, or a mixture of these.

The most common beginner illusion is:

> As long as the reward curve goes up, the model is getting better.

The reward-hacking experiment is designed to shatter this illusion. We deliberately write a reward function with an obvious loophole, making it easy for the model to find a strategy that scores high while producing worse answers. This has three benefits:

1. You can see firsthand how reward diverges from real quality.
2. You learn which monitoring metrics raise early warnings.
3. You understand why real-world RLHF needs manual spot-checks and regression evaluation.

## The Flawed Reward Function

Reward hacking is most easily underestimated because the training curves usually look beautiful. The best way to get started is to deliberately create a loophole: make the reward function favor long answers, list formats, and fixed polite phrases, then observe how the model learns to "pad its word count."

```python
def flawed_reward(prompt: str, response: str) -> float:
    """
    A flawed reward function.
    Core problem: "detailed" is mis-encoded as "the longer the better",
    and fixed formats and polite phrases receive bonus points.
    """
    length_score = len(response) / 100.0

    format_score = 0.0
    if "- " in response or "1." in response:
        format_score += 0.5
    if "**" in response:
        format_score += 0.5

    politeness_score = 0.0
    for phrase in ["I'm happy to help", "Hope this helps", "Please note", "Here are some"]:
        if phrase in response:
            politeness_score += 0.3

    return length_score + format_score + politeness_score
```

This reward function intends to express "answers should be detailed, structured, and polite," but what it actually encodes is:

```text
longer is better
lists are better
bold is better
polite templates are better
```

Once PPO or GRPO discovers this pattern, it will push up answer length, headings, lists, and polite phrases together. The model is not doing anything wrong — it has simply found what your reward function truly rewards.

### Manually Computing the Bad Reward for Two Answers

Given the same prompt:

```text
Explain PPO's KL penalty in one sentence.
```

Answer A:

```text
The KL penalty prevents the new policy from deviating too far from the reference policy, keeping PPO updates under control.
```

Answer B:

```text
I'm happy to help you. Here are some important points:
- **First**: PPO is very important.
- **Second**: KL is also very important.
- **Third**: Hope this helps you.
```

A rough calculation using `flawed_reward`:

| Answer | Length score | Format score | Politeness score | Total | Human impression     |
| ------ | ------------ | ------------ | ---------------- | ----- | -------------------- |
| A      | ~0.32        | 0.0          | 0.0              | 0.32  | Concise and accurate |
| B      | ~0.75        | 1.0          | 0.6              | 2.35  | Verbose and empty    |

The bad reward strongly prefers B. As long as PPO keeps optimizing, the model will produce more answers like B.

## Experimental Setup

Do not start with a large model running full RLHF. This experiment only needs a small model, a small prompt set, and short training steps.

| Item              | Suggestion                                         |
| ----------------- | -------------------------------------------------- |
| Model             | A small SFT model, or a small chat model           |
| Prompt count      | 50 to 200 prompts                                  |
| Generation length | Start with `max_new_tokens=256`                    |
| Reward            | `flawed_reward`                                    |
| Baseline          | SFT raw outputs on the same prompts                |
| Monitoring        | reward, length, repetition rate, manual spot-check |

The goal is not to train a good model, but to observe how the model exploits the reward loophole.

A pseudocode loop:

```python
for step in range(num_steps):
    prompts = sample_prompts(prompt_pool)
    responses = actor.generate(prompts, max_new_tokens=256)

    rewards = [flawed_reward(p, r) for p, r in zip(prompts, responses)]
    kl = compute_kl(actor, reference, prompts, responses)
    total_rewards = [r - beta * k for r, k in zip(rewards, kl)]

    ppo_update(actor, critic, prompts, responses, total_rewards)

    if step % eval_interval == 0:
        log_reward_hacking_metrics(step, prompts, responses, rewards, kl)
```

Even without actually running PPO, you can use different decoding strategies to generate samples and verify whether `flawed_reward` prefers worse answers. Reward function unit tests should always precede large-scale training.

## Observing Three Warning Signals

When running this experiment, do not only log reward. Plot at least three curves simultaneously:

| Metric            | Normal behavior              | Reward hacking signal                                       |
| ----------------- | ---------------------------- | ----------------------------------------------------------- |
| `reward_mean`     | Rises slowly                 | Rises steadily and much faster than manual quality improves |
| `response_length` | Fluctuates within task needs | Grows continuously alongside reward                         |
| `distinct_ngram`  | Remains relatively stable    | Drops noticeably, indicating increasingly templated output  |

Add two more PPO-RLHF-specific metrics:

| Metric           | Why watch it                                                          |
| ---------------- | --------------------------------------------------------------------- |
| `kl_mean`        | Check whether the Actor is rapidly drifting from the reference        |
| `judge_win_rate` | Check whether an external judge or human actually prefers the outputs |

A rough but useful detector:

```python
def reward_hacking_report(rows):
    """
    rows: [{"reward": float, "text": str}, ...]
    """
    import numpy as np
    from collections import Counter

    rewards = np.array([row["reward"] for row in rows])
    lengths = np.array([len(row["text"]) for row in rows])
    length_corr = float(np.corrcoef(rewards, lengths)[0, 1])

    phrases = Counter()
    for row in rows:
        words = row["text"].split()
        phrases.update(" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3)))

    unique_4grams = len(phrases)
    total_4grams = sum(phrases.values())
    distinct_4 = unique_4grams / max(total_4grams, 1)

    return {
        "length_reward_corr": length_corr,
        "distinct_4": distinct_4,
        "top_phrases": phrases.most_common(5),
        "warning": length_corr > 0.7 or distinct_4 < 0.5,
    }
```

This detector cannot prove the model has been hacked, but it can expose the most common length hacks and template hacks. To truly confirm the problem, you still need to spot-check samples with the highest reward, the largest length growth, and the most repeated phrases.

## Reward Curves vs. Sample Quality

A controlled experiment might produce logs like this:

| step | reward | length | distinct-4 | KL   | Manual notes              |
| ---- | ------ | ------ | ---------- | ---- | ------------------------- |
| 0    | 0.8    | 120    | 0.82       | 0.00 | SFT outputs look normal   |
| 50   | 1.4    | 180    | 0.76       | 0.04 | Answers slightly longer   |
| 100  | 2.2    | 310    | 0.61       | 0.09 | Polite phrases increasing |
| 150  | 3.1    | 520    | 0.42       | 0.18 | Clearly templated         |
| 200  | 4.0    | 760    | 0.31       | 0.27 | Heavy repetition in lists |

Looking at reward alone, this is a beautiful curve. Looking at the samples, training has gone wrong.

A typical bad sample might look like this:

```text
I'm happy to help you. Here are some important points:

1. **First**, PPO is a very important method.
2. **Second**, the KL penalty is also a very important method.
3. **Third**, understanding the KL penalty is important for understanding PPO.
4. **Finally**, hope this helps you better understand PPO.

Please note that the above is just a brief overview. Hope this helps.
```

It is polite, has lists, uses bold, and is long — so the bad reward gives it a high score. But it barely explains the substance of the KL penalty.

## Multi-Dimensional Reward Repair

The fix is not simply to "penalize length." Instead, decompose the previously conflated objectives:

```python
def safer_reward(prompt: str, response: str) -> float:
    helpfulness = judge_helpfulness(prompt, response)
    correctness = judge_correctness(prompt, response)
    format_score = validate_required_format(prompt, response)
    repetition_penalty = ngram_repetition_rate(response, n=4)
    length_penalty = max(0, len(response) - target_max_length(prompt)) / 400

    return (
        0.40 * helpfulness
        + 0.35 * correctness
        + 0.15 * format_score
        - 0.05 * repetition_penalty
        - 0.05 * length_penalty
    )
```

This version has several improvements:

| Improvement                           | Effect                                                             |
| ------------------------------------- | ------------------------------------------------------------------ |
| Helpfulness and correctness separated | Prevents conflating "helpful but wrong" with "correct but useless" |
| Format gets a small weight            | Prevents the model from sacrificing content for format             |
| Length is a penalty, not a reward     | Prevents higher scores for longer answers                          |
| Repetition penalized separately       | Prevents template looping                                          |
| Target length depends on the prompt   | Short-answer and long-answer tasks handled separately              |

Then add the KL constraint from PPO-RLHF to keep the policy from drifting too far from the SFT reference while chasing the new reward. Whether the fix works cannot be judged from the new reward alone — you must also check that manual preference, regression benchmarks, length distribution, and repetition rate all improve together.

## Supplementing Rejected Data

If the RM has already learned to prefer long nonsense, simply adding a length penalty may be a superficial fix. A more robust approach is to add the bad samples back into preference data, so the RM explicitly learns that they should score low:

```json
{
  "prompt": "Explain PPO's KL penalty in one sentence.",
  "chosen": "The KL penalty prevents the new policy from deviating too far from the reference policy, keeping PPO updates from overshooting.",
  "rejected": "I'm happy to help you. Here are some important points: PPO is important, KL is important, hope this helps...",
  "tags": ["length_hack", "template_hack"],
  "source": "ppo_badcase"
}
```

This is the basic move of the data flywheel: the model exposes a failure mode, we turn the failure mode into training data, and then we evaluate in regression whether it has been fixed.

## Data Flywheel Engineering Template

RLHF does not end after one training run. A practical data flywheel typically looks like this:

```text
Deploy the model or run offline evaluation
  -> Collect badcases, user feedback, evaluation failures
  -> Cluster by error type
  -> Targeted production of SFT / preference data
  -> Pass through quality gates
  -> Train SFT, RM, or PPO-RLHF
  -> Regression evaluation and manual spot-checks
  -> Deploy again after passing
```

The key is to "supplement data by error type" rather than blindly scaling up data volume. An active learning cycle can be written as:

```python
def active_learning_cycle(model, eval_set, data_producer):
    errors = evaluate_and_collect_errors(model, eval_set)
    clusters = cluster_errors_by_type(errors)

    new_data = []
    for cluster in clusters.top_k(k=3):
        new_data.extend(data_producer.generate(
            task_type=cluster.type,
            difficulty=cluster.difficulty,
            num_samples=1000,
        ))

    cleaned = quality_gate(new_data)
    updated_model = train_on_new_data(model, cleaned)
    report = regression_eval(updated_model, eval_set)
    return updated_model, report
```

The `quality_gate` here should at minimum perform deduplication, evaluation set contamination checks, length filtering, chosen/rejected divergence checks, difficulty stratification, and a small amount of manual spot-checking. The faster the data flywheel spins, the more you need evaluation sets and manual spot-checks as brakes — otherwise the model will quickly overfit to the current evaluation and the current judge's preferences.

## Error Type Clustering

If badcases are stored as a pile of plain text, nobody will look at them. A better approach is to tag each failure sample:

| Tag             | Meaning                                           | Data supplementation direction                    |
| --------------- | ------------------------------------------------- | ------------------------------------------------- |
| `length_hack`   | Answers grow longer but information density drops | Add short precise chosen, verbose rejected        |
| `template_hack` | Fixed boilerplate phrases repeat                  | Add diverse-style chosen, template rejected       |
| `hallucination` | Fabricated facts or citations                     | Add fact-verification data, reject when uncertain |
| `over_refusal`  | Refuses questions it should answer                | Add safety-boundary samples                       |
| `under_refusal` | Fails to refuse high-risk questions               | Add safety refusal preference pairs               |
| `format_fail`   | JSON / code blocks / word count not met           | Add rule rewards and format SFT                   |
| `reasoning_gap` | Reasoning skips steps or conclusion is wrong      | Add process supervision or verifiable problems    |

The goal of clustering is not to produce pretty reports — it is to decide how to produce data in the next round.

## Two Typical Cases

**Reasoning data loop.** For math and code tasks, a verifier can directly judge whether an answer is correct. The model samples multiple answers for the same prompt; the verifier labels them as correct or incorrect. Correct answers become positive examples, incorrect ones become negative examples. Problem types that still fail after evaluation are further clustered, and similar problems are generated in a targeted manner.

```text
Sample N answers
  -> Verifier judges correct / incorrect
  -> Correct and concise as chosen
  -> Incorrect, skip-step, or verbose as rejected
  -> Train RM / DPO / RLVR
  -> Regression on GSM8K, MATH, HumanEval, or custom test sets
```

**Agent trajectory data loop.** Data for agentic RL is not ordinary text — it consists of trajectories from the model interacting with an environment. Successful trajectories can serve as chosen; failed trajectories must be broken down into whether the error was in planning, tool invocation, observation understanding, or the final answer. Only by knowing the failure type can subsequent data supplementation become more than "just generate more."

```text
Agent executes tasks
  -> Collect success / failure trajectories
  -> Successful trajectories as chosen
  -> Failed trajectories tagged by cause
  -> Revise failed trajectories or supplement local training data
  -> Regression on SWE-bench, WebArena, or custom tasks
```

## Minimum Acceptance Criteria

After fixing reward hacking, check at least:

| Metric           | Expectation                                              |
| ---------------- | -------------------------------------------------------- |
| reward           | No longer monotonically driven by length                 |
| length           | Returns to the reasonable range for the task             |
| distinct n-gram  | Not significantly worse than SFT                         |
| Human preference | Post-fix answers are genuinely more usable               |
| Regression set   | Existing capabilities have not regressed                 |
| Stress cases     | Long nonsense and template nonsense no longer score high |

If the new reward drops but human quality rises, do not panic. That just means the old reward was unreliable to begin with. In RLHF debugging, fixing a bad reward often makes curves look "ugly" in the short term, but the model's true quality is healthier.

## Section Summary

Reward hacking requires controlled experiments to practice diagnosis; the data flywheel requires quality gates to prevent self-deception. The truly hard part of RLHF is not just the algorithms — it is making reward, data, and evaluation counterbalance each other.

This chapter is now fully closed: a base model is not an assistant. SFT gives it behavioral starting point, the RM gives it preference direction, PPO lets it practice under the reward, and evaluation plus the data flywheel prevent it from learning the wrong things. The next chapter departs from this classic RLHF pipeline to explain why modern methods aim to simplify the RM, the Critic, or human preferences themselves.

## Exercises

1. Modify `flawed_reward` to deliberately favor answers that "contain specialized terminology," and design 3 stress cases.
2. Tag a reward-hacking sample: is it a length, template, format, or semantic hack?
3. Design one round of a data flywheel: from badcase collection to retraining, what quality gates would you set up?
