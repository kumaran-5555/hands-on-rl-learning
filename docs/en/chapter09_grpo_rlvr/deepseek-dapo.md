---
title: 9.3 The R1-Zero Paradigm
---

# 9.4 DeepSeek-R1 and DAPO: A New Paradigm for Pure RL Training

In the previous section, we understood GRPO's group-wise normalization mechanism: it replaces the Critic with the mean and standard deviation inside a sampled group, neatly removing the need for a full extra model. In this section, we zoom out and look at two of the most exciting RL breakthroughs of 2025: DeepSeek-R1-Zero showed that pure RL can work without SFT, and DAPO further improved the engineering efficiency of GRPO.

## DeepSeek-R1-Zero

Before DeepSeek-R1, alignment for large models had an almost unquestioned rule: **a base model must first go through SFT, learning how to speak in an instruction-following format, before it can do RL**. The reason sounded straightforward. If you run RL directly on a base model that has never seen SFT, its outputs may mix languages, ignore format, and look like unusable noise, so RL training seems to have no stable starting point.

In January 2025, the DeepSeek team published a paper that shook the AI community. They found that **in domains with clear objective rule-based rewards, such as matching a math answer or passing code compilation, SFT is not necessary for cold start**. It is feasible to train a base model directly with large-scale GRPO. This discovery produced DeepSeek-R1-Zero: a pure RL model trained with no SFT at all.

Why is this discovery so important? In the traditional view of RLHF, SFT was considered a prerequisite for RL. The logic went like this: a base model only knows how to "continue text"; it does not know how to "answer a question." If you apply RL directly to a base model, its outputs may not even look like reasonable answers, and a reward model has nothing meaningful to score. Therefore, one must first teach the model the basic conversational format with SFT, and only then optimize answer quality with RL.

DeepSeek-R1-Zero broke that assumption. When the reward is rule-verifiable, meaning correct answers get a score and incorrect answers do not, the model does not first need to learn "how to answer." It only needs to find output patterns that receive high scores through many trials. Even if the initial output is messy, as long as the model occasionally gives a correct answer and receives reward, RL will reinforce that path. After enough training steps, the model can discover a clear reasoning format by itself.

### Emergence and the Aha Moment

The most surprising finding during R1-Zero training was **emergent behavior**. Without any human demonstrations, the model independently developed these abilities:

- **Long chain-of-thought**: the model gradually moved from "giving the answer directly" to "analyzing the problem, writing formulas, and calculating step by step," without anyone teaching it to do so.
- **Self-reflection**: when an answer was wrong, the model learned to go back, inspect its reasoning process, find mistakes, and correct them.
- **Strategy switching**: for different kinds of problems, the model automatically chose different solving strategies.

These abilities were not manually designed. They were "discovered" by the model itself in order to obtain higher rule-based rewards. The DeepSeek team called this the **"Aha Moment"**: at a certain stage of training, the model suddenly seemed to "get it" and began showing reasoning abilities it had never displayed before.

More concretely, DeepSeek observed the following emergence timeline:

- **Early training** (0-100 steps): outputs are short and messy, often jumping directly to an incorrect numeric answer.
- **Middle training** (100-500 steps): the model starts to show simple calculation steps, but often makes mistakes halfway through.
- **Aha moment** (around 500-1000 steps): the model suddenly starts checking its own calculations, with behaviors such as "wait, let me recalculate."
- **Late training** (1000+ steps): the model forms a stable three-step reasoning pattern of "analysis -> calculation -> verification."

This emergent behavior raises a deeper scientific question: **where does the model's reasoning ability come from?** The likely answer is that pretraining already gives the model the raw materials of reasoning, including logic, mathematics, and language knowledge. RL training merely organizes these raw materials into usable problem-solving strategies. This also explains why 1-shot RLVR can work: the model already has reasoning ability, and RL only activates it.

### Open-Source Reproduction: SimpleRL-reason

At this point, a learner naturally asks: is the R1-Zero phenomenon something that only DeepSeek's large-scale models and training system can observe? If we switch to open-source base models, smaller data, and public training frameworks, can we still see "reasoning activated by RL alone, without SFT"?

[SimpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) and the follow-up paper [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) answer exactly this question. They do not propose a brand-new policy optimization algorithm. Instead, they shrink R1-Zero's key hypothesis into an easier experimental setting: start from a base model, do no SFT, train no reward model, and use only verifiable math problems plus rule-based rewards for reinforcement learning.

First look at the training loop:

```text
Math problem x
-> base model generates solution process y
-> extract the final answer from y
-> compare with the reference answer
-> obtain a 0/1 rule-based reward
-> update the model with PPO / RL
```

This has the same spirit as R1-Zero: the reward does not come from human preference or a reward model, but from objective answer verification. The early SimpleRL-reason implementation was based on OpenRLHF, using Ray for distributed scheduling and vLLM for efficient sampling. The SimpleRL-Zoo paper further extended the observations to multiple model families and sizes, including Llama, Mistral, DeepSeek-Math, Qwen2.5-Math, and Qwen2.5 models at different scales.

The teaching value of this case is that it decomposes "whether pure RL is feasible" into several more concrete questions:

| Question                                                | What SimpleRL-Zoo lets us observe                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Is the base model important enough to matter?           | Different model families have very different zero-RL starting points and ceilings.                                  |
| Is more data always better?                             | A small amount of high-quality, verifiable math data can still produce a clear training signal.                     |
| Does a longer chain of thought mean stronger reasoning? | Response length, accuracy, and self-verification behavior must be examined separately.                              |
| Is a "simple recipe" really simple?                     | The reward form is simple, but distributed sampling, length control, and evaluation are still engineering-critical. |

Therefore, SimpleRL-reason is best understood as an open-source reference experiment for R1-Zero, not as a separate new chapter. It tells us that zero RL is not just a slogan, and it is not something that only happens inside closed-source large systems. As long as the base model already has latent ability and the task can provide stable verifiable rewards, RL has a chance to organize those latent abilities into more reliable problem-solving strategies.

The boundary should also be stated clearly. The "simple" in SimpleRL-reason mainly refers to the simplicity of the training signal and recipe; it does not mean the hardware cost can be ignored. Public reproduction experiments still rely on multi-GPU training, parallel rollout, and standard evaluation sets. In other words, it shows that the R1-Zero idea can be open-sourced and experimentally studied, not that writing an answer-matching function is enough to reproduce full reasoning emergence on a personal computer.

### R1-Zero's Limits and Engineering Compromises

Although R1-Zero proved the feasibility of pure RL, it had an obvious weakness: **poor language quality**. Because it had not gone through SFT, its answers often mixed languages, used messy formatting, and were hard to read. Its reasoning ability was strong, but the answer could look like a brilliant student who cannot express ideas clearly.

Therefore, the final released DeepSeek-R1 used a multi-stage engineering compromise:

1. **Cold start**: a small amount of high-quality SFT data teaches the model a basic output format.
2. **Large-scale GRPO**: strengthens reasoning ability. This is the core stage.
3. **Rejection sampling**: filters high-quality data from the model after GRPO training.
4. **SFT fine-tuning**: further improves format and language quality using the filtered data.
5. **Second RL stage**: combines an RM and GRPO for final alignment training.

## DAPO

GRPO has already shown that RL can work without a Critic, but it still has several engineering pain points. DAPO, short for Decoupled Clip and Dynamic Sampling Policy Optimization, addresses these problems directly and was accepted as a NeurIPS 2025 poster.

### DAPO's Four Improvements

| Improvement                 | GRPO's problem                                                                          | DAPO's solution                                                             | Effect                          |
| --------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------- |
| **Clip-Higher**             | Symmetric upper/lower clipping over-suppresses low-probability actions.                 | Decouple clipping ranges and give low-probability actions more upward room. | Better exploration              |
| **Dynamic sampling**        | All prompts participate in training, wasting compute.                                   | Filter out prompts the model already solves.                                | 2-3x better training efficiency |
| **Token-level loss**        | Sequence-level reward normalization ignores differences between tokens.                 | Token-level policy gradients provide finer credit assignment.               | Better long-sequence training   |
| **Overlong Reward Shaping** | Overlong answers are directly truncated and penalized, causing discontinuous gradients. | Use a smooth length penalty function.                                       | More stable training            |

The intuition behind **Clip-Higher** is this: GRPO clips the policy ratio symmetrically, for example to $[0.8, 1.2]$. This is reasonable for high-probability actions that the model is already fairly confident about. But for actions whose current probability is low, say 0.01, yet still promising, the lower bound of 0.8 means they can be pushed down to 0.008, almost completely suppressing them. DAPO decouples the upper and lower clipping ranges, giving low-probability actions more room to rise.

**Dynamic sampling** solves the "graduation problem." In the previous section, we observed that late in training, many problems have near-zero within-group variance because the model already solves them. These problems provide no gradient signal. DAPO directly filters out these "graduated problems" and keeps only prompts with useful gradients. On the AIME 2024 math competition, DAPO reached a score of 50 using **half the training steps** of DeepSeek-R1.

**Token-level loss** addresses another blind spot in GRPO. Standard GRPO normalizes the entire sequence: an answer is either fully reinforced if it is correct, or fully suppressed if it is wrong. In reality, the first 80% of reasoning steps in an incorrect answer may be right, with only the final calculation wrong. Token-level loss lets GRPO distinguish "which tokens are good and which are bad," enabling finer credit assignment. This directly corresponds to the credit assignment problem discussed in [Chapter 7 on GAE](../chapter07_ppo/gae-reward-model): in long sequences, we need to know each token's contribution to the final outcome.

**Overlong Reward Shaping** solves a common engineering problem in GRPO training: response length can get out of control. The model may learn that "writing more is better" because longer answers are more likely to contain correct reasoning, producing verbose 2000+ token responses. The original GRPO approach sets a maximum length, truncates anything beyond it, and assigns a penalty. But truncation is a hard boundary: a 499-token answer is fine, while a 501-token answer is penalized, creating a discontinuous gradient signal. DAPO replaces hard truncation with a smooth length penalty function, allowing the model to learn length control naturally.

```python
# ==========================================
# DAPO dynamic sampling sketch
# ==========================================
def dynamic_sampling(prompts, model, reward_fn, threshold=0.95):
    """
    Filter out prompts the model has already mastered.
    """
    useful_prompts = []

    for prompt in prompts:
        # Sample each prompt multiple times and compute accuracy.
        correct_count = 0
        num_samples = 8
        for _ in range(num_samples):
            response = model.generate(prompt)
            reward = reward_fn(prompt, response)
            if reward >= 1.0:  # Correct answer.
                correct_count += 1

        accuracy = correct_count / num_samples
        # Keep only prompts whose accuracy is below the threshold.
        if accuracy < threshold:
            useful_prompts.append(prompt)

    print(f"Before filtering: {len(prompts)} problems")
    print(f"After filtering: {len(useful_prompts)} problems")
    print(f"Filtered out: {len(prompts) - len(useful_prompts)} mastered problems")
    return useful_prompts
```

DeepSeek-R1-Zero and DAPO show the enormous potential of pure RL training: no SFT, no Critic, as long as the reward signal is clear enough. But this raises a prerequisite question: **where does the reward come from?** In the next section, we turn to RLVR and see how verifiable rewards can fully replace the reward model.
