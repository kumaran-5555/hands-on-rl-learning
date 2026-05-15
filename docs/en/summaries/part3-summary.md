---
title: Part III Summary
---

# Part 3: Reinforcement Learning for LLMs - Knowledge Summary

## What Did We Learn in This Part?

These three chapters are the main turning point of the book. We took the RL theory built up in the first six chapters and applied it to real scenarios: LLM alignment and agent training. After finishing this part, you should understand:

- **The RLHF engineering pipeline**: the three stages SFT -> RM -> RL; mixed reward functions such as $R = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}}$; defenses against reward hacking; and RLAIF, which uses AI instead of humans for labeling.
- **DPO implicit reward**: $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$. The reward function is hidden in the policy probability ratio; no separate reward model is required.
- **DPO loss**: enforce higher implicit reward for preferred answers than for rejected ones, turning an RL problem into a classification problem. Two models are enough; no Critic and no reward model are needed.
- **The DPO family**: KTO needs only good/bad labels (no paired comparisons), SimPO removes the reference model, and ORPO merges SFT with alignment.
- **GRPO within-group normalization**: $A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$. For the same prompt, generate $k$ answers and use a within-group z-score in place of a Critic. Only one model is needed.
- **DAPO's four improvements**: Clip-Higher (more headroom for low-probability actions), token-level loss (compute gradients per token), dynamic sampling (filter prompts already mastered), and overlong filtering.
- **RLVR**: in objective tasks such as math and code, replace human-labeled reward models with rule-based verifiers (answer matching, unit tests). DeepSeek-R1-Zero shows that pure RLVR training can induce emergent reasoning ability.
- **Agentic RL**: ORM (reward only at the end) vs PRM (reward at each step), and the tool-use training recipe "SFT teaches format + RL teaches strategy."

Now let us review the chapters.

## Chapter 7: The Full RLHF Pipeline - From Theory to Engineering

### Reward Function Design

In real RLHF systems, the reward function is far more than "a single reward model score." It is usually a mixture:

$$R_{\text{total}} = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}} + \gamma R_{\text{correctness}}$$

Reward granularity also matters: sequence-level (one score per response), step-level (score each step, i.e., PRM), and token-level (reward signals per token).

### Reward Hacking: When the Model Learns to Game the Score

Classic reward hacking patterns include length inflation, repetitive score farming, and format cheating. Defenses include analyzing correlations between length and reward, counting high-frequency phrases, and running periodic human evaluations. A KL penalty term such as $-\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ is one important safety layer.

### RLAIF: Using AI Instead of Humans

RLAIF replaces humans with stronger models for preference labeling. Constitutional AI lets a model critique and revise its own outputs, forming a data flywheel: deploy model -> collect user feedback -> identify weak spots -> use AI to construct preference data -> retrain.

## Chapter 8: Alignment and Reasoning Reinforcement (DPO + GRPO + RLVR)

### From RLHF to DPO: A Key Mathematical Equivalence

Traditional RLHF is complex: train a reward model from human preferences, then train a language model with PPO to maximize reward while adding a KL penalty to prevent drift. This requires four models running together and leads to high GPU memory usage and engineering complexity.

DPO's breakthrough is a clean mathematical observation: the RL objective with a KL constraint,

$$\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

has a closed-form optimum:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

Taking logs and rearranging, the reward can be expressed by a probability ratio:

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

Since $Z(x)$ depends only on the prompt $x$ (not the answer $y$), it cancels in the Bradley-Terry preference model $P(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$. This yields the DPO loss:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

### GRPO: Replacing the Critic with Within-Group Statistics

PPO needs a Critic network to estimate advantages $A(s,a)$. In LLM settings, the Critic itself is a large model, which is expensive in GPU memory. GRPO (Group Relative Policy Optimization) proposes a clever alternative:

For the same prompt $x$, generate $k$ answers $y_1, y_2, \ldots, y_k$, score each with a reward function to get $r_1, r_2, \ldots, r_k$, then compute a within-group normalized advantage:

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$$

This matches PPO's logic of "how much better than baseline" (PPO uses $A=Q-V$ from a Critic), but GRPO uses within-group statistics instead of an explicit Critic.

### DAPO: Four Improvements That Make GRPO Stronger

**Clip-Higher** decouples upper and lower clipping ranges, giving exploration more room. **Token-level loss** sums gradients per token, locating errors more precisely. **Dynamic sampling** filters prompts already mastered, maintaining a difficulty curriculum. **Overlong filtering** removes samples that exceed length limits.

### RLVR: Rule-Based Verification Instead of Human Labels

GRPO/DAPO no longer rely on a reward model. As long as something can produce a score, training can proceed. For objective tasks like math reasoning and code generation, that scorer can be a rule-based verifier: for math, match the final answer; for code, run unit tests. Experiments such as DeepSeek-R1-Zero suggest that a base model can exhibit emergent chain-of-thought reasoning after RLVR-only training, even without any SFT.

## Chapter 9: Agentic RL - Teaching Models to Use Tools

### Multi-Turn Interaction and Credit Assignment

Classic RL alignment is typically single-turn, but real agents act over multiple turns. **ORM** (Outcome Reward Model) gives reward only at the end, which is sparse. **PRM** (Process Reward Model) scores each step, providing dense signals but at higher labeling cost.

### RL Training for Tool Use

Web agents and code agents are typical agentic RL settings. A common recipe is "SFT teaches format + RL teaches strategy": use supervised data to teach how to call tools, then use RL to teach when and how to use tools to complete tasks.

## Summary

Part 3 shows a clear evolution:

RLHF needs 4 models -> DPO cuts it down to 2 via implicit reward -> GRPO cuts it down to 1 via within-group normalization -> RLVR removes the reward model entirely via rule-based verification.

Each step replaces complex components with simpler mechanisms while remaining mathematically equivalent, or even stronger.

At the same time, RL expands from "aligning human preferences" to "eliciting reasoning" and "training agents." Moving from single-turn dialogue to multi-turn tool interactions, RL is playing an increasingly central role in LLM post-training.

> **Next stop**: [Part 4: Frontier and Advanced Topics](/chapter11_vlm_rl/intro)
