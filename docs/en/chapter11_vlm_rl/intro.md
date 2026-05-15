---
title: 11. VLM Reinforcement Learning
---

# VLM Reinforcement Learning

In earlier chapters, we pushed RL from classic control to LLM post-training:

- DQN learns from pixels in Atari,
- PPO stabilizes policy updates,
- DPO/GRPO optimize language models with preference signals or verifiable rewards.

Most of those settings share a simplifying assumption: there is only one input modality (state vectors, pixels, or text tokens).

The real world is not text-only. You see images, screenshots, charts, videos, and 3D scenes. Before you can reason and act, you must first **understand visual evidence**. Vision-language models (VLMs) bring images and language into a single model. RL then asks a harder question:

Can we use outcome feedback to make the model not only describe images, but _see more accurately, reason more reliably, and answer more truthfully_?

![VISTA-Gym Overview](../../chapter11_vlm_rl/images/ref-vista-gym-overview.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: VISTA-Gym (and VISTA-R1) illustrates a typical VLM-RL loop: visual QA + tool use + trajectory reward + policy updates, pushing beyond "look-and-answer" toward "look, verify with tools, and improve by feedback". Source: the VISTA-Gym / VISTA-R1 blog.</em>
</div>

Moving RL from text to multimodal models is not "just add image tokens." Once you train seriously, you run into a set of problems that do not appear in text-only RL:

1. **Who is responsible for an error?** If the answer is wrong, was the vision encoder wrong, or was the language reasoning wrong?
2. **Should the vision encoder be updated by RL?** Update too aggressively and you can degrade vision (the model "goes blind"); freeze it completely and you cannot improve visual ability.
3. **Will the model pretend it saw the image?** If guessing can get reward, RL can reinforce visual hallucinations.
4. **How does vision connect to action?** In driving, robotics, and GUI agents, visual outputs affect real decisions. Safety and latency become training constraints.

This chapter opens these issues in a progressive way: first a minimal GRPO experiment to build intuition, then the special challenges of VLM RL, then frameworks that connect experiments to real applications, and finally RL post-training for visual generation models.

::: tip Prerequisites

- [GRPO](../chapter09_grpo_rlvr/grpo-practice-and-mechanism): group-based optimization without a critic
- [Reward model design](../chapter08_rlhf/reward-function-design): rules vs model rewards, hacking risks
- [PPO-RLHF loop](../chapter08_rlhf/ppo-rlhf-loop): KL penalty, clipping, reference model
  :::

## VLM RL vs Text-Only RL

In text-only RL, inputs and outputs are tokens. If an answer is bad, we usually ask one question: did the generated tokens match the reward target?

In VLM RL, there is a visual pipeline:

image -> vision encoder -> visual tokens -> multimodal fusion -> language reasoning -> output.

So training becomes "see correctly, then reason correctly." A single scalar reward does not automatically tell you where the failure came from. This is the core credit-assignment problem in multimodal RL.

## Learning Path

This chapter is organized as: run something minimal -> see the new problems -> understand systems -> extend to generation.

| Section                                              | Question it answers                                                                       |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [11.1 Hands-On: GRPO for a VLM](./vlm-grpo-hands-on) | How do we train a VLM to "look then reason" under verifiable rewards?                     |
| [11.2 Challenges](./vlm-challenges)                  | How do we assign reward across vision vs language? How do we reduce visual hallucination? |
| [11.3 Frameworks](./vlm-frameworks)                  | What systems bridge experiments to applications (tools, environments, self-play)?         |
| [11.4 Visual Generation RL](./visual-generation-rl)  | How does RL apply to diffusion/video generation, and what does "policy" mean there?       |
| [11.5 Hands-On: EasyR1 GeoQA](./easyr1-geoqa)        | How do we run an industrial-style VLM GRPO training loop on a real dataset?               |

## Learning Goals

After this chapter, you should be able to:

- map VLM RL back to the same RL primitives you already know (policy, trajectory, reward, KL constraints),
- explain why multimodality changes reward design and credit assignment,
- identify the typical failure modes (visual hallucination, encoder collapse, mis-grounding),
- and evaluate whether a VLM is truly using visual evidence rather than language priors.
