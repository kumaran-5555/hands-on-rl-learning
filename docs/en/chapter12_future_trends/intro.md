---
title: 12. Future Trends
---

# Future Trends

From Chapter 1 (CartPole) to Chapter 9 (GRPO), we walked through the core arc of modern reinforcement learning:

- Q-learning and DQN: learning from trial-and-error,
- policy gradients: optimizing behavior directly,
- PPO: stable post-training for large models,
- GRPO: using verifiable rewards to drive reasoning,
- Agentic RL: moving from single-turn answers to multi-turn tool-using interaction.

But the RL story is not finished. In 2025-2026, several shifts have become increasingly clear:

1. RL is moving into the physical world (embodied intelligence).
2. RL is not only "training-time optimization" (it increasingly interacts with test-time search and planning).
3. RL is no longer only single-agent (multi-agent collaboration and self-play are re-emerging as central drivers).

This chapter does not attempt to cover every frontier direction. That is not realistic. Instead, we pick representative themes that connect directly back to the concepts you already learned in earlier chapters. The goal is to help you recognize recurring structure: the same foundations reappear under new labels.

| Section                                                   | Core question                                                                     |
| --------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [Embodied Intelligence](./embodied-intelligence/)         | How does RL enter the physical world (perception, action, safety)?                |
| [Model-Based RL](./embodied-intelligence/model-based-rl/) | Can a world model reduce real-world trial-and-error via planning and imagination? |
| [Self-Play](./self-play-outlook/)                         | Can models improve beyond human data by competing with themselves?                |
| [Multi-Agent RL for LLMs](./llm-multi-agent-rl/)          | How do role-specialized agents learn to collaborate and coordinate?               |
| [Offline RL](./offline-rl/)                               | If you cannot explore online, how do you learn from historical data safely?       |
| [Scaling Trends](./rl-scaling-outlook)                    | Where is the ceiling: training-time scaling, test-time scaling, process rewards?  |

We begin with the first step of RL entering the physical world:

[Embodied Intelligence](./embodied-intelligence/).
