<div align="center">
  <img src="docs/public/readme/logo.png" alt="Hands-On Modern RL Logo" width="500" />
  <h1>Hands-On Modern RL</h1>
  <p><strong>A hands-on modern reinforcement learning course</strong></p>
  <p><em>A practice-first guide to modern RL, from classic control to LLM post-training, RLVR, and multimodal agents.</em></p>

  <p>
    <a href="https://walkinglabs.github.io/hands-on-modern-rl/"><img src="https://img.shields.io/badge/Course-Online-2563eb?style=flat-square" alt="Online Course" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-111827?style=flat-square" alt="CC BY-NC-SA 4.0 License" /></a>
    <img src="https://img.shields.io/badge/Node-%3E%3D18-16a34a?style=flat-square" alt="Node >= 18" />
    <img src="https://img.shields.io/badge/Docs-VitePress-646cff?style=flat-square" alt="VitePress" />
  </p>

  <p>
    <a href="README.md">Chinese</a> |
    <a href="README.en.md">English</a>
  </p>

  <p>
    <a href="#course-preview">Course Preview</a> |
    <a href="#overview">Overview</a> |
    <a href="#news">News</a> |
    <a href="#course-outline">Course Outline</a> |
    <a href="#experiment-code">Experiment Code</a> |
    <a href="#quick-start">Quick Start</a> |
    <a href="#contributing">Contributing</a>
  </p>
</div>

## Course Preview

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-learning-path.png" alt="Course learning map screenshot" width="100%" />
      <br />
      <strong>A clear learning map</strong>
      <br />
      <sub>From the preface and foundations to frontier topics, the chapter tree and page outline help you navigate quickly.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-code-focus.png" alt="PPO code focus screenshot" width="100%" />
      <br />
      <strong>Line-by-line code focus</strong>
      <br />
      <sub>Key PPO, DPO, and GRPO implementations include code maps that connect formulas to readable code.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-training-metrics.png" alt="CartPole training metrics screenshot" width="100%" />
      <br />
      <strong>Training metric visualization</strong>
      <br />
      <sub>Real curves, metric explanations, and failure signals sit together so you can debug while running experiments.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-rlhf-pipeline.png" alt="RLHF pipeline screenshot" width="100%" />
      <br />
      <strong>LLM post-training pipelines</strong>
      <br />
      <sub>RLHF, DPO, GRPO, RLVR, and related topics are tied together through flows, artifacts, and cases.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-agentic-rl.png" alt="Agentic RL project screenshot" width="100%" />
      <br />
      <strong>Project-oriented Agentic RL</strong>
      <br />
      <sub>Tool use, trajectory synthesis, evaluation, and multi-tool code agents turn into full engineering exercises.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-vlm-rl.png" alt="VLM reinforcement learning screenshot" width="100%" />
      <br />
      <strong>Multimodal and frontier directions</strong>
      <br />
      <sub>VLM reinforcement learning, visual generation RL, embodied intelligence, and future trends extend into frontier systems.</sub>
    </td>
  </tr>
</table>

---

> [!NOTE]
> We hope this open course gives more learners the courage to climb toward the frontier of intelligence and solve more of the hard problems on the path to AGI.
>
> The course is evolving quickly. We recommend focusing on chapters that are not marked as under construction; chapters still in progress may contain mistakes, and corrections or suggestions are welcome.

> **Help Wanted**
>
> Because compute resources are limited, we are seeking GPU support. If you can help with GPU access, please contact physicoada@gmail.com.

## Contents

- [Course Preview](#course-preview)
- [Overview](#overview)
  - [Design Principles](#design-principles)
  - [Audience](#audience)
  - [Learning Goals](#learning-goals)
  - [Current Status](#current-status)
- [News](#news)
- [Roadmap](#roadmap)
- [Course Outline](#course-outline)
  - [Preface](#preface)
  - [Part 1: Foundations by Practice](#part-1-foundations-by-practice)
  - [Part 2: Core Theory and Methods](#part-2-core-theory-and-methods)
  - [Part 3: LLM-era RL](#part-3-llm-era-rl)
  - [Part 4: Frontier and Advanced Systems](#part-4-frontier-and-advanced-systems)
  - [Appendices](#appendices)
- [Experiment Code](#experiment-code)
- [Recommended Learning Path](#recommended-learning-path)
- [Quick Start](#quick-start)
  - [Read Online](#read-online)
  - [Run the Documentation Site Locally](#run-the-documentation-site-locally)
  - [Verify the Site](#verify-the-site)
  - [Run Course Code](#run-course-code)
- [Repository Structure](#repository-structure)
- [Development Commands](#development-commands)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

**Hands-On Modern RL** is an open course for learning modern reinforcement learning through practice. Instead of the usual "formula first, black-box API later" route, this course takes a **practice-first** path: learners begin with runnable code and observable training behavior, then use those concrete traces to understand states, value functions, policy gradients, reward modeling, credit assignment, and the rest of the mathematical structure behind RL.

The course spans classic control and connects directly to current AI frontiers, including large language model (LLM) post-training, preference alignment with DPO and GRPO, reinforcement learning with verifiable rewards (RLVR), multi-turn tool-use agents, Agentic RL, and vision-language model (VLM) reinforcement learning.

The goal is to provide a solid ladder: from solving CartPole for the first time to building modern post-training and agent systems.

### Design Principles

The course is organized around these engineering and teaching principles:

1. **Practice before formalism.** Each major topic starts from experiments, metrics, failure cases, or implementation details, then introduces the mathematical abstraction.
2. **Theory explains behavior.** MDPs, Bellman equations, policy gradients, GAE, PPO clipping, DPO objectives, and GRPO-style group advantages are introduced as tools for explaining what the code does.
3. **Modern RL goes beyond classic RL.** The course covers classic control and deep RL, then moves into RLHF, preference optimization, RLVR, VLM reinforcement learning, and multi-turn agent training.
4. **Debugging is first-class.** Training collapse, reward hacking, KL drift, entropy decay, OOM failures, and evaluation blind spots are treated as core material.
5. **Readable systems beat black boxes.** Examples favor explicit implementations, inspectable metrics, and clear experiment boundaries so learners can modify and extend them.

### Audience

This course is for learners who want to understand reinforcement learning by building and inspecting working systems.

It is especially useful for:

- Machine learning engineers moving from supervised learning into RL.
- Researchers and students preparing to read modern RL and alignment papers.
- LLM practitioners interested in RLHF, DPO, GRPO, RLVR, and post-training systems.
- Builders of tool-use agents, web agents, code agents, and evaluation pipelines.
- Self-learners who prefer code, experiments, and visual intuition before dense derivations.

Recommended background:

- Python programming experience.
- Basic PyTorch familiarity.
- Introductory linear algebra, probability, and calculus for machine learning.
- Ability to read papers and trace open-source training scripts.

The course includes math review appendices, so full mathematical fluency is not required on day one.

### Learning Goals

After completing the course, learners should be able to:

- Implement and explain the core RL loop: environment interaction, trajectory collection, reward feedback, policy updates, and evaluation.
- Connect MDPs, value functions, Bellman equations, TD learning, policy gradients, and advantage estimation to concrete training behavior.
- Read and modify implementations of DQN, REINFORCE, Actor-Critic, PPO, DPO, GRPO, and related methods.
- Reason about LLM post-training pipelines, including SFT, reward modeling, PPO-style RLHF, DPO-family methods, and RLVR training.
- Understand multi-turn interaction and credit assignment, and build tool-use, trajectory-synthesis, and Agentic RL systems.
- Extend reinforcement learning ideas to VLMs, embodied intelligence, multi-agent self-play, and other frontier areas.
- Diagnose common RL failure modes and design reasonable algorithms, engineering evaluations, and debugging workflows for new RL problems.

### Current Status

This repository is an active courseware project. Content is being expanded chapter by chapter, with emphasis on correctness, runnable examples, and a stable learning path.

- Course site: [walkinglabs.github.io/hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
- Source content: [`docs/`](docs/)
- Runnable examples: [`code/`](code/)
- Local verification: `npm run verify`
- License: [CC BY-NC-SA 4.0](LICENSE)

Issues and pull requests are welcome for typo fixes, conceptual corrections, reproducibility improvements, references, and focused course extensions.

## News

> **Note:** This course was created with AI assistance and has not yet been fully reviewed. It may contain factual mistakes or code that does not run as expected. Issues and pull requests are very welcome.

- **[2026-05-02]** Initial browsable open-source release for testing and feedback.

## Roadmap

The course is under active development. Planned milestones:

- [x] **2026-05-02:** Initial open-source browsable release for community testing and feedback.
- [ ] **2026-05-10:** Publish a first stable minor version, fix early typos, and stabilize Part 1 and Part 2 content and code.
- [ ] **Late May 2026:** Improve reproducible LLM RL experiments and add a full RLVR hands-on module with evaluation.
- [ ] **Early June 2026:** Deliver Agentic RL projects step by step, from single-tool use to complex Deep Research trajectory synthesis.
- [ ] **Late June 2026:** Add Unity-based embodied RL environments and trainable project examples.
- [ ] **July 2026 and later:** Expand multimodal frontier content with full VLM RL or Diffusion RL hands-on cases.

## Course Outline

The course is divided into four parts plus appendices. The online site includes full text, diagrams, code references, and chapter navigation.

### Preface

| Topic                                                                            | Description                                                        |
| :------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| [Course Guide](docs/preface/intro.md)                                            | Course positioning, learning path, and how to use the materials.   |
| [A Brief History of Reinforcement Learning](docs/preface/brief-history/index.md) | From trial-and-error learning to AlphaGo, RLHF, and LLM alignment. |
| [Environment Setup](docs/preface/env-setup.md)                                   | Installation and dependency setup for the course.                  |

### Part 1: Foundations by Practice

| Chapter | Topic                                                                           | Core Question                                                                                               |
| :------ | :------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------- |
| 01      | [CartPole](docs/chapter01_cartpole/intro.md)                                    | In a real environment, what are states, actions, rewards, policies, values, entropy, and training curves?   |
| 1.1     | [States, Actions, Rewards, and Policies](docs/chapter01_cartpole/principles.md) | What basic objects make up an RL problem?                                                                   |
| 1.2     | [Reward, Entropy, Value Loss, and KL](docs/chapter01_cartpole/metrics.md)       | What do the key training-curve metrics tell us?                                                             |
| 02      | [DPO Preference Fine-tuning](docs/chapter02_dpo/intro.md)                       | How does preference optimization change model behavior, and what do loss, reward margin, and accuracy mean? |
| 2.1     | [Post-Training Pipeline and DPO Derivation](docs/chapter02_dpo/principles.md)   | How does DPO derive a training objective from preference data and a reference model?                        |
| 2.2     | [Loss, Reward Margin, and Accuracy](docs/chapter02_dpo/metrics.md)              | How should DPO training metrics be interpreted?                                                             |
| Summary | [Part 1 Summary](docs/summaries/part1-summary.md)                               | What intuition should be in place before formal theory?                                                     |

### Part 2: Core Theory and Methods

| Chapter | Topic                                                                                          | Core Question                                                                                                |
| :------ | :--------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| 03      | [MDPs and Value Functions](docs/chapter03_mdp/intro.md)                                        | How do bandits, MDPs, value functions, Bellman equations, and TD error formalize sequential decision-making? |
| 3.1     | [Two-Armed Bandit: The Smallest RL Problem](docs/chapter03_mdp/bandit.md)                      | How does the simplest trial-and-error problem show exploration and exploitation?                             |
| 3.2     | [MDP: The Formal Framework for RL](docs/chapter03_mdp/mdp.md)                                  | How do states, actions, transitions, rewards, and discounting define a sequential decision model?            |
| 3.3     | [V(s) and the Bellman Equation](docs/chapter03_mdp/value-bellman.md)                           | How can a value function recursively evaluate a situation?                                                   |
| 3.4     | [DP, MC, and TD](docs/chapter03_mdp/dp-mc-td.md)                                               | How do dynamic programming, Monte Carlo, and temporal-difference learning estimate value?                    |
| 3.5     | [Q(s, a)](docs/chapter03_mdp/value-q.md)                                                       | How does action value turn "is this state good?" into "which action should I choose?"                        |
| 3.6     | [Policy Objective J(theta)](docs/chapter03_mdp/policy-objective.md)                            | When directly optimizing a policy, what exactly does the objective maximize?                                 |
| 3.7     | [Where Algorithm Data Comes From](docs/chapter03_mdp/algorithm-taxonomy.md)                    | How do on-policy, off-policy, and data sources affect algorithm design?                                      |
| 3.8     | [Reward Shaping](docs/chapter03_mdp/reward-design.md)                                          | How can reward functions guide learning, and how can they be misused?                                        |
| 3.9     | [Chapter Panorama](docs/chapter03_mdp/panorama.md)                                             | How do the MDP chapter concepts connect into an algorithm map?                                               |
| 04      | [Q-Learning and DQN](docs/chapter04_dqn/intro.md)                                              | Why are replay buffers, target networks, CNN encoders, and DQN variants important?                           |
| 4.1     | [Hands-On: Q-Learning and GridWorld](docs/chapter04_dqn/q-learning.md)                         | How can we update values by hand from tabular Q-learning?                                                    |
| 4.2     | [From Tabular Q to DQN](docs/chapter04_dqn/from-q-to-dqn.md)                                   | How do neural networks replace tables for approximating Q functions?                                         |
| 4.3     | [Replay, Target Networks, and CNNs](docs/chapter04_dqn/dqn-components.md)                      | What stability problems do replay, target networks, and encoders solve?                                      |
| 4.4     | [Training Analysis](docs/chapter04_dqn/training-analysis.md)                                   | What do DQN training curves and Q-value changes reveal?                                                      |
| 4.5     | [Mountain Car and Sparse Rewards](docs/chapter04_dqn/mountain-car.md)                          | Why do exploration and initialization matter when rewards are rare?                                          |
| 4.6     | [Double, Dueling, and Rainbow](docs/chapter04_dqn/dqn-family.md)                               | How did the DQN family fix overestimation, representation, and sampling issues?                              |
| 4.7     | [Project: DQN and Visual Games](docs/chapter04_dqn/visual-game-projects.md)                    | What engineering changes are needed when moving from low-dimensional control to visual games?                |
| 05      | [Policy Gradient and REINFORCE](docs/chapter05_policy_gradient/intro.md)                       | How can policies be optimized directly, and why do baselines reduce gradient variance?                       |
| 5.1     | [Hands-On: Dice Gambling Bandit](docs/chapter05_policy_gradient/dice-game.md)                  | How does a minimal experiment reveal policy-gradient sampling updates?                                       |
| 5.2     | [Policy Gradient and REINFORCE](docs/chapter05_policy_gradient/policy-gradient.md)             | How does REINFORCE increase the probability of high-return actions?                                          |
| 5.3     | [Hands-On: Baseline Variance Reduction](docs/chapter05_policy_gradient/baseline-experiment.md) | Why does a baseline reduce variance without changing the expectation?                                        |
| 06      | [Actor-Critic](docs/chapter06_actor_critic/intro.md)                                           | How do actor and critic split the learning problem, and how does TD error become an advantage signal?        |
| 6.1     | [Advantage Function](docs/chapter06_actor_critic/advantage-function.md)                        | How does advantage answer "how much better was this action than average?"                                    |
| 6.2     | [Training the Critic with TD Error](docs/chapter06_actor_critic/critic-training.md)            | How does the critic learn value estimates from bootstrapped signals?                                         |
| 6.3     | [Actor-Critic Architecture](docs/chapter06_actor_critic/actor-critic.md)                       | How do actor and critic work together in one training loop?                                                  |
| 6.4     | [Project: A Simple AlphaGo Reproduction](docs/chapter06_actor_critic/alphago.md)               | How do policy networks, value networks, and search combine into a game-playing agent?                        |
| 07      | [PPO](docs/chapter07_ppo/intro.md)                                                             | How do clipping, trust-region intuition, GAE, and reward models stabilize policy optimization?               |
| 7.1     | [Hands-On: PPO on LunarLander](docs/chapter07_ppo/ppo-lunar-lander.md)                         | How does PPO behave on a more complex control task, and how should it be tuned?                              |
| 7.2     | [PPO Math Derivation](docs/chapter07_ppo/ppo-math.md)                                          | How does the PPO objective move from policy gradient to clipped surrogate objective?                         |
| 7.3     | [Trust Regions and Clipping](docs/chapter07_ppo/trust-region-clipping.md)                      | How does clipping limit policy update size?                                                                  |
| 7.4     | [GAE and Reward Models](docs/chapter07_ppo/gae-reward-model.md)                                | How does GAE balance bias and variance, and how does it connect to reward model training?                    |
| Summary | [Part 2 Summary](docs/summaries/part2-summary.md)                                              | What algorithmic patterns repeat across classic and modern RL?                                               |

### Part 3: LLM-era RL

| Chapter | Topic                                                                                                           | Core Question                                                                                                    |
| :------ | :-------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| 08      | [The Full RLHF Pipeline](docs/chapter08_rlhf/intro.md)                                                          | How do instruction data, reward models, PPO training, evaluation, and scaling fit together?                      |
| 8.1     | [Why a Base Model Is Not Yet an Assistant](docs/chapter08_rlhf/base-model-to-assistant.md)                      | What is the gap between a pretrained model and an assistant model?                                               |
| 8.2     | [The Standard RLHF Pipeline](docs/chapter08_rlhf/standard-rlhf-pipeline.md)                                     | How do SFT, RM, and RL connect as three training stages?                                                         |
| 8.3     | [SFT: Teaching the Model to Follow Instructions](docs/chapter08_rlhf/imitation-learning-pipeline.md)            | How does supervised fine-tuning build basic instruction-following ability?                                       |
| 8.4     | [Reward Model: Teaching a Judge](docs/chapter08_rlhf/reward-function-design.md)                                 | How does a reward model turn human preferences into an optimizable signal?                                       |
| 8.5     | [PPO-RLHF: Practicing Against Rewards](docs/chapter08_rlhf/ppo-rlhf-loop.md)                                    | How does PPO optimize a language model under a KL constraint?                                                    |
| 8.6     | [Evaluation: Did RLHF Actually Improve the Model?](docs/chapter08_rlhf/evaluation.md)                           | How can we tell whether alignment training improved the model?                                                   |
| 8.7     | [Scaling from Small Models to Large Models](docs/chapter08_rlhf/scaling-to-large-models.md)                     | What engineering problems appear when the same RLHF pipeline is scaled up?                                       |
| 8.8     | [Extended Practice: Reward Hacking and Data Flywheels](docs/chapter08_rlhf/extended-practice.md)                | How can reward gaming be detected, and how can data iteration keep improving the model?                          |
| 09      | [Post-Training Alignment](docs/chapter09_alignment/intro.md)                                                    | How do DPO, GRPO, DeepSeek-R1, and verifiable rewards train reasoning behavior?                                  |
| 9.1     | [DPO, IPO, and KTO](docs/chapter09_alignment/dpo-theory-and-family.md)                                          | How does the preference-optimization family bypass explicit reward models?                                       |
| 9.2     | [Hands-On: DPO Alignment Experiment](docs/chapter09_alignment/dpo-hands-on.md)                                  | How can a DPO training experiment be run and inspected end to end?                                               |
| 9.3     | [GRPO Practice and Mechanism](docs/chapter09_grpo_rlvr/grpo-practice-and-mechanism.md)                          | How does GRPO replace a critic with within-group relative advantage?                                             |
| 9.4     | [DeepSeek-R1 and DAPO](docs/chapter09_grpo_rlvr/deepseek-dapo.md)                                               | What new RL lessons appear in reasoning-model training?                                                          |
| 9.5     | [RLVR: Verifiable Rewards](docs/chapter09_grpo_rlvr/rlvr.md)                                                    | How can rule-checkable tasks provide stable rewards for RL?                                                      |
| 9.6     | [On-Policy Distillation](docs/chapter09_grpo_rlvr/on-policy-distillation.md)                                    | How can online RL behavior be distilled back into a more usable model?                                           |
| 10      | [Agentic RL](docs/chapter10_agentic_rl/intro.md)                                                                | How do multi-turn interaction, tool use, trajectory synthesis, and agent systems engineering change RL problems? |
| 10.1    | [Multi-Turn Interaction and Credit Assignment](docs/chapter10_agentic_rl/multi-turn-rl.md)                      | In multi-step tasks, how can final outcomes be assigned back to intermediate actions?                            |
| 10.2    | [Tool Use, Trajectory Synthesis, and Agentic Engineering](docs/chapter10_agentic_rl/tool-use-and-trajectory.md) | How do tool execution results enter RL trajectories and training data?                                           |
| 10.3    | [Industrial Practice, Evaluation, and Bad Cases](docs/chapter10_agentic_rl/industrial-evaluation.md)            | What failure modes appear most often in engineering evaluation for Agentic RL?                                   |
| 10.4    | [Project 1: Multi-Tool Code Agent](docs/chapter10_agentic_rl/multi-tool-code-agent.md)                          | How can a model be trained to switch among search, coding, and testing?                                          |
| 10.5    | [Project 2: Deep Research Agent](docs/chapter10_agentic_rl/deep-research-agent.md)                              | How do research agents organize search, citations, and answer-quality rewards?                                   |
| 10.6    | [Extended Readings](docs/chapter10_agentic_rl/extended-readings.md)                                             | What should learners read next to go deeper into Agentic RL?                                                     |
| Summary | [Part 3 Summary](docs/summaries/part3-summary.md)                                                               | What makes RL for LLMs different from RL in classic environments?                                                |

### Part 4: Frontier and Advanced Systems

| Chapter | Topic                                                                                        | Core Question                                                                                     |
| :------ | :------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| 11      | [VLM Reinforcement Learning](docs/chapter11_vlm_rl/intro.md)                                 | How do visual rewards, multimodal frameworks, and visual generation RL change the training loop?  |
| 11.1    | [Hands-On: GRPO Training for VLMs](docs/chapter11_vlm_rl/vlm-grpo-hands-on.md)               | How can GRPO training be extended to visual question answering tasks?                             |
| 11.2    | [Visual Rewards and Hallucination](docs/chapter11_vlm_rl/vlm-challenges.md)                  | What new problems do multimodal rewards and visual hallucinations introduce?                      |
| 11.3    | [Open-R1, R1-V, and VLM-R1](docs/chapter11_vlm_rl/vlm-frameworks.md)                         | How do frontier VLM-RL frameworks organize data, rewards, and training?                           |
| 11.4    | [Visual Generation RL](docs/chapter11_vlm_rl/visual-generation-rl.md)                        | How can image generation models be optimized with preferences and rewards?                        |
| 12      | [Future Trends](docs/chapter12_future_trends/intro.md)                                       | Where are embodied intelligence, model-based RL, self-play, multi-agent RL, and offline RL going? |
| 12.1    | [Embodied Intelligence](docs/chapter12_future_trends/embodied-intelligence/index.md)         | How does RL enter robotics and the physical world?                                                |
| 12.2    | [Model-Based RL](docs/chapter12_future_trends/embodied-intelligence/model-based-rl/index.md) | How can world models reduce the cost of real environment interaction?                             |
| 12.3    | [Self-Play and Self-Evolution](docs/chapter12_future_trends/self-play-outlook/index.md)      | How can self-play drive continuous capability improvement?                                        |
| 12.4    | [LLM Multi-Agent RL](docs/chapter12_future_trends/llm-multi-agent-rl/index.md)               | How can multiple language agents collaborate, compete, and learn together?                        |
| 12.5    | [Offline Reinforcement Learning](docs/chapter12_future_trends/offline-rl/index.md)           | How can a policy be learned from fixed data when online trial and error is unavailable?           |
| 12.6    | [RL Scaling Outlook](docs/chapter12_future_trends/rl-scaling-outlook.md)                     | Where might large-scale RL training go next?                                                      |
| Summary | [Part 4 Summary](docs/summaries/part4-summary.md)                                            | What directions should learners follow after finishing the core course?                           |

### Appendices

| Appendix | Topic                                                                                               | Description                                                                                           |
| :------- | :-------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| A        | [Training Debugging Guide](docs/appendix_common_pitfalls/intro.md)                                  | Failure modes, symptoms, root causes, and fixes for RL training.                                      |
| B        | [RL Engineering Practice](docs/appendix_industrial_training/intro.md)                               | Training infrastructure, agent sandboxes, evaluation benchmarks, and industrial exercises.            |
| B.1      | [Training System Foundations](docs/appendix_industrial_training/rl-infrastructure.md)               | What infrastructure does an RL training system need?                                                  |
| B.2      | [Agent Sandboxes and Tool Scheduling](docs/appendix_industrial_training/agentic-rl-infra.md)        | How should tool-use agent training isolate execution environments?                                    |
| B.3      | [RL and Agent Benchmarks](docs/appendix_industrial_training/evaluation-badcase.md)                  | How should evaluations and bad-case analysis be designed?                                             |
| B.4      | [Training Metrics Glossary](docs/appendix_industrial_training/metrics-glossary.md)                  | What do common training metrics indicate?                                                             |
| B.5      | [Industrial Practice Exercises](docs/appendix_industrial_training/industrial-exercises.md)          | How can engineering concepts be turned into practice tasks?                                           |
| D        | [Learning Resources and Project Recommendations](docs/appendix_game_projects/intro.md)              | Curated resources and reproduction projects for expanding course examples.                            |
| E        | [Math Foundations](docs/appendix_math/intro.md)                                                     | Linear algebra, probability and statistics, calculus and optimization, and information theory for RL. |
| E.1      | [Math Objects and Linear Algebra](docs/appendix_math/linear-algebra.md)                             | How do vectors, matrices, and function approximation support RL representations?                      |
| E.2      | [Probability, Expectation, and Stochastic Estimation](docs/appendix_math/probability-statistics.md) | What probability tools do returns, sampling, and trajectory estimation depend on?                     |
| E.3      | [Calculus and Optimization](docs/appendix_math/calculus-optimization.md)                            | How do gradients, the chain rule, and optimizers drive policy updates?                                |
| E.4      | [Information Theory and Distribution Distance](docs/appendix_math/information-theory.md)            | How do entropy, cross-entropy, and KL explain exploration and alignment constraints?                  |

## Experiment Code

The [`code/`](code/) directory contains runnable examples aligned with course chapters. Each chapter's code is intentionally compact so it can be inspected, run, and modified independently.

| Area                   | Code Path                                                                                                          | Representative Experiments                                                                         |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| Classic control        | [`code/chapter01_cartpole/`](code/chapter01_cartpole/)                                                             | Train CartPole, inspect rewards and episode length, and compare PPO implementations.               |
| Preference fine-tuning | [`code/chapter02_dpo/`](code/chapter02_dpo/)                                                                       | Generate preference data, train with DPO, and compare model behavior before and after fine-tuning. |
| MDP and value learning | [`code/chapter03_mdp/`](code/chapter03_mdp/)                                                                       | Run bandit strategies, solve GridWorld, and verify Bellman updates numerically.                    |
| Deep Q-learning        | [`code/chapter04_dqn/`](code/chapter04_dqn/)                                                                       | Implement replay buffers, target networks, and Double DQN variants.                                |
| Policy gradient        | [`code/chapter05_policy_gradient/`](code/chapter05_policy_gradient/)                                               | Compare REINFORCE, baseline variants, and Actor-Critic updates.                                    |
| PPO                    | [`code/chapter07_ppo/`](code/chapter07_ppo/)                                                                       | Train LunarLander, inspect clipping, visualize GAE, and compare training stability.                |
| RLHF                   | [`code/chapter08_rlhf/`](code/chapter08_rlhf/)                                                                     | Walk through SFT, reward model training, and PPO-style alignment.                                  |
| Alignment and RLVR     | [`code/chapter09_alignment/`](code/chapter09_alignment/), [`code/chapter09_grpo_rlvr/`](code/chapter09_grpo_rlvr/) | Explore DPO rewards, GRPO group advantages, and rule-based verifiable rewards.                     |
| VLM and agents         | [`code/chapter10_agentic_rl/`](code/chapter10_agentic_rl/), [`code/chapter11_vlm_rl/`](code/chapter11_vlm_rl/)     | Build tool-use agent trajectory synthesis and implement multimodal model RL examples.              |
| Advanced topics        | [`code/chapter12_future_trends/`](code/chapter12_future_trends/)                                                   | Study frontier directions including multi-agent RL and model-based RL.                             |

See [`code/README.md`](code/README.md) for a code index and chapter-specific dependency notes.

## Recommended Learning Path

A practical path through the repository:

1. Read the [course guide](docs/preface/intro.md) and run the CartPole example.
2. Skim the DPO chapter early, even before finishing all theory, to anchor the motivation for LLM post-training.
3. Study Chapters 03-07 in order; this is the conceptual core.
4. After understanding policy gradients and PPO, return to RLHF, DPO, GRPO, and RLVR.
5. Use the debugging and engineering appendices whenever a training run behaves strangely.
6. Treat frontier chapters as extensions: VLM reinforcement learning, Agentic RL, continuous control, multi-agent systems, and test-time reasoning.

## Quick Start

### Read Online

Published course site:

```text
https://walkinglabs.github.io/hands-on-modern-rl/
```

### Run the Documentation Site Locally

Requirements:

- Node.js >= 18.0.0
- npm

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

Then open the local VitePress URL shown in the terminal, usually:

```text
http://localhost:5173
```

### Verify the Site

Before submitting a pull request that changes documentation structure, theme code, navigation, build scripts, or generated assets, run:

```bash
npm run verify
```

This checks formatting, lints the VitePress theme, builds the site, and verifies expected build artifacts.

### Run Course Code

Most code examples use Python and are organized by chapter.

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For smaller installs, use chapter-specific requirements files:

```bash
pip install -r chapter01_cartpole/requirements.txt
python chapter01_cartpole/1-ppo_cartpole.py
```

Some chapters may require additional system libraries, GPU support, model downloads, or environment-specific setup. Start with Chapter 01 before running examples that involve LLMs, VLMs, or heavy simulators.

## Repository Structure

```text
hands-on-modern-rl/
|-- docs/                      # VitePress course content
|   |-- .vitepress/            # Site config, navigation, and theme overrides
|   |-- public/                # Static assets copied into the built site
|   |-- preface/               # Course introduction and history
|   |-- chapter*/              # Main course chapters
|   |-- appendix*/             # Supplementary material and references
|   `-- summaries/             # Part-level review and summary notes
|-- code/                      # Runnable examples aligned with chapters
|-- scripts/                   # Maintenance and verification scripts
|-- package.json               # Site scripts and dependencies
|-- AGENTS.md                  # Repository maintenance guide
`-- README.md                  # Main project overview
```

## Development Commands

```bash
npm run dev           # Start the local documentation server
npm run build         # Build the static site
npm run preview       # Preview the built site locally
npm run format        # Format repository files with Prettier
npm run format:check  # Check formatting
npm run lint          # Lint VitePress theme code
npm run verify        # Run format check, lint, build, and artifact verification
```

## Contributing

Contributions should make the course clearer, more accurate, easier to reproduce, or easier to navigate.

Good contributions include:

- Fixing conceptual errors, formulas, diagrams, broken links, or typos.
- Improving explanations without changing the intended learning path.
- Adding small, reproducible experiments that clarify existing chapters.
- Improving scripts, build reliability, navigation, or accessibility.
- Adding high-quality references to papers, official documentation, or widely used open-source implementations.

Please keep pull requests focused. A good PR usually changes one chapter, one experiment, one group of diagrams, or one infrastructure issue at a time.

When adding content:

1. Put course material under [`docs/`](docs/).
2. Use kebab-case for new directories and files.
3. Prefer directory-based routes with `index.md`.
4. Update [`docs/.vitepress/config.mjs`](docs/.vitepress/config.mjs) when adding navigable pages.
5. Run `npm run verify` before requesting review if your change touches config, theme, scripts, or generated site output.
6. Use Conventional Commits, such as `docs: clarify ppo clipping` or `fix: repair chapter link`.

For repository-specific maintenance rules, see [`AGENTS.md`](AGENTS.md).

## Citation

If you use this course in teaching materials, study notes, or derivative non-commercial educational work, please cite the repository:

```bibtex
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```

## License

This course is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).

You may share and adapt the material for non-commercial purposes, provided that you give appropriate credit and distribute derivative works under the same license.

## Star History

<a href="https://star-history.com/#walkinglabs/hands-on-modern-rl&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=walkinglabs/hands-on-modern-rl&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=walkinglabs/hands-on-modern-rl&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=walkinglabs/hands-on-modern-rl&type=Date" />
  </picture>
</a>

---

<div align="center">
  <sub>Maintained by WalkingLabs and contributors.</sub>
</div>
