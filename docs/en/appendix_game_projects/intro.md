---
title: D. Learning Resources and Reproduction Projects
---

# Appendix D: Learning Resources and Reproduction Projects

> **Goal of this appendix**: provide clear navigation for your continued advancement. The first half organizes textbooks and courses with solid theory and clear exposition to help you systematically build foundations or explore frontiers. The second half surveys classic milestones and common environments in RL's game and simulation ecosystem, giving you inspiration and coordinates for your next hands-on reproduction project.

## Recommended Learning Resources

> **How to use this list**: This book covers the complete pipeline from MDP basics through PPO, DPO, and GRPO, but RL goes far beyond that. If you want to dive deeper into a specific direction, compare different teaching styles, or find hands-on practice resources, this list can serve as a starting point. All resources are free or publicly accessible.

**Choose based on your goal:**

- **Just finished Chapter 3, want to see how other textbooks cover basic theory**: start with Zhao Shiyu's _Mathematical Principles of RL_ or Sutton & Barto.
- **Want to follow video lectures**: start with David Silver's course or Li Hongyi's course.
- **Want to write code**: start with OpenAI Spinning Up or _Dive into Deep Reinforcement Learning_.
- **Interested in LLM alignment / RLHF / GRPO**: start with Nathan Lambert's RLHF Book or Ernest Ryu's RL-LLM course.
- **Want to explore frontier theory**: start with Princeton ECE 524 or Alberta CMPUT 365.

**I. Classic Textbooks**

**Reinforcement Learning: An Introduction (Sutton & Barto, 2nd Edition, 2018)**

**URL**: [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html) | [Chinese Translation](https://rl.qiwihui.com/)

The standard textbook for RL, listed as required reading by nearly every university RL course. Three parts: Part I (tabular methods, Ch1-8) covers MDP, DP, MC, TD, n-step bootstrapping, planning; Part II (approximate methods, Ch9-13) covers function approximation, eligibility traces, policy gradients; Part III (Ch14-17) discusses psychology, neuroscience, and applications. Free PDF; Chinese translation quality is high. **Best for systematically building foundations.**

**Mathematical Foundations of Reinforcement Learning (Zhao Shiyu)**

**URL**: [github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) (GitHub 10k+ stars)

Published by Springer + Tsinghua University Press. 10 chapters rigorously deriving core RL algorithms from a mathematical perspective: Bellman equations → VI/PI → MC → TD (including Sarsa, Q-Learning, n-step Sarsa) → function approximation → policy gradients → Actor-Critic. Each chapter includes mathematical proofs and exercises. **Best for readers who prefer rigorous derivation and want to understand "why these algorithms work" at the mathematical level.**

**Deep Reinforcement Learning (Zhang Zhihua, Peking University)**

**URL**: [PDF Draft](https://www.math.pku.edu.cn/teachers/zhzhang/drl_v1.pdf)

Textbook for Peking University's math department course. Assumes ML basics but not necessarily RL familiarity. Covers value-based learning (DQN), policy learning (Policy Gradient), Actor-Critic, TRPO, etc. Paired with Wang Shusen's Bilibili video course. **Best for Chinese readers seeking a quick DRL introduction.**

**Dive into Deep Reinforcement Learning (Zhang Weinan, Shen Jian, Yu Yong)**

**URL**: [Online Version](https://hrl.boyuai.com/) | Shanghai Jiao Tong University RL course textbook

Practice-oriented with runnable Jupyter code throughout. Three parts: basics (Bandit → MDP → DP → MC → Planning) → advanced (function approximation → DQN → policy gradients → PPO) → frontier (Model-Based RL, Offline RL). **Best for learners who want to read and code simultaneously.**

**II. University Courses**

**European and American Courses**

**Stanford CS234: Reinforcement Learning (Emma Brunskill)**

**URL**: [web.stanford.edu/class/cs234/](https://web.stanford.edu/class/cs234/)

Stanford's foundational RL course. From tabular MDPs through policy evaluation, Q-Learning, function approximation, policy gradients, Offline RL, exploration, MCTS, and finally RLHF. About half the lectures build theory; the other half cover advanced topics. Textbook: Sutton & Barto.

**Stanford CS224R: Deep Reinforcement Learning (Chelsea Finn)**

**URL**: [cs224r.stanford.edu](https://cs224r.stanford.edu/) | [YouTube 2025](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL)

Stanford's Deep RL course. Assumes RL basics; starts directly with imitation learning, quickly moving into policy gradients, Actor-Critic, Q-Learning, Model-Based RL, Offline RL, Reward Learning, RLHF, and Meta-RL. **Best for learners who already know basics and want to dive deep into DRL directions.**

**MIT 6.7920: Reinforcement Learning Foundations and Methods (Cathy Wu)**

**URL**: [web.mit.edu/6.7920/www/](https://web.mit.edu/6.7920/www/)

MIT's RL theory course. Two-thirds "exploitation" (known theory: DP 7 lectures + RL core methods 9 lectures), one-third "exploration" (frontier topics). DP section is very solid, covering finite/infinite horizon, LQR, policy/value iteration, convergence proofs. **Best for learners seeking theoretical depth.**

**UC Berkeley CS285: Deep Reinforcement Learning (Sergey Levine)**

**URL**: [rail.eecs.berkeley.edu/deeprlcourse/](https://rail.eecs.berkeley.edu/deeprlcourse/)

Berkeley's flagship Deep RL course. Only 1 lecture reviews RL basics, then dives into imitation learning, policy gradients, Actor-Critic, Value-Based RL, advanced policy gradients, variational inference & RL, LLM RL, Model-Based RL, Offline RL, and exploration. The 2026 spring edition adds hands-on assignments for LLM RL and Offline RL. **Content most aligned with current industrial frontiers.**

**CMU 10-703: Deep Reinforcement Learning and Control**

**URL**: [cmudeeprl.github.io/703website_f25/](https://cmudeeprl.github.io/703website_f25/)

CMU's Deep RL course. After covering classical theory (MDP, DP, MC, TD), moves into function approximation, Deep Q-Learning, MCTS, policy gradients, imitation learning, inverse RL, optimal control, Model-Based RL, and exploration. **Balanced theory and practice with broad coverage.**

**University of Alberta CMPUT 365: Introduction to RL (Marlos Machado)**

**URL**: [Syllabus PDF](https://webdocs.cs.ualberta.ca/~machado/cmput365/w26/syllabus.pdf)

Introductory RL course at Sutton's university, strictly following Sutton & Barto order: Bandits → MDP → DP (including PI, VI, GPI) → MC prediction and control → TD prediction → **TD control (Sarsa, Q-Learning)** → Planning (Dyna-Q) → function approximation → policy gradients. **Most faithful course implementation of Sutton & Barto.**

**Georgia Tech CS 7642: Reinforcement Learning (OMSCS)**

**URL**: [omscs.gatech.edu/cs-7642-reinforcement-learning](https://omscs.gatech.edu/cs-7642-reinforcement-learning)

Online RL course. Covers DP, TD (including Sarsa), n-step TD, Lambda Return, DQN, policy gradients, multi-agent RL, game theory, and POMDP. **One of the best-regarded RL courses in the OMSCS program.**

**Princeton ECE 524: Foundations of RL (Chi Jin)**

**URL**: [sites.google.com/view/cjin/teaching/ece524](https://sites.google.com/view/cjin/teaching/ece524) | [YouTube](https://www.youtube.com/playlist?list=PLYXvCE1En13epbogBmgafC_Yyyk9oQogl)

Theory-oriented, emphasizing finite-sample analysis and convergence proofs. Part I covers tabular MDPs, planning, exploration (Bandit and MDP), lower bounds; Part II covers large state spaces, linear VI, function approximation, multi-agent, and POMDP. **Best for learners aiming to do RL theory research.**

**David Silver RL Course (UCL / DeepMind)**

**URL**: [davidsilver.uk/teaching](https://www.davidsilver.uk/teaching/) | [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

10 classic lectures: MDP → DP → Model-Free Prediction → Model-Free Control → function approximation → policy gradients → Learning & Planning → exploration → classic game case studies. David Silver is the first author of AlphaGo/AlphaZero. **Concise structure, clear explanations; the most widely disseminated RL video course.**

**DeepMind x UCL RL Lecture Series (2021)**

**URL**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

Updated version of David Silver's course, taught by DeepMind researchers (Hado van Hasselt et al.). 13 lectures covering exploration and control, MDPs and DP, model-free methods, function approximation, planning, policy gradients and Actor-Critic, approximate DP, multi-step and off-policy, and Deep RL. **More in-depth than the 2015 version with additional frontier content.**

**Chinese University Courses**

**Tsinghua University Reinforcement Learning (Fall 2025)**

**URL**: [coai.cs.tsinghua.edu.cn/Courses/RL2025/\_site/](https://coai.cs.tsinghua.edu.cn/Courses/RL2025/_site/)

Undergraduate RL course. Starting from multi-armed bandits, covers MDP, Planning (DP), MC, TD Learning, policy gradients, function approximation, and Deep RL. 4 programming assignments (Bandit → MDP → TD & PG → Deep RL) + course project. Lecture slides are publicly available.

**Nanjing University Foundations of Reinforcement Learning (Yu Yang, 2024)**

**URL**: [lamda.nju.edu.cn/introrl](https://www.lamda.nju.edu.cn/introrl/)

Based on Sutton & Barto. 9 lectures covering RL basics, MDP, DP, MC, TD, and DQN. 5 programming assignments (Dagger → Q-Learning → DQN → Model-Based → Offline RL). **One of the most theoretically solid Chinese university RL courses.**

**Nanjing University Advanced Reinforcement Learning (Yuan Lei, 2025)**

**URL**: [lamda.nju.edu.cn/advanceRL](https://www.lamda.nju.edu.cn/advanceRL/)

Graduate advanced course. Covers DDPG/TD3, PPO techniques, multi-agent, RLHF/DPO theoretical derivations, and paper reading.

**Shanghai Jiao Tong University Reinforcement Learning (Zhang Weinan, 2024)**

**URL**: [wnzhang.net/teaching/sjtu-rl-2024](https://wnzhang.net/teaching/sjtu-rl-2024/)

Uses _Dive into Deep Reinforcement Learning_ as textbook. 9 chapters covering basics through frontiers.

**III. Chinese Online Courses and Tutorials**

**Li Hongyi Deep Reinforcement Learning (National Taiwan University)**

**URL**: [Course Page](https://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html) | [Bilibili 2025](https://www.bilibili.com/video/BV1SJvAzfEL2/)

Uses Policy Gradient as the main thread, deeply explaining PPO (including Importance Sampling, On-policy → Off-policy derivation), then Q-Learning (DQN, Double DQN, Dueling DQN) and Actor-Critic. Lively explanations with polished slides. **Most in-depth PPO coverage among Chinese courses.**

**Wang Shusen Deep Reinforcement Learning**

**URL**: [Bilibili Video](https://www.bilibili.com/video/BV1oEWDz1Ez5/)

Video companion to Peking University's math department course. Five modules: basic concepts → value learning (DQN) → policy learning (Policy Gradient) → Actor-Critic (A3C, TRPO) → advanced (DDPG, AlphaGo, multi-agent). Paired with Zhang Zhihua's _Deep Reinforcement Learning_ textbook. **Concise content suitable for quick introduction.**

**Mushu Book EasyRL (Datawhale)**

**URL**: [Online Version](https://datawhalechina.github.io/easy-rl/) | [GitHub](https://github.com/datawhalechina/easy-rl)

Synthesizes the best of Zhoubolei's _RL Outline_, Li Hongyi's course, and Baidu's _World Champion Takes You from Zero to RL Practice_. 13 chapters + special topics, covering basics through DQN, PPO, DDPG, and AlphaStar. **Most active open-source RL tutorial in the Chinese community.**

**Spinning Up Chinese Edition**

**URL**: [spinningup.qiwihui.com/zh-cn/latest](https://spinningup.qiwihui.com/zh-cn/latest/)

Chinese translation of OpenAI Spinning Up. Includes core concepts, algorithm taxonomy, policy gradient derivations, and implementations of VPG, TRPO, PPO, DDPG, TD3, and SAC.

**IV. LLM Reinforcement Learning Specialized Resources**

**Nathan Lambert — RLHF Book + Course**

**URL**: [rlhfbook.com](https://rlhfbook.com/) | [Course](https://rlhfbook.com/course) | [GitHub](https://github.com/natolambert/rlhf-book) | [YouTube](https://www.youtube.com/playlist?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)

RLHF monograph by AI2 researcher Nathan Lambert. Covers the full RLHF pipeline: instruction tuning → reward model training → rejection sampling → PPO → DPO. Code repository implements PPO, REINFORCE, GRPO, RLOO and other policy gradient methods. 4 video lectures. **Most systematic publicly available textbook on LLM alignment.**

**Ernest Ryu — Reinforcement Learning of Large Language Models (UCLA)**

**URL**: [ernestryu.com/courses/RL-LLM.html](https://ernestryu.com/courses/RL-LLM.html)

The only university course that systematically combines classical RL theory with LLM RL. Three parts: Ch1 (5 lectures on classic RL: MDP → VI → PG → PPO/GRPO → AlphaGo) → Ch2 (4 lectures on LLM basics: NLP → Transformer → ICL/SFT) → Ch3 (2 lectures on LLM RL: RLHF/PPO/DPO → RLVR). **LLM RL course with the deepest RL foundations.**

**DeepLearning.AI — Reinforcement Fine-Tuning LLMs with GRPO**

**URL**: [deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/)

1-hour short course, 10 lessons. Uses Wordle as the running example, covering GRPO algorithm, reward function design, LLM-as-Judge, and reward hacking. 7 code experiments. **Best for practitioners with LLM basics who want to quickly get started with GRPO.**

**Hugging Face — Deep RL Course**

**URL**: [huggingface.co/learn/deep-rl-course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)

8 units covering Q-Learning → DQN → Policy Gradient → A2C/A3C → PPO → multi-agent → Offline RL. Each unit includes theory and code practice. Bonus unit covers RLHF. **Best for learners wanting to do RL experiments in the Hugging Face ecosystem.**

**V. Practical Tutorials and Technical Blogs**

**OpenAI Spinning Up in Deep RL**

**URL**: [spinningup.openai.com](https://spinningup.openai.com/en/latest/)

The gold standard for RL basics education. Three parts: core concepts (V/Q/Bellman/Advantage) → algorithm taxonomy (Model-Based vs Model-Free) → policy optimization derivation (deriving Policy Gradient from scratch). Implements VPG, TRPO, PPO, DDPG, TD3, and SAC. **Best combination of theoretical explanation and code implementation.**

**Cameron Wolfe — Deep (Learning) Focus**

**URL**: [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm) | [Online vs Offline RL for LLMs](https://cameronrwolfe.substack.com/p/online-rl)

Blog series explaining PPO in LLMs, online vs offline RL tradeoffs, DPO principles, etc., in accessible language. **Best for readers wanting to understand "why LLM RL uses these algorithms."**

**Sebastian Raschka — Ahead of AI**

**URL**: [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) | [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)

Technical blog by the author of _Build a Large Language Model From Scratch_. Covers RLHF, DPO, RLVR, GRPO, inference-time scaling, and other frontier topics.

## Reproduction Project Recommendations

RL projects can be split into two eras. The non-LLM era focuses on fixed simulation environments, game benchmarks, continuous control, multi-agent, and model learning. The LLM era extends actions to tokens, tool calls, web operations, visual reasoning, and long-horizon agent trajectories, with rewards expanding from environment scores to preference models, rule verifiers, process rewards, and real task success rates.

### Reproduction Roadmap Quick Reference

| Target Direction               | Priority Resources                                                                                                                                                                                                                                                                    | What to Reproduce                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Classic algorithm introduction | [CleanRL](https://github.com/vwxyzjn/cleanrl), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [Dopamine](https://github.com/google/dopamine)                                                     | DQN, PPO, SAC, TD3, Rainbow DQN, Atari benchmarks                                                 |
| Environments & game benchmarks | [Gymnasium](https://gymnasium.farama.org/), [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [MiniGrid](https://minigrid.farama.org/), [Procgen](https://github.com/openai/procgen), [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)                  | CartPole, LunarLander, Atari, FPS, procedurally generated environments                            |
| Multi-agent & games            | [PettingZoo](https://pettingzoo.farama.org/), [OpenSpiel](https://github.com/google-deepmind/open_spiel), [SMAC](https://github.com/oxwhirl/smac), [Google Research Football](https://github.com/google-research/football)                                                            | Self-play, cooperative/competitive MARL, StarCraft micromanagement, football                      |
| Robotics & embodied control    | [MuJoCo](https://mujoco.readthedocs.io/), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/), [ManiSkill](https://maniskill.readthedocs.io/), [Meta-World](https://github.com/Farama-Foundation/Metaworld), [LeRobot](https://github.com/huggingface/lerobot)                         | Continuous control, robot arms, mobile robots, imitation learning + RL                            |
| Model-Based / world models     | [DreamerV3](https://github.com/danijar/dreamerv3), [TD-MPC2](https://github.com/nicklashansen/tdmpc2), [mbrl-lib](https://github.com/facebookresearch/mbrl-lib), [MBPO](https://github.com/JannerM/mbpo)                                                                              | Learn dynamics models from pixels/states, then plan or optimize policies                          |
| LLM post-training              | [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155), [TRL](https://huggingface.co/docs/trl/index), [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL), [verl](https://github.com/verl-project/verl)                                                                               | PPO, DPO, GRPO, RLHF, preference alignment, reward model training                                 |
| LLM reasoning                  | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [TinyZero](https://github.com/Jiayi-Pan/TinyZero), [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                    | RLVR, math/code reasoning, R1-style reproduction, verifier design                                 |
| Deep Research RL               | [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/), [Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [WebThinker](https://github.com/RUC-NLPIR/WebThinker)             | Search, reading, evidence filtering, citation, research-style answers                             |
| Agentic RL                     | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), [Google ADK](https://github.com/google/adk-python), [Agent Lightning](https://github.com/microsoft/agent-lightning), [AReaL](https://github.com/inclusionAI/AReaL)                                               | Code, tool calling, web browsing, long-horizon task success rate optimization                     |
| GUI / Computer Use             | [OpenAI CUA](https://openai.com/index/computer-using-agent/), [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool), [UI-TARS](https://github.com/bytedance/UI-TARS), [OSWorld](https://os-world.github.io/)                      | Web, desktop, mobile GUI operations and visual grounding                                          |
| VLM                            | [TRL VLM GRPO](https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [Open Vision Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner), [Gemini Robotics](https://deepmind.google/models/gemini-robotics/) | Image QA, visual reasoning, GUI/web, robotic visual operations, vision-language rewards           |
| Generative model RL            | [DDPO](https://github.com/jannerm/ddpo), [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo), [AlignProp](https://align-prop.github.io/), [RLAIF-V](https://github.com/RLHF-V/RLAIF-V), [VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                 | Optimize image/multimodal generation with preference, aesthetics, safety, and consistency rewards |

### RL Directions Overview

For systematically choosing reproduction directions, use three axes: "algorithm problem + environment type + reward source." The table below can serve as a long-term maintained directory skeleton.

| Category                             | Representative Problem                                            | Recommended Projects/Frameworks                                                                                                                                                                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Value-Based RL                       | Learn discrete-action policy from Q-values                        | DQN, Double DQN, Dueling DQN, Rainbow; [Dopamine](https://github.com/google/dopamine), [CleanRL](https://github.com/vwxyzjn/cleanrl)                                                                                                                                                            |
| Policy Gradient / Actor-Critic       | Directly optimize policy, handle continuous or stochastic actions | REINFORCE, A2C/A3C, PPO, TRPO; [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [TRL PPO](https://huggingface.co/docs/trl/ppo_trainer)                                                                                                                                         |
| Off-Policy / Maximum Entropy         | Improve sample efficiency, encourage exploration and robustness   | DDPG, TD3, SAC, REDQ; [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [Tianshou](https://github.com/thu-ml/tianshou)                                                                                                                                                          |
| Distributional RL                    | Learn return distribution instead of single expectation           | C51, QR-DQN, IQN, FQF; [Dopamine](https://github.com/google/dopamine), [DI-engine](https://github.com/opendilab/DI-engine)                                                                                                                                                                      |
| Exploration / Curiosity              | Sparse rewards, long-horizon exploration, intrinsic motivation    | RND, ICM, count-based exploration; [MiniGrid](https://minigrid.farama.org/), [Procgen](https://github.com/openai/procgen)                                                                                                                                                                       |
| Model-Based RL                       | Learn environment model, then plan or imagine rollouts            | PETS, MBPO, Dreamer, TD-MPC; [mbrl-lib](https://github.com/facebookresearch/mbrl-lib), [DreamerV3](https://github.com/danijar/dreamerv3), [TD-MPC2](https://github.com/nicklashansen/tdmpc2)                                                                                                    |
| Offline / Batch RL                   | Use only offline data, no online exploration                      | BCQ, CQL, IQL, TD3+BC; [D4RL](https://github.com/Farama-Foundation/D4RL), [Minari](https://github.com/Farama-Foundation/Minari), [d3rlpy](https://github.com/takuseno/d3rlpy), [CORL](https://github.com/corl-team/CORL)                                                                        |
| Imitation / Reward Learning          | Learn from expert trajectories, preferences, or inverse RL        | BC, DAgger, GAIL, AIRL; [imitation](https://github.com/HumanCompatibleAI/imitation), [robomimic](https://github.com/ARISE-Initiative/robomimic), [LeRobot](https://github.com/huggingface/lerobot)                                                                                              |
| Goal-Conditioned / Hierarchical      | Long-horizon tasks, subgoals, options, and skills                 | HER, Options, HIRO, skill discovery; [MiniGrid/BabyAI](https://minigrid.farama.org/), [Meta-World](https://github.com/Farama-Foundation/Metaworld)                                                                                                                                              |
| Meta-RL / Multitask / Generalization | Cross-task transfer, fast adaptation, generalization              | MAML-RL, PEARL, multi-task PPO/SAC; [Meta-World](https://github.com/Farama-Foundation/Metaworld), [Procgen](https://github.com/openai/procgen), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)                                                                                     |
| Safe / Constrained RL                | Constrain costs, risks, safe exploration                          | CPO, PPO-Lagrangian, shielding; [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium), [OmniSafe](https://github.com/PKU-Alignment/omnisafe)                                                                                                                                    |
| Multi-Agent RL / Game AI             | Cooperation, competition, self-play, communication                | QMIX, MADDPG, MAPPO, AlphaZero; [PettingZoo](https://pettingzoo.farama.org/), [OpenSpiel](https://github.com/google-deepmind/open_spiel), [JaxMARL](https://github.com/FLAIROx/JaxMARL)                                                                                                         |
| Robotics / Embodied RL               | Continuous control, manipulation, navigation, Sim2Real            | PPO/SAC on robots, domain randomization, VLA; [Isaac Lab](https://isaac-sim.github.io/IsaacLab/), [ManiSkill](https://maniskill.readthedocs.io/), [robosuite](https://robosuite.ai/), [OpenVLA](https://github.com/openvla/openvla)                                                             |
| Distributed / Systems RL             | High-throughput rollout, multi-node training, productionization   | IMPALA, APPO, distributed PPO; [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html), [Sample Factory](https://github.com/alex-petrenko/sample-factory), [DI-engine](https://github.com/opendilab/DI-engine), [Acme](https://github.com/google-deepmind/acme)                             |
| RLHF / Preference Alignment          | Optimize language/multimodal models from human or AI preferences  | PPO, DPO, IPO, KTO, ORPO; [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155), [Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback), [TRL](https://huggingface.co/docs/trl/index), [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) |
| RLVR / Reasoning RL                  | Rule-verifiable rewards, math/code reasoning, long CoT            | GRPO, DAPO, RLOO, REINFORCE++; [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [DAPO](https://github.com/BytedTsinghua-SIA/DAPO), [reasoning-gym](https://github.com/open-thought/reasoning-gym)                                  |
| Agentic RL                           | Search, tool calling, code execution, web/desktop tasks           | Trajectory reward, tool-use reward, process reward; [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), [Google ADK](https://github.com/google/adk-python), [Agent Lightning](https://github.com/microsoft/agent-lightning), [SkyRL](https://docs.skyrl.ai/docs)               |
| VLM / GUI / Computer-Use RL          | Image understanding, GUI grounding, web/mobile/desktop control    | Multimodal GRPO, GUI action RL; [OpenAI CUA](https://openai.com/index/computer-using-agent/), [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [OSWorld](https://os-world.github.io/)  |
| Generative Model RL                  | Optimize image, video, audio generation models with rewards       | DDPO, AlignProp, RLAIF-V; [DDPO](https://github.com/jannerm/ddpo), [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo), [AlignProp](https://align-prop.github.io/), [VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                                               |

### Non-LLM Era: Fixed Environments, Simulation, and Classic Algorithms

This track is best for building solid RL fundamentals. Start with single-file implementations in small environments, then gradually move to Atari, continuous control, multi-agent, robotics, and Model-Based RL.

#### Environments and Algorithm Libraries

| Environment/Tool                                                                                | Type                         | Description                                                                                 | Recommended Use                                                   |
| ----------------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [Gymnasium](https://gymnasium.farama.org/)                                                      | General RL environment       | Successor to OpenAI Gym; CartPole, LunarLander, and other classic environments              | Getting started, algorithm debugging, course experiments          |
| [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Game environment             | Atari 2600 standard benchmark, used in DQN-series papers                                    | Pixel input, discrete actions, DQN family                         |
| [MiniGrid](https://minigrid.farama.org/)                                                        | Grid world                   | Lightweight GridWorld for studying exploration, sparse rewards, and generalization          | Introduction to exploration, hierarchical RL, task generalization |
| [Procgen](https://github.com/openai/procgen)                                                    | Procedurally generated games | 16 procedurally generated environments focusing on generalization                           | Overfitting analysis, generalization experiments                  |
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)                                         | FPS 3D environment           | First-person shooter, partially observable, visual input, long-horizon decisions            | Visual policies, POMDP, navigation and combat                     |
| [Stable-Retro](https://github.com/Farama-Foundation/stable-retro)                               | Classic games                | Gymnasium-style wrapper for retro console games                                             | Classic game reproduction, course demonstrations                  |
| [MuJoCo](https://mujoco.readthedocs.io/)                                                        | Physics simulation           | High-precision physics engine; HalfCheetah, Ant, Humanoid benchmarks                        | PPO, SAC, TD3, continuous control                                 |
| [PyBullet](https://pybullet.org/wordpress/)                                                     | Physics simulation           | Open-source robotics simulation, lightweight ecosystem                                      | Robotics introduction, MuJoCo alternative experiments             |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)                                              | GPU parallel simulation      | NVIDIA successor to Isaac Gym; large-scale parallel robot training                          | Large-scale embodied RL, Sim2Real                                 |
| [ManiSkill](https://maniskill.readthedocs.io/)                                                  | Robot manipulation           | Benchmark for robotic arm manipulation, visual control, and large-scale parallel simulation | Visual manipulation, imitation learning + RL                      |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld)                                    | Multi-task robotics          | Multi-task robotic arm benchmark                                                            | Multi-task RL, meta-learning, generalization                      |
| [PettingZoo](https://pettingzoo.farama.org/)                                                    | Multi-agent environment      | Multi-agent version of Gymnasium, supporting cooperative and competitive scenarios          | MARL introduction, parallel/turn-based action interfaces          |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel)                                      | Game framework               | Board games, card games, matrix games, and multi-agent algorithm collection                 | Self-play, CFR, AlphaZero variants                                |
| [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)                                     | Distributed RL               | Distributed RL library in the Ray ecosystem                                                 | Large-scale training, multi-agent production experiments          |
| [CleanRL](https://github.com/vwxyzjn/cleanrl)                                                   | Algorithm implementation     | Single-file, readable, reproduction-friendly                                                | Learning algorithm details, writing course code                   |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)                                | Algorithm library            | Well-packaged DQN, PPO, SAC, TD3 implementations                                            | Quick baselines, hyperparameter tuning, comparisons               |
| [Dopamine](https://github.com/google/dopamine)                                                  | Atari algorithm library      | Google's DQN/Rainbow/IQN research framework                                                 | Atari paper reproduction, distributional value learning           |

#### Recommended Reproduction Ladder

| Stage | Project Suggestion                    | Recommended Tools                                     | Acceptance Criteria                                       |
| ----- | ------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| 1     | CartPole, MountainCar, LunarLander    | Gymnasium, CleanRL, Stable-Baselines3                 | Can plot reward curves, understand replay and GAE         |
| 2     | DQN / Rainbow on Atari                | ALE, Dopamine, CleanRL                                | Reproduce at least 1 Atari experiment                     |
| 3     | PPO / SAC / TD3 on MuJoCo             | MuJoCo, Stable-Baselines3, RL Baselines3 Zoo          | Can explain entropy, target networks, Q bias              |
| 4     | Self-play and multi-agent             | PettingZoo, OpenSpiel, SMAC, Google Research Football | Can distinguish cooperative, competitive, and mixed games |
| 5     | Robot manipulation and visual control | Isaac Lab, ManiSkill, Meta-World, LeRobot             | Can run parallel simulation or imitation-to-RL pipeline   |
| 6     | Model-Based RL / World Models         | DreamerV3, TD-MPC2, mbrl-lib, MBPO                    | Can explain latent dynamics and planning                  |

#### Advanced Directions and Exercise Suggestions

| Direction                             | Recommended Reproduction Projects                                                                                                                                                   | Course Assignment Ideas                                                                                        |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Single-file algorithm implementation  | CleanRL's DQN, PPO, SAC, C51, PPO-LSTM                                                                                                                                              | Write 200-500 lines clearly covering replay, GAE, target networks, entropy                                     |
| High-performance RL systems           | [Sample Factory](https://github.com/alex-petrenko/sample-factory), [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html), [DI-engine](https://github.com/opendilab/DI-engine) | Compare single-machine, multi-process, and distributed rollout throughput and sample efficiency                |
| JAX / GPU parallelism                 | [Brax](https://github.com/google/brax), [PureJaxRL](https://github.com/luchris429/purejaxrl), [JaxMARL](https://github.com/FLAIROx/JaxMARL)                                         | Use jit/vmap/pmap for large-batch environments; understand the "environments can also be accelerated" paradigm |
| Offline RL                            | D4RL + CQL/IQL/TD3+BC, [Minari](https://github.com/Farama-Foundation/Minari), [d3rlpy](https://github.com/takuseno/d3rlpy), [CORL](https://github.com/corl-team/CORL)               | Compare online RL and offline RL extrapolation error                                                           |
| Imitation learning                    | BC, DAgger, GAIL, AIRL; [imitation](https://github.com/HumanCompatibleAI/imitation), [robomimic](https://github.com/ARISE-Initiative/robomimic)                                     | Train policy from expert trajectories, then fine-tune with RL                                                  |
| Reward learning & preference learning | GAIL/AIRL, preference comparison, reward model                                                                                                                                      | Construct "human preferences" or scripted preferences, observe reward hacking                                  |
| Safe & constrained RL                 | Safety-Gymnasium, OmniSafe, PPO-Lagrangian, CPO                                                                                                                                     | Plot both reward curve and cost curve; learn constrained optimization                                          |
| Exploration & sparse rewards          | MiniGrid, Montezuma's Revenge, Procgen; RND, ICM, episodic curiosity                                                                                                                | Study whether intrinsic rewards actually improve exploration vs just inflating training scores                 |
| Hierarchical & goal-conditioned RL    | HER, Options, HIRO, BabyAI, Meta-World                                                                                                                                              | Decompose long-horizon tasks into subgoals; compare flat vs hierarchical policies                              |
| Multi-task & generalization           | Procgen, Meta-World, LIBERO, ContinualWorld                                                                                                                                         | High scores on training environments aren't enough; test on unseen tasks and seeds                             |
| Multi-agent cooperation/competition   | PettingZoo, OpenSpiel, SMAC, Google Research Football, JaxMARL                                                                                                                      | Compare independent PPO, MAPPO, QMIX, self-play                                                                |
| Robot manipulation                    | MuJoCo, Isaac Lab, ManiSkill, robosuite, Meta-World                                                                                                                                 | Do reaching, pushing, pick-and-place, then add visual input                                                    |
| World models & planning               | DreamerV3, TD-MPC2, mbrl-lib, MBPO, IRIS                                                                                                                                            | Learn dynamics model first, then compare model-free vs model-based sample efficiency                           |
| Industrial applications               | [RecSim](https://github.com/google-research/recsim), [FinRL](https://github.com/AI4Finance-Foundation/FinRL), [Pearl](https://github.com/facebookresearch/Pearl)                    | Bandit/RL experiments in recommendation, advertising, financial trading; emphasize offline evaluation and risk |

#### Unity ML-Agents Introduction

[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) is a unique RL toolkit that enables training directly inside a 3D game engine. Unlike Gymnasium's 2D grids or PyBullet's pure physics simulation, ML-Agents provides complete 3D spaces including occlusion, perspective, gravity, and collision, suitable for studying visual navigation and spatial reasoning.

**Typical usage**:

```python
# Unity ML-Agents is compatible with the Gymnasium interface
from mlagents_envs.environment import UnityEnvironment

# Load a pre-built Unity environment (3D platform jumping)
env = UnityEnvironment(file_name="3DBall")

# ML-Agents uses its own API, but can be wrapped as a Gymnasium interface
from mlagents_envs.gym_utils import UnityToGymWrapper
gym_env = UnityToGymWrapper(env)

# Then train with Stable-Baselines3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", gym_env)
model.learn(total_timesteps=100000)
```

**Classic ML-Agents environment examples**:

| Environment   | Task Type                 | Difficulty   | Best For                                      |
| ------------- | ------------------------- | ------------ | --------------------------------------------- |
| 3DBall        | Balance control           | Introductory | Understanding continuous action spaces        |
| Crawler       | Quadruped walking         | Intermediate | Continuous control + multi-joint coordination |
| Walker        | Bipedal walking           | Intermediate | Compare with PyBullet's Walker2d              |
| PushBlock     | Push blocks               | Introductory | Goal-conditioned RL                           |
| FoodCollector | Collect food              | Intermediate | Multi-objective + navigation                  |
| HideAndSeek   | Multi-agent hide-and-seek | Advanced     | Multi-agent emergent behavior                 |

See the [Environment Setup Guide](../preface/env-setup) for installation and environment access.

#### Classic Milestone Project Reference

Below are 30 common game and simulation reproduction directions from the non-LLM era, organized by theme:

##### Classic/Board Games

| #   | Name         | Game/Environment | Year | Key Information                                                            |
| --- | ------------ | ---------------- | ---- | -------------------------------------------------------------------------- |
| 1   | TD-Gammon    | Backgammon       | 1992 | Gerald Tesauro; reached human expert level through self-play RL            |
| 2   | Deep Blue    | Chess            | 1997 | IBM; defeated world champion Kasparov; primarily search-based, not pure RL |
| 3   | AlphaGo      | Go               | 2016 | DeepMind; RL + MCTS defeated Lee Sedol                                     |
| 4   | AlphaGo Zero | Go               | 2017 | No human game records; learned from self-play alone                        |
| 5   | AlphaZero    | Go/Chess/Shogi   | 2017 | Universal board-game RL algorithm; mastered three games simultaneously     |
| 6   | MuZero       | Go/Chess/Atari   | 2020 | No explicit game rules needed; simultaneously learns model and policy      |

##### Atari Series

| #   | Name                             | Game/Environment | Year | Key Information                                                                 |
| --- | -------------------------------- | ---------------- | ---- | ------------------------------------------------------------------------------- |
| 7   | DQN (Playing Atari with Deep RL) | Atari 2600       | 2013 | First to use deep RL to learn multi-game policies directly from pixels          |
| 8   | Human-level Control through DRL  | Atari 2600       | 2015 | Nature 2015; improved DQN reaching human-level on multiple Atari games          |
| 9   | Prioritized Experience Replay    | Atari            | 2015 | Improved experience replay; prioritizes high TD-error experiences               |
| 10  | Rainbow DQN                      | Atari            | 2017 | Integrates Double DQN, Dueling, PER, NoisyNet, Distributional RL, n-step return |
| 11  | IQN (Implicit Quantile Networks) | Atari            | 2018 | Distributional RL; learns quantile representations of return distributions      |

##### RTS / MOBA

| #   | Name                                      | Game/Environment | Year | Key Information                                                                 |
| --- | ----------------------------------------- | ---------------- | ---- | ------------------------------------------------------------------------------- |
| 12  | SC2LE (StarCraft II Learning Environment) | StarCraft II     | 2017 | DeepMind provides SC2 RL research environment and benchmarks                    |
| 13  | AlphaStar                                 | StarCraft II     | 2019 | Multi-agent RL reaching Grandmaster level                                       |
| 14  | TStarBot                                  | StarCraft II     | 2019 | Tencent's StarCraft II agent system                                             |
| 15  | OpenAI Five                               | Dota 2           | 2019 | 5v5 defeated world champion OG; large-scale distributed RL                      |
| 16  | Honor of Kings 1v1                        | Honor of Kings   | 2020 | Tencent AI Lab; dual-clipped PPO; mastered complex operation control            |
| 17  | Honor of Kings 5v5                        | Honor of Kings   | 2020 | Multi-hero, multi-role, global cooperation MOBA AI system                       |
| 18  | Honor of Kings Arena                      | Honor of Kings   | 2022 | Open MOBA RL environment; focuses on generalization challenges                  |
| 19  | Mini Honor of Kings                       | Honor of Kings   | 2024 | Lightweight MARL environment; suitable for personal devices and course projects |

##### FPS / 3D Games

| #   | Name                              | Game/Environment | Year | Key Information                                                          |
| --- | --------------------------------- | ---------------- | ---- | ------------------------------------------------------------------------ |
| 20  | Playing FPS Games with Deep RL    | ViZDoom          | 2016 | Deep RL for FPS games with visual input and partially observable states  |
| 21  | Quake III Arena: Capture the Flag | Quake III CTF    | 2019 | DeepMind; complex team cooperation and multi-agent emergent behavior     |
| 22  | Obstacle Tower                    | Unity 3D         | 2019 | Tests 3D navigation, visual generalization, and long-horizon exploration |
| 23  | Sample Efficient RL in Minecraft  | Minecraft/MineRL | 2021 | Using human demonstration data to improve sample efficiency in Minecraft |

##### Sports/Racing/Other

| #   | Name                     | Game/Environment | Year | Key Information                                                                           |
| --- | ------------------------ | ---------------- | ---- | ----------------------------------------------------------------------------------------- |
| 24  | Google Research Football | Football 11v11   | 2020 | Open-source football simulator supporting multi-agent RL research                         |
| 25  | RL in Rocket League      | Rocket League    | 2022 | High-dimensional continuous control and team cooperation in a racing-plus-football hybrid |
| 26  | Deep RL for Flappy Bird  | Flappy Bird      | 2015 | Early deep RL game practice project                                                       |

##### Multi-Agent/Comprehensive

| #   | Name                             | Game/Environment          | Year | Key Information                                                               |
| --- | -------------------------------- | ------------------------- | ---- | ----------------------------------------------------------------------------- |
| 27  | Deep RL for General Game Playing | General board games       | 2020 | Extending AlphaZero-style methods to general game playing                     |
| 28  | OpenSpiel                        | Board/card games          | 2019 | DeepMind game framework containing multiple games and classic game algorithms |
| 29  | Hide-and-Seek                    | Multi-agent hide-and-seek | 2019 | OpenAI; emergent tool use and complex strategies from multi-agent self-play   |
| 30  | Multi-Agent RL in Video Games    | Survey                    | 2025 | Covers Rocket League, Doom, Minecraft, StarCraft, Dota, MOBA directions       |

### LLM Era: Post-Training, Reasoning, Agentic, VLM, and World Models

LLM-era RL is no longer just "maximize scores in fixed environments." Actions can be text, searches, tool calls, web clicks, code patches, visual grounding, or even entire multi-step agent trajectories. Rewards expand from environment scores to preference models, rule verifiers, process rewards, unit tests, web task success rates, and multimodal grounding signals.

#### Modern and Classic Resource Quick Reference

The recommended reading order: start with classic papers and official documentation to build concepts, then pick a "small model + verifiable reward" project to run end-to-end, and finally move into distributed training, Deep Research, GUI/Computer Use, and multimodal environments.

| Direction                        | Recommended First Look                                                                                                                                                                                                                                                               | Type                           | Why It's Worth Reading                                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| RLHF / post-training classics    | [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155), [Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback), [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)                                                    | Classic papers/official docs   | Understand the basic paradigms of SFT, RM, PPO, DPO, RLAIF, and safety alignment                                     |
| Modern post-training engineering | [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL), [verl](https://github.com/verl-project/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                               | Production/research frameworks | See directly how rollout, vLLM/SGLang, Ray, Megatron, GRPO/DAPO, and async agentic RL are implemented                |
| Reasoning RLVR                   | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [DeepSeek-R1 Nature](https://www.nature.com/articles/s41586-025-09422-z), [Open-R1](https://github.com/huggingface/open-r1), [TinyZero](https://github.com/Jiayi-Pan/TinyZero)                                            | Modern reasoning reproduction  | Best for learning verifiable reward, GRPO/RLVR, cold-start data, long reasoning, and reward hacking                  |
| Open-source base models          | [Qwen3.6](https://github.com/QwenLM/Qwen3.6), [Qwen3](https://github.com/QwenLM/Qwen3), [Meta Llama Models](https://github.com/meta-llama/llama-models)                                                                                                                              | Open-source models             | Suitable for SFT/DPO/GRPO, tool calling, long context, and agentic coding experiments                                |
| Deep Research                    | [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/), [Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch), [WebThinker](https://github.com/RUC-NLPIR/WebThinker), [Search-R1](https://github.com/PeterGriffinJin/Search-R1)            | Product/open-source research   | Turn search, reading, evidence filtering, citation, and long report synthesis into trainable trajectories            |
| Agent frameworks & tool calling  | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), [Google ADK](https://github.com/google/adk-python), [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning), [AutoGen](https://github.com/microsoft/autogen)                                  | Agent engineering frameworks   | Learn engineering boundaries: tools, handoffs, guardrails, tracing, sessions, agent trajectories, and RL interfaces  |
| GUI / Computer Use               | [OpenAI CUA](https://openai.com/index/computer-using-agent/), [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool), [ByteDance UI-TARS](https://github.com/bytedance/UI-TARS), [OSWorld](https://os-world.github.io/)           | Models/tools/benchmarks        | Core materials for modern computer use: screenshots, coordinate actions, web/desktop/mobile task success rates       |
| VLM / VLA / Robotics             | [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [Open Vision Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner), [Gemini Robotics](https://deepmind.google/models/gemini-robotics/), [LeRobot](https://github.com/huggingface/lerobot)                             | Multimodal/embodied            | Connect visual QA, grounding, GUI clicks, robot actions, and verifiable rewards                                      |
| World models                     | [DreamerV3 Nature](https://www.nature.com/articles/s41586-025-08744-2), [DreamerV3 Code](https://github.com/danijar/dreamerv3), [Google DeepMind Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | Classic/frontier/simulation    | From reproducible world models to interactive world generation to parallel robot simulation                          |
| Generative model RL              | [DDPO](https://github.com/jannerm/ddpo), [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo), [AlignProp](https://align-prop.github.io/), [RLAIF-V](https://github.com/RLHF-V/RLAIF-V), [VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                | Image/video/multimodal rewards | Learn to turn aesthetics, preferences, safety, text-image consistency, or video quality into optimization objectives |

(The remaining sections — LLM Post-Training, LLM Reasoning, Deep Research RL, Agentic RL & Tool Calling, GUI & Computer Use, VLM, World Models & Simulators, Generative Model RL, Evaluation Benchmarks, and Reproduction Order — contain detailed project recommendations, common pitfalls, and resource tables that follow the same pattern as the sections above. Each subsection includes: reproduction goals, resource tables with links, and a "common pitfalls" list.)

### Evaluation Benchmarks and Projects

LLM-era RL evaluation is often more error-prone than training. A direction must simultaneously track final success rate, process quality, format constraints, reward hacking, length bias, data leakage, and multi-sample stability.

#### Acceptance Checklist

- **Final metrics**: accuracy, pass rate, task success rate, preference win rate.
- **Process metrics**: tool call count, invalid action ratio, repeated search ratio, citation accessibility rate, code test failure types.
- **Stability metrics**: effectiveness across different random seeds, sampling temperatures, and model sizes.
- **Safety metrics**: whether the model is more prone to fabricated citations, unauthorized tool calls, environment information leakage, or broken format constraints.
- **Cost metrics**: average tokens, average tool calls, average latency, training and evaluation GPU/CPU overhead.

#### Badcase Template

For each direction, maintain a `badcases.jsonl` or spreadsheet recording at minimum: task ID, input, model output, reward, scoring rationale, failure type, reproducibility, and fix suggestion. For LLM RL, badcases are not an afterthought — they are the entry point for next-round reward design, data filtering, and environment fixes.

### Reproduction Order Suggestion

First use 0.5B to 3B small models with math, code, and format verification tasks to observe reward hacking, length bias, and sampling temperature effects; then migrate from TRL/TinyZero/Open-R1 to distributed frameworks like verl/OpenRLHF. For Agentic RL, prioritize tasks with clear success rates like search, web, and code; for VLM RL, prioritize scorable tasks like image-text answers, grounding, OCR, and GUI clicks; for world models and embodied directions, first run DreamerV3/TD-MPC2, then add vision and real-robot complexity.

#### A Solid Roadmap

1. **Week 1: Rule-reward tasks**
   Use TRL or TinyZero to run a small verifiable task like Countdown, formatted JSON, or simple math. Goal: understand rollout, reward, advantage, KL, length bias, and log saving.

2. **Week 2: Preference optimization and post-training comparison**
   Use the same small model for SFT, DPO/KTO, and PPO/GRPO comparison. Don't change too many variables — just observe how different training methods affect the same batch of prompts.

3. **Week 3: Reasoning RLVR**
   Introduce Math-Verify, reasoning-gym, or code unit tests so reward evolves from "format correct" to "answer verifiable." Focus on observing reward sparsity and verifier loopholes.

4. **Week 4: Tool calling or Deep Research**
   Build a small search/reading environment and record complete trajectories. Start with offline trajectory replay, then move to online rollout.

5. **Week 5: VLM or GUI**
   Choose a visual QA, bbox grounding, or web click task and add visualized badcases. Focus on checking coordinate systems, screenshot states, and reward interpretability.

6. **Week 6+: Distributed and industrial frameworks**
   Move into verl, OpenRLHF, AReaL, SkyRL, and similar frameworks. By now you know what reward, logging, and evaluation you need — you won't be led by engineering complexity.

#### When to Increase Difficulty?

When a task meets three criteria, you can move to the next level: first, the pre/post-training difference on a fixed evaluation set is stable; second, badcases can be clearly classified; third, when reward rises, human spot-check quality also rises. Otherwise, don't rush to switch to a larger model or more complex environment — fix reward, data, and logging first.
