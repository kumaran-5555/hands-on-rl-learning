---
title: PPO Game Project Practice Guide
description: A PPO game-project selection guide from starter to advanced practice, organized by technical themes rather than vague difficulty labels.
outline:
  level: [2, 3]
---

# 7.5 PPO Game Project Practice Guide

After learning PPO's math and implementation, the next step is to put it into real game environments. The goal of this section is not to list every game that can be trained with PPO. It is to help you build a practical path from "it runs" to "I understand its boundary."

## Learning Goals

After this section, you should be able to:

1. **Judge whether a game is a good PPO starter project**: low-dimensional state, small action space, and immediate feedback are the three key signals.
2. **Separate "can run" from "can teach"**: PPO may run on a project without being the best teaching example; some projects are more valuable because they expose exploration, generalization, or reward-design issues.
3. **Choose projects by your own goal**: from Flappy Bird, which can run in a few hours, to Pokemon Red, which requires persistent debugging, every project should have a clear purpose and exit condition.

## How to Evaluate a Project

Do not choose by a vague difficulty label. Look at the combination of three **technical dimensions**:

| Dimension        | Meaning                          | Low requirement          | High requirement              |
| ---------------- | -------------------------------- | ------------------------ | ----------------------------- |
| **State space**  | What the policy observes         | low-dimensional vector   | raw pixels + frame stacking   |
| **Action space** | What the policy can do           | discrete, few actions    | continuous, multi-dimensional |
| **Reward delay** | Distance from action to feedback | resolved within one step | tens or hundreds of steps     |

Flappy Bird is low on all three dimensions and runs quickly. Pokemon Red is high on all three and has many failure modes. The following projects are discussed through these dimensions: how PPO is applied, what results to expect, and what you can learn.

---

### Flappy Bird: The Minimal Closed Loop

![Flappy Bird PPO demo, source: wangjia184/rl project](https://user-images.githubusercontent.com/44725090/67148880-e7dba280-f2a4-11e9-8dbf-d154842ee0cf.gif)

Flappy Bird is the fastest way to verify that **PPO can really learn**. The state can be reduced to a few numbers: bird height, velocity, distance to the next pipe, and pipe-gap position. There are only two actions: flap or do nothing. Failure feedback is immediate, so the policy does not need much long-term memory.

Dhyanesh18's implementation uses **Stable-Baselines3**, `CnnPolicy`, **4-frame stacking**, and an entropy-coefficient annealing callback for pixel input. For a first project, however, the low-dimensional-vector version is more instructive. A two- or three-layer MLP is enough; a learning rate around **$3 \times 10^{-4}$**, gamma **0.99**, and PPO clip **0.2** are reasonable starting points. Within the first few hundred episodes you should see the policy move from frequent crashes to stable pipe traversal. Pixel-based training is more expensive; around **10M timesteps** can produce long stable flights.

The value of this project is the full process: **environment wrapper -> policy network -> training loop -> visualization**. If the PPO implementation is broken, this environment exposes it quickly.

| Entry type           | Link                                                                                                                                               |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [Flappy Bird web game](https://flappybird.gg/)                                                                                                     |
| Training environment | [flappy-bird-gymnasium](https://pypi.org/project/flappy-bird-gymnasium/)                                                                           |
| PPO references       | [Yuanpeng-Li/Flappy-Bird-AI](https://github.com/Yuanpeng-Li/Flappy-Bird-AI), [Dhyanesh18/flappbird-rl](https://github.com/Dhyanesh18/flappbird-rl) |

---

### Snake: The First Reward-Design Lesson

![Snake PPO2 training example, source: gym-snake-rl article](https://d1k6fapei95iy6.cloudfront.net/imgs_taemin/ppo2-fullobs.gif)

Snake adds one important dimension beyond Flappy Bird: **the body changes over time**. The policy cannot only look at where the food is. It also has to understand the shape of its own body, or it will trap itself while moving toward food.

The most useful experiment here is reward design. If the only reward is **+10 for food and -10 for death**, the policy may learn to circle in place. That behavior avoids death, but it does not seek food. Time penalties or distance-based shaping are needed to push the agent toward active exploration. With Stable-Baselines PPO2, a 5x5 board can reach a maximum score around **23**, close to the theoretical optimum. On a 10x10 board, after **100M steps**, average score may still be only **6-7**. Compared with DQN, PPO can sometimes reach a higher peak score but usually converges more slowly, which shows the sample-efficiency weakness of policy-gradient methods on small discrete tasks.

The lesson is direct: the same PPO algorithm can produce completely different behavior under different reward functions.

| Entry type           | Link                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Play                 | [Snake web game](https://snaketap.com/)                                                                                  |
| Training environment | [gym-snake-rl](https://jfpettit.svbtle.com/introducing-gym-snake-rl), [Gym-Snake](https://github.com/grantsrb/Gym-Snake) |
| PPO reference        | [Introducing gym-snake-rl](https://jfpettit.svbtle.com/introducing-gym-snake-rl)                                         |

---

### 2048: The Trap of Immediate Reward

![2048 PPO training example, source: tejpshah/2048-DeepRL](https://github.com/tejpshah/2048-DeepRL/raw/main/gifs/PPO.gif)

2048 has no complex graphics, but it introduces a subtle issue: **immediate reward can mislead the policy**. If the reward is only merge score, the policy may greedily merge small tiles and quickly fill the board. The real skill is to preserve space and push large tiles into stable positions, but those choices only pay off many steps later.

In arturf1's Unity ML-Agents PPO experiments, a complex reward made from merge score, new-highest-tile bonuses, and a 2048 bonus reached only about **4%** win rate after **215k games**. A simpler reward, which only rewarded first reaching a new tile, reached about **37%** win rate after **450k games**. The result is counterintuitive: a simpler reward produced a better global strategy. In tejpshah's PyTorch PPO, one-hot board input sometimes reaches the **512 tile**, but most runs stop around **256**; DDQN in the same comparison more often reaches **1024** or **2048**.

The project is valuable because it shows that reward design, not the algorithm name, often sets the ceiling.

| Entry type           | Link                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Play                 | [Original 2048 web game](https://classic.play2048.co/)                                                             |
| Training environment | [gymnasium-2048](https://pypi.org/project/gymnasium-2048/), [2048 source](https://github.com/gabrielecirulli/2048) |
| PPO references       | [arturf1/2048](https://github.com/arturf1/2048), [tejpshah/2048-DeepRL](https://github.com/tejpshah/2048-DeepRL)   |

---

### Tetris: A Trial of Long-Term Planning

![Tetris screenshot, source: Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/5/5c/Tetris_freemade.png)

In Tetris, failure often begins many pieces before the board collapses. A single bad placement can create a hole that only becomes fatal after dozens of later decisions. The action space is discrete, but the planning horizon is long.

A recent bitboard-optimized PPO study proposed an afterstate-evaluating actor network and reached an average score around **3829** on a 10x10 board after **61,440 steps**. But transferring such policies to the standard 10x20 board remains unstable. This suggests that small-board training may learn local patterns rather than the real structure of Tetris. In practice, carefully designed heuristics and evolutionary methods often beat PPO on Tetris, making it a useful case for discussing **RL versus expert knowledge**.

Start with board matrix plus current-piece features rather than raw RGB. For reward, experiment with line clears, height penalties, hole penalties, and death penalties.

| Entry type           | Link                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [Official Tetris web game](https://play.tetris.com/)                                                                      |
| Training environment | [ALE Tetris](https://ale.farama.org/environments/tetris/), [tetris-gymnasium](https://pypi.org/project/tetris-gymnasium/) |
| PPO reference        | [chirbard/ppo-Tetris-v5](https://huggingface.co/chirbard/ppo-Tetris-v5)                                                   |

---

### Breakout: The Standard Pixel-Input Pipeline

![Breakout PPO example, source: Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/en/master/_images/breakout.gif)

Breakout is a clean first project for **playing from pixels**. The ball position, velocity, and paddle position are not explicit variables; the policy must infer them from frames. Stable-Baselines3 PPO is a strong baseline here: after **10M steps**, scores around **398 +/- 33** are typical, and **5M steps** can already approach human-level play.

The key engineering point is preprocessing. A single frame cannot reveal ball velocity, so **frame stacking** is required, usually with four frames. Grayscale conversion and image cropping reduce computation substantially. Average reward alone is not enough: watch replays to confirm that the policy is tracking the ball rather than getting lucky.

| Entry type           | Link                                                                                                                                                |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [Atari Breakout web game](https://brickbreaker.app/atari-breakout/)                                                                                 |
| Training environment | [ALE Breakout](https://ale.farama.org/environments/breakout/)                                                                                       |
| PPO references       | [sb3/ppo-BreakoutNoFrameskip-v4](https://huggingface.co/sb3/ppo-BreakoutNoFrameskip-v4), [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) |

---

### Procgen: Generalization versus Memorization

![Procgen environments, source: OpenAI procgen GitHub](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)

Procgen is not mainly about whether PPO can run. It is about **generalization**. Training and test levels are generated from different random seeds. The question is: did the policy learn the rules, or did it memorize layouts?

Start with `coinrun`. OpenAI's benchmark used an IMPALA-CNN architecture, learning rate **$5 \times 10^{-4}$**, gamma **0.999**, and reward normalization. On easy CoinRun, PPO training score can approach the maximum of 10, while test score is around **8.31 +/- 0.12**. The train-test gap is the most useful signal: a large gap means the policy overfit to level geometry rather than learning "run right, jump, avoid hazards."

| Entry type           | Link                                                                                                                                                        |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [Procgen PyPI](https://pypi.org/project/procgen/), run `python -m procgen.interactive --env-name coinrun`                                                   |
| Training environment | [OpenAI Procgen Benchmark](https://openai.com/index/procgen-benchmark/), [procgen PyPI](https://pypi.org/project/procgen/)                                  |
| PPO references       | [OpenAI Procgen Benchmark](https://openai.com/index/procgen-benchmark/), [CleanRL ppo_procgen.py](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppoprocgenpy) |

---

### CarRacing: Pixels plus Continuous Control

![CarRacing PPO run, source: Solving CarRacing with PPO](https://notanymike.github.io/img/posts/SolvingCarRacing/4.gif)

CarRacing combines pixel input with continuous control. The policy sees a **96x96** image and outputs three continuous values: steering, throttle, and brake. Preprocessing is crucial. Grayscale input can be dramatically better than raw RGB at early training, and four-frame stacking helps the policy infer road curvature and recent motion.

Training often moves through recognizable stages: around **400k-500k steps**, scores in the **450-620** range mean the car can finish much of the track; around **2M steps**, scores in the **740-920+** range approach strong driving. A2C often fails badly here, which makes CarRacing a good demonstration of PPO's clipped surrogate objective in continuous control.

This project is a natural follow-up after BipedalWalker. It is still continuous control, but the observation channel now looks like a game.

| Entry type           | Link                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [Gymnasium CarRacing docs](https://gymnasium.farama.org/environments/box2d/car_racing/)                                                                  |
| Training environment | [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)                                                                    |
| PPO references       | [Solving CarRacing with PPO](https://notanymike.github.io/Solving-CarRacing/), [Rinnnt/ppo-CarRacing-v3](https://huggingface.co/Rinnnt/ppo-CarRacing-v3) |

---

### Super Mario Bros: Reward Design Sets the Policy Ceiling

![Super Mario Bros PPO demo, source: vietnh1009/Super-mario-bros-PPO-pytorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/raw/master/demo/video-1-1.gif)

Mario is a classic project, but its reward is easy to write incorrectly. Rewarding only rightward movement can make the policy rush into danger. Rewarding only level completion makes exploration too sparse. vietnh1009's PyTorch implementation uses a CNN Actor-Critic, frame skipping, grayscale conversion, 84x84 resizing, and 4-frame stacking. Harder levels require smaller learning rates, such as **$7 \times 10^{-5}$**, for stable convergence.

Reported results reach **31/32 levels**, significantly better than the same author's earlier A3C result of **19/32**. The comparison shows why PPO's clipped objective matters: under the same environment, PPO remained stable enough to clear many more levels.

Do not start with the full NES controller. Begin with `RIGHT_ONLY` or `SIMPLE_MOVEMENT`, solve World 1-1, then expand the action space.

| Entry type           | Link                                                                                                                                                                       |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [gym-super-mario-bros CLI](https://github.com/Kautenja/gym-super-mario-bros): run `gym_super_mario_bros -e SuperMarioBros-v0 -m human`                                     |
| Training environment | [Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros), [PyPI](https://pypi.org/project/gym-super-mario-bros/)                                  |
| PPO references       | [vietnh1009/Super-mario-bros-PPO-pytorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch), [super-mario-agent](https://github.com/nemanja-m/super-mario-agent) |

---

### Sonic / Gym Retro: The Value of Transfer Learning

![Sonic / Retro Contest screenshot, source: OpenAI Retro Contest Results](https://images.ctfassets.net/kftzwdyauwt9/1gGJCzTHFTNJn6m4GkROAB/6cb51a280c362510874d3ee15f5da6e8/retro-contest-results.jpg?w=3840&q=90&fm=webp)

The **Retro Contest 2018** focused on generalization: train on one set of levels and evaluate on unseen levels. Human average score was about **7438**, while the official Joint PPO baseline reached only **3128**.

The winning solution used a key technique: **transfer learning**. It pretrained on training levels and then used PPO to fine-tune on test levels, raising the score to **4692**. That shows both the strength and the limit of PPO: fine-tuning is useful, but cross-level generalization remains far below human play.

PPO often fails on unfamiliar terrain hazards rather than because the low-level controls are impossible. The network has learned visual patterns from training levels, not abstract game rules.

| Entry type           | Link                                                                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [Gym Retro interactive script](https://retro.readthedocs.io/en/latest/getting_started.html), legal ROM required                           |
| Training environment | [Gym Retro docs](https://retro.readthedocs.io/en/latest/), [OpenAI Gym Retro](https://openai.com/research/gym-retro)                      |
| PPO references       | [OpenAI Retro Contest](https://openai.com/index/retro-contest/), [Retro Contest Results](https://openai.com/index/retro-contest-results/) |

---

### Unity SoccerTwos: Multi-Agent Self-Play

![Unity SoccerTwos PPO example, source: Hugging Face Deep RL Course](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit10/soccertwos.gif)

SoccerTwos is 2v2 soccer, so it introduces **multi-agent learning**. PPO is usually paired with **self-play**: agents play against historical versions of themselves, and progress is tracked with ELO. Unity ML-Agents provides the most direct PPO implementation path.

The training process often shows strategy evolution. Early agents chase the ball chaotically. As ELO rises from **1200** toward **1600**, simple attacker-defender role separation can appear. SAC tends to perform much worse in this setting, which highlights PPO's stability in competitive multi-agent training.

Curriculum learning matters. Gradually changing start positions and ball speed can reduce training time, but a poorly designed curriculum can hurt final performance. Multi-agent reward design is also delicate: rewarding only goals is sparse; over-rewarding distance to the ball makes every agent crowd the same location.

| Entry type           | Link                                                                                                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [Hugging Face Space: ML-Agents SoccerTwos](https://huggingface.co/spaces/unity/ML-Agents-SoccerTwos)                                                                            |
| Training environment | [Unity ML-Agents docs](https://unity-technologies.github.io/ml-agents/Training-ML-Agents/), [HF Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit7/introduction) |
| PPO references       | [blu666/ppo-SoccerTwos](https://huggingface.co/blu666/ppo-SoccerTwos), [Sekiraw/SoccerTwos](https://huggingface.co/Sekiraw/SoccerTwos)                                          |

---

### ViZDoom: Partial Observability

![ViZDoom PPO run, source: GuillBla/RL-Doom](https://github.com/GuillBla/RL-Doom/raw/master/demos/demo_basic.gif)

First-person view introduces **partial observability**. The current frame is not the full state: an enemy may be behind the agent, and supplies may be behind a wall. GuillBla's implementation uses Stable-Baselines3 `CnnPolicy`, grayscale conversion, resize to **100x160**, and cropping of the bottom UI area.

The scenarios form a useful progression. **Basic** has three actions and can be learned after about **100k steps**. **Defend** is harder but learnable after about **200k steps**. **Deadly Corridor** uses seven actions and requires reward shaping and curriculum learning; even then, results can remain unstable.

Deadly Corridor is instructive because it shows that complex 3D navigation plus a larger action space is not solved by hyperparameter tuning alone. LSTM memory helps partial observability, but it cannot fully replace missing state information.

| Entry type           | Link                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [ViZDoom default scenarios](https://vizdoom.farama.org/environments/default/), [Freedoom](https://github.com/freedoom/freedoom)    |
| Training environment | [ViZDoom docs](https://vizdoom.farama.org/), [ViZDoom default scenarios](https://vizdoom.farama.org/environments/default/)         |
| PPO references       | [GuillBla/RL-Doom](https://github.com/GuillBla/RL-Doom), [callumhay/vizdoom_ppo_rnd](https://github.com/callumhay/vizdoom_ppo_rnd) |

---

### Pokemon Red: Extreme Delay and Engineering Compensation

![Pokemon Red RL exploration screenshot, source: PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments/raw/master/assets/grid.png?raw=true)

Pokemon Red combines map transitions, story flags, menus, dialog, battles, and long exploration. Reward delay is far beyond the previous examples. PWhiddy's baseline uses **PPO + PyBoy**. The V2 version replaces KNN frame-similarity rewards with coordinate-exploration rewards and successfully reaches **Cerulean City**.

Two failure modes are common. The first is **action loops**: the policy repeats a path because it still receives exploration reward. The second is **menu spam**: the agent opens and closes menus because some button presses have no immediate negative consequence. PokeRL adds a loop-aware environment wrapper and anti-spam mechanism, then targets earlier milestones such as leaving the bedroom, reaching Viridian City, and winning the first battle.

The lesson is important: when rewards are extremely sparse, PPO's boundary appears quickly. The solution often requires environment wrappers and task decomposition, not only hyperparameter tuning.

| Entry type           | Link                                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Play                 | [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments), legal game file required                                                          |
| Training environment | [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments), [davidpaulius/PokemonRedRL](https://github.com/davidpaulius/PokemonRedRL) |
| PPO references       | [PokeRL project page](https://drubinstein.github.io/pokerl/), [PokeRL paper](https://arxiv.org/abs/2604.10812)                                               |

---

### Crafter: PPO's Exploration Bottleneck

![Crafter environment screenshot, source: danijar/crafter GitHub](https://raw.githubusercontent.com/danijar/crafter/main/media/video.gif)

Crafter includes survival, gathering, crafting, sleeping, combat, and **22 hierarchical achievements**, but the engineering burden is far smaller than Minecraft. Evaluation uses the geometric mean achievement success rate, which rewards broad progress rather than repeating one easy behavior. Human experts score around **50.5%**. Standard Stable-Baselines3 CNN PPO scores around **4.6%**; replacing the network with a deeper Impala ResNet can reach around **15.6%**.

The achievement distribution reveals the bottleneck. Basic survival achievements such as drinking water and finding food are reachable. Mid-level achievements such as stone tools sometimes appear. Long-horizon achievements such as iron and diamond tools are mostly unreachable. The issue is not simply network capacity: PPO learns quickly from data it sees, but it struggles to discover goals that require 10 or more precise steps before the reward appears.

| Entry type           | Link                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Play                 | [danijar/crafter](https://github.com/danijar/crafter): run `python3 -m crafter.run_gui`                            |
| Training environment | [Crafter GitHub](https://github.com/danijar/crafter), [Crafter project page](https://danijar.com/project/crafter/) |
| PPO reference        | [Benchmarking and Improving RL Generalization with Crafter](https://arxiv.org/abs/2208.03374)                      |

---

### MiniHack / NetHack: The Upper Limit of Long-Horizon Credit Assignment

![MiniHack / NetHack subtask example, source: hr0nix/omega](https://github.com/hr0nix/omega/raw/main/images/river.gif)

MiniHack decomposes NetHack into subtasks such as crossing rivers, pushing boulders, opening doors, finding keys, and fighting monsters. Its observations are symbolic ASCII/Unicode grids rather than pixels, so the policy network often uses CNNs or Transformers over symbols.

On the full NetHack Learning Environment, Sample Factory's high-throughput PPO reaches a NetHackScore around **700** with **480 parallel environments** in 24 hours on a single RTX 2080Ti, outperforming the original TorchBeast baseline. But on MiniHack subtasks, PPO can saturate quickly. Adding intrinsic motivation methods such as RND, NovelD, or E3B often does not reliably beat extrinsic-reward PPO. That suggests the issue is not merely "not enough exploration"; PPO's long-horizon credit assignment has a real ceiling.

In later research, the direction is often not "improve PPO alone," but add hierarchical options or stronger exploration mechanisms on top of PPO. In this setting, PPO is a reliable lower bound that new methods must beat.

| Entry type           | Link                                                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Play                 | [MiniHack trying out](https://minihack.readthedocs.io/en/latest/getting-started/trying_out.html): run `python -m minihack.scripts.play_gui --env MiniHack-River-v0` |
| Training environment | [MiniHack docs](https://minihack.readthedocs.io/), [MiniHack GitHub](https://github.com/facebookresearch/minihack)                                                  |
| PPO reference        | [hr0nix/omega: PPO and MuZero agents](https://github.com/hr0nix/omega)                                                                                              |

---

## Quick Index

| Game               | Core topic                                   | Typical PPO result                                      |
| ------------------ | -------------------------------------------- | ------------------------------------------------------- |
| Flappy Bird        | verifying that PPO learns                    | stable flight; vector-state version converges quickly   |
| Snake              | reward design controls behavior              | 5x5 max around 23; 10x10 average around 6-7             |
| 2048               | immediate reward versus long-term layout     | simple reward can reach 37% win rate; DQN often better  |
| Tetris             | delayed feedback, RL versus expert knowledge | 10x10 around 3800; standard board remains hard          |
| Breakout           | standard pixel preprocessing pipeline        | about 398 after 10M steps, near human level             |
| Procgen            | generalization versus memorization           | CoinRun test around 8.31 +/- 0.12; train nearly perfect |
| CarRacing          | continuous-control stability                 | 2M steps around 740-920+; A2C around -90                |
| Super Mario Bros   | reward design sets ceiling                   | 31/32 levels, much better than A3C                      |
| Sonic / Gym Retro  | transfer learning                            | winning score 4692; human level around 7438             |
| Unity SoccerTwos   | multi-agent self-play                        | ELO around 1600; curriculum can accelerate training     |
| ViZDoom            | partial observability                        | Basic/Defend succeed; Corridor remains hard             |
| Pokemon Red        | extreme delay and wrappers                   | reaches Cerulean City; needs anti-loop wrappers         |
| Crafter            | exploration bottleneck                       | CNN around 4.6%; ResNet around 15.6%                    |
| MiniHack / NetHack | long-horizon credit assignment limit         | NetHackScore around 700; many subtasks saturate         |

---

## Choosing by Need

**If you only want a video-friendly demo**, choose Flappy Bird, Snake, 2048, Breakout, or CarRacing. They train relatively quickly and produce clear visual results.

**If you want a more visually interesting interface**, choose Procgen, Mario, SoccerTwos, ViZDoom, or Crafter.

**If you want handheld or retro-console tasks**, choose early Pokemon Red milestones, Super Mario Bros World 1-1, or short Sonic / Gym Retro levels. Handle simulator and licensing issues carefully; do not set "finish the whole game" as the first target.

**If you want a research-flavored project**, choose Crafter, MiniHack, or Pokemon Red. These tasks are well suited for discussing PPO's boundaries: delayed reward, incomplete state, complex action semantics, and exploration traps.

---

## Reflection Questions

1. Why can Flappy Bird have such a small state space while still producing intelligent-looking behavior? What does that say about state representation quality?
2. In Snake, if the reward is only "+10 for food, -10 for death," what unintended behavior might the policy learn? How could reward design prevent it?
3. Breakout needs frame stacking, but 2048 does not. What information is unavailable in a single frame?
4. Procgen and Retro Contest both test generalization, but in different ways. What are the strengths and weaknesses of procedurally generated levels versus fixed unseen levels?
5. Pokemon Red's extreme reward delay points to what fundamental limitation of PPO and policy-gradient methods? What algorithmic mechanisms might help?

---

## Summary

PPO is a general policy-optimization framework, but "general" does not mean "equally good on every task." The projects in this section build intuition for PPO's boundaries:

- **Low-dimensional, immediate-feedback** tasks are fast and stable.
- **Pixel input** requires preprocessing work, but it does not change the algorithmic core.
- **Continuous control** requires careful hyperparameters, especially learning rate and entropy regularization.
- **Delayed reward, partial observability, and multi-agent settings** reach PPO's design boundary; failures there usually require algorithmic extensions, not only tuning.

When choosing a project, do not ask only whether the game looks impressive. Ask which side of PPO it will help you understand.
