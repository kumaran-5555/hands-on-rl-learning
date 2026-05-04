<div align="center">
  <img src="docs/public/readme/logo.png" alt="Hands-On Modern RL Logo" width="500" />
  <h1>Hands-On Modern RL</h1>
  <p><strong>现代强化学习实战课程</strong></p>
  <p><em>现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体。</em></p>

  <p>
    <a href="https://walkinglabs.github.io/hands-on-modern-rl/"><img src="https://img.shields.io/badge/Course-Online-2563eb?style=flat-square" alt="Online Course" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-111827?style=flat-square" alt="CC BY-NC-SA 4.0 License" /></a>
    <img src="https://img.shields.io/badge/Node-%3E%3D18-16a34a?style=flat-square" alt="Node >= 18" />
    <img src="https://img.shields.io/badge/Docs-VitePress-646cff?style=flat-square" alt="VitePress" />
  </p>

  <p>
    <a href="README.md">中文</a> ·
    <a href="README.en.md">English</a>
  </p>

  <p>
    <a href="#wechat-group">💬 WeChat Group</a>
  </p>

  <p>
    <a href="#课程简介">课程简介</a> ·
    <a href="#简介">简介</a> ·
    <a href="#🔥-最新动态-news">最新动态</a> ·
    <a href="#目录">目录</a> ·
    <a href="#课程大纲">课程大纲</a> ·
    <a href="#实验代码">实验代码</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#参与贡献">参与贡献</a>
  </p>
</div>

## 课程简介

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-learning-path.png" alt="课程学习地图截图" width="100%" />
      <br />
      <strong>一眼看懂的学习地图</strong>
      <br />
      <sub>从前言、基础导论到前沿专题，章节树和页内大纲帮助你快速定位。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-code-focus.png" alt="PPO 代码聚焦截图" width="100%" />
      <br />
      <strong>代码逐行聚焦</strong>
      <br />
      <sub>PPO、DPO、GRPO 关键实现配有代码地图，把公式落到可读代码。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-training-metrics.png" alt="CartPole 训练指标截图" width="100%" />
      <br />
      <strong>训练指标可视化</strong>
      <br />
      <sub>真实曲线、指标解释和失败信号放在一起，方便边跑实验边诊断。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-rlhf-pipeline.png" alt="RLHF 流水线截图" width="100%" />
      <br />
      <strong>LLM 后训练流水线</strong>
      <br />
      <sub>RLHF、DPO、GRPO、RLVR 等主题以流程、artifact 和案例串联起来。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-agentic-rl.png" alt="Agentic RL 项目截图" width="100%" />
      <br />
      <strong>Agentic RL 项目化</strong>
      <br />
      <sub>工具调用、轨迹合成、评测与多工具 Code Agent 走向完整工程练习。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-vlm-rl.png" alt="VLM 强化学习截图" width="100%" />
      <br />
      <strong>多模态与前沿方向</strong>
      <br />
      <sub>VLM 强化学习、视觉生成 RL、具身智能和未来趋势延伸到前沿系统。</sub>
    </td>
  </tr>
</table>

---

> [!NOTE]
> 希望本开源教程能够让更多人拥有向智能上限发起攀登的勇气，解决更多通往 AGI 道路上的问题。
>
> 当前教程快速迭代中。建议只看非 🚧 状态的章节，🚧状态的章节很可能有错误，也欢迎修正和建议 。

> **寻求帮助**
>
> 由于资源稀缺问题，我们正在寻求显卡支持，如果您有显卡使用方式愿意支持非常欢迎联系 physicoada@gmail.com。

## 目录

- [课程简介](#课程简介)
- [目录](#目录)
- [简介](#简介)
  - [设计原则](#设计原则)
  - [目标受众](#目标受众)
  - [学习目标](#学习目标)
  - [当前状态](#当前状态)
- [🔥 最新动态 (News)](#-最新动态-news)
- [🗺️ 演进路线图 (Roadmap)](#️-演进路线图-roadmap)
- [课程大纲](#课程大纲)
  - [前言](#前言)
  - [第一部分：基础导论](#第一部分基础导论)
  - [第二部分：核心理论与方法](#第二部分核心理论与方法)
  - [第三部分：大模型 RL](#第三部分大模型-rl)
  - [第四部分：前沿与高级系统](#第四部分前沿与高级系统)
  - [附录](#附录)
- [实验代码](#实验代码)
- [推荐学习路径](#推荐学习路径)
- [快速开始](#快速开始)
  - [在线阅读](#在线阅读)
  - [本地运行文档网站](#本地运行文档网站)
  - [验证网站](#验证网站)
  - [运行课程代码](#运行课程代码)
- [仓库结构](#仓库结构)
- [开发命令](#开发命令)
- [参与贡献](#参与贡献)
- [其他课程](#其他课程)
- [WeChat Group](#wechat-group)
- [引用](#引用)
- [开源协议](#开源协议)

## 简介

**Hands-On Modern RL** 是一门面向现代强化学习实践的开放课程。与传统的“先讲公式，再给黑盒 API”不同，本课程采用 **“实践优先”** 的路径：从一行行可运行的代码和直观的训练现象出发，让学习者先看到智能体如何在环境中试错并从奖励中改进行为，再回头深入剖析其背后的状态、价值函数、策略梯度、奖励建模与信用分配等核心数学结构。

课程内容跨越经典控制理论，直接连接到当前最前沿的 AI 进展，包括大语言模型（LLM）后训练、偏好对齐（DPO/GRPO）、可验证奖励（RLVR）、多轮工具调用的 Agentic RL 以及视觉语言模型（VLM）强化学习等核心主题。

我们希望为你铺设一条坚实的阶梯——从解出 CartPole 的第一步，一直通往构建大模型后训练与智能体系统的前沿实践。

### 设计原则

课程围绕以下工程和教学原则组织：

1. **实践先于形式化。** 每个主要主题都从实验、指标、失败案例或实现细节开始，然后再引入数学抽象。
2. **理论用于解释行为。** MDP、贝尔曼方程、策略梯度、GAE、PPO 截断、DPO 目标和 GRPO 风格的组优势，都是作为解释代码行为的工具引入的。
3. **现代强化学习，不止于经典强化学习。** 课程涵盖经典控制和深度强化学习，然后进入 RLHF、偏好优化、RLVR、VLM 强化学习和多轮智能体训练。
4. **将调试能力视为一等公民。** 训练崩溃、奖励破解（Reward hacking）、KL 漂移、熵衰减、OOM 故障和评估盲区被视为核心内容，而不是补充说明。
5. **可读的系统优于黑盒。** 代码示例倾向于显式的实现、可检查的指标和清晰的实验边界，以便学习者可以修改和扩展它们。

### 目标受众

本课程专为希望通过构建和检查工作系统来理解强化学习的学习者而设计。

它特别适合：

- 从监督学习转向强化学习的机器学习工程师；
- 准备阅读现代强化学习和对齐论文的研究人员和学生；
- 希望了解 RLHF、DPO、GRPO、RLVR 和后训练系统的大语言模型（LLM）从业者；
- 工具使用智能体、Web 智能体、代码智能体和评估流水线的构建者；
- 喜欢在密集的公式推导前先看代码、实验和直观可视化的自主学习者。

推荐背景：

- Python 编程经验；
- 基础的 PyTorch 熟练度；
- 了解入门机器学习级别的线性代数、概率论和微积分；
- 能够阅读论文并追踪开源训练脚本。

课程附带了数学基础复习附录，因此不要求从第一天起就完全精通数学。

### 学习目标

完成本课程后，学习者应能够：

- 实现并解释核心的强化学习循环：环境交互、轨迹收集、奖励反馈、策略更新和评估；
- 将 MDP、价值函数、贝尔曼方程、TD 学习、策略梯度和优势估计与具体的训练行为联系起来；
- 阅读并修改 DQN、REINFORCE、Actor-Critic、PPO、DPO、GRPO 及相关实现；
- 推理大模型（LLM）的后训练流水线，包括 SFT、奖励建模、PPO 风格的 RLHF、DPO 系列方法和可验证奖励（RLVR）训练；
- 理解多轮交互与信用分配，构建工具调用、轨迹合成与 Agentic RL 智能体系统；
- 将强化学习延伸到 VLM（视觉语言模型）、具身智能与多智能体自我博弈等前沿领域；
- 诊断常见的强化学习失败模式，为新的 RL 问题设计合理的算法、工程评测与调试方案。

### 当前状态

本仓库是一个活跃的课件项目。课程内容正在逐章扩展和完善，重点关注正确性、可运行的示例和稳定的学习路径。

- 课程网站: [walkinglabs.github.io/hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
- 源码内容: [`docs/`](docs/)
- 可运行示例: [`code/`](code/)
- 本地验证: `npm run verify`
- 开源协议: [CC BY-NC-SA 4.0](LICENSE)

欢迎提交 Issue 和 Pull Request 来修复拼写错误、修正概念、改进可复现性、补充参考文献以及在合理范围内的课程扩展。

## 🔥 最新动态 (News)

> ⚠️ **备注**：本教程由于有 AI 协助生成，目前尚未全面审稿结束，很有可能会有事实性或代码不可运行的错误。欢迎大家在阅读过程中提交 Issue 或 PR 帮助指正。

- **[2026-05-02]** 🎉 教程初期浏览版正式开源发布，开放测试与建议收集。

## 🗺️ 演进路线图 (Roadmap)

本课程正在持续迭代中，以下是接下来的开发计划：

- [x] **2026-05-02**：开源初始浏览版，用于收集社区测试反馈。
- [ ] **2026-05-10**：发布正式小版本，修正初版笔误，稳定第一部分（基础）与第二部分（核心理论）的内容与代码。
- [ ] **2026-05 下旬**：完善大模型强化学习可复现实验，补充完整的 RLVR（可验证奖励）实战与评估。
- [ ] **2026-06 上旬**：分步骤交付 Agentic RL 动手项目（从单工具调用到 Deep Research 复杂轨迹合成）。
- [ ] **2026-06 下旬**：增加基于 Unity 的具身强化学习（Embodied RL）可训练模型环境与项目。
- [ ] **2026-07 及以后**：扩展多模态前沿，补充 VLM 强化学习或 Diffusion RL 的完整实战案例。

## 课程大纲

课程分为四个部分以及附录。在线网站包含全文、图表、代码参考和章节导航。

### 前言

| 主题                                                | 描述                                     |
| :-------------------------------------------------- | :--------------------------------------- |
| [课程导读](docs/preface/intro.md)                   | 课程定位、学习路径及资料使用说明。       |
| [强化学习简史](docs/preface/brief-history/index.md) | 从试错学习到 AlphaGo、RLHF 和 LLM 对齐。 |
| [环境安装指南](docs/preface/env-setup.md)           | 课程所需环境与依赖的安装配置指南。       |

### 第一部分：基础导论

| 章节 | 主题                                                                | 核心问题                                                               |
| :--- | :------------------------------------------------------------------ | :--------------------------------------------------------------------- |
| 01   | [CartPole 倒立摆](docs/chapter01_cartpole/intro.md)                 | 在真实环境中，状态、动作、奖励、策略、价值、熵和训练曲线是什么？       |
| 1.1  | [状态、动作、奖励与策略](docs/chapter01_cartpole/principles.md)     | 一个 RL 问题由哪些基本对象组成？                                       |
| 1.2  | [奖励、熵、Value Loss 与 KL](docs/chapter01_cartpole/metrics.md)    | 训练曲线里的关键指标分别反映什么？                                     |
| 02   | [DPO 偏好微调](docs/chapter02_dpo/intro.md)                         | 偏好优化如何改变模型行为，损失、奖励裕度（margin）和准确率意味着什么？ |
| 2.1  | [Post-Training 流水线与 DPO 推导](docs/chapter02_dpo/principles.md) | DPO 如何从偏好数据和参考模型中得到训练目标？                           |
| 2.2  | [Loss、Reward Margin 与 Accuracy](docs/chapter02_dpo/metrics.md)    | DPO 训练中的损失、奖励差和偏好准确率如何解读？                         |
| 总结 | [本篇小结](docs/summaries/part1-summary.md)                         | 在进入形式化理论之前，应该建立哪些直观认识？                           |

### 第二部分：核心理论与方法

| 章节 | 主题                                                                           | 核心问题                                                        |
| :--- | :----------------------------------------------------------------------------- | :-------------------------------------------------------------- |
| 03   | [MDP 与价值函数](docs/chapter03_mdp/intro.md)                                  | 老虎机、MDP、价值函数、贝尔曼方程和 TD 误差如何形式化序贯决策？ |
| 3.1  | [两台老虎机：RL 的最小问题](docs/chapter03_mdp/bandit.md)                      | 最简单的试错问题如何体现探索与利用？                            |
| 3.2  | [MDP：RL 的形式化框架](docs/chapter03_mdp/mdp.md)                              | 状态、动作、转移、奖励和折扣如何构成序列决策模型？              |
| 3.3  | [V(s) 与贝尔曼方程](docs/chapter03_mdp/value-bellman.md)                       | 如何用价值函数递归评估一个局面？                                |
| 3.4  | [DP、MC、TD](docs/chapter03_mdp/dp-mc-td.md)                                   | 动态规划、蒙特卡洛和时序差分分别怎样学习价值？                  |
| 3.5  | [Q(s, a)](docs/chapter03_mdp/value-q.md)                                       | 动作价值如何把“局面好不好”变成“动作该不该选”？                  |
| 3.6  | [策略目标 J(theta)](docs/chapter03_mdp/policy-objective.md)                    | 直接优化策略时，目标函数到底在最大化什么？                      |
| 3.7  | [算法数据来源](docs/chapter03_mdp/algorithm-taxonomy.md)                       | On-policy、off-policy 与数据来源如何影响算法设计？              |
| 3.8  | [Reward Shaping](docs/chapter03_mdp/reward-design.md)                          | 奖励函数如何引导学习，又可能怎样被误用？                        |
| 3.9  | [本章总结](docs/chapter03_mdp/panorama.md)                                     | MDP 章节里的概念如何连成一张算法地图？                          |
| 04   | [Q-Learning 与 DQN](docs/chapter04_dqn/intro.md)                               | 为什么经验回放、目标网络、CNN 编码器及 DQN 系列扩展很重要？     |
| 4.1  | [动手：Q-Learning 与 GridWorld](docs/chapter04_dqn/q-learning.md)              | 如何从表格 Q 学习开始亲手更新价值？                             |
| 4.2  | [从表格 Q 到 DQN](docs/chapter04_dqn/from-q-to-dqn.md)                         | 神经网络如何替代表格来近似 Q 函数？                             |
| 4.3  | [Replay、Target 与 CNN](docs/chapter04_dqn/dqn-components.md)                  | 经验回放、目标网络和编码器分别稳定了什么？                      |
| 4.4  | [训练过程分析](docs/chapter04_dqn/training-analysis.md)                        | DQN 训练曲线和 Q 值变化暴露了哪些学习现象？                     |
| 4.5  | [Mountain Car 与稀疏奖励](docs/chapter04_dqn/mountain-car.md)                  | 奖励很少出现时，探索和初始化为什么变得关键？                    |
| 4.6  | [Double、Dueling 与 Rainbow](docs/chapter04_dqn/dqn-family.md)                 | DQN 家族如何逐步修复高估、表示和采样问题？                      |
| 4.7  | [项目：DQN 实战与视觉游戏](docs/chapter04_dqn/visual-game-projects.md)         | 从低维控制到视觉游戏时，DQN 工程要改什么？                      |
| 05   | [策略梯度与 REINFORCE](docs/chapter05_policy_gradient/intro.md)                | 如何直接优化策略，为什么基线（baselines）能减少梯度方差？       |
| 5.1  | [动手：摇骰子赌博机](docs/chapter05_policy_gradient/dice-game.md)              | 如何用最小实验理解策略梯度的采样更新？                          |
| 5.2  | [策略梯度与 REINFORCE](docs/chapter05_policy_gradient/policy-gradient.md)      | REINFORCE 如何把高回报动作的概率推高？                          |
| 5.3  | [动手：Baseline 降方差](docs/chapter05_policy_gradient/baseline-experiment.md) | Baseline 为什么不改变期望，却能让梯度更稳？                     |
| 06   | [Actor-Critic](docs/chapter06_actor_critic/intro.md)                           | Actor 和 Critic 如何分担学习问题，TD 误差如何转化为优势信号？   |
| 6.1  | [优势函数](docs/chapter06_actor_critic/advantage-function.md)                  | 优势函数如何回答“这个动作比平均水平好多少”？                    |
| 6.2  | [TD 误差训练 Critic](docs/chapter06_actor_critic/critic-training.md)           | Critic 如何用自举信号学习价值估计？                             |
| 6.3  | [Actor-Critic 架构](docs/chapter06_actor_critic/actor-critic.md)               | Actor 与 Critic 如何在同一个训练循环中协作？                    |
| 6.4  | [项目：AlphaGo 简单复现](docs/chapter06_actor_critic/alphago.md)               | 策略网络、价值网络和搜索如何组合成棋类智能体？                  |
| 07   | [PPO](docs/chapter07_ppo/intro.md)                                             | 截断、信任区域直觉、GAE 和奖励模型如何使策略优化更稳定？        |
| 7.1  | [动手：PPO 训练 LunarLander](docs/chapter07_ppo/ppo-lunar-lander.md)           | PPO 在更复杂控制任务上如何表现和调参？                          |
| 7.2  | [PPO 数学推导](docs/chapter07_ppo/ppo-math.md)                                 | PPO 目标如何从策略梯度一步步变成裁剪代理目标？                  |
| 7.3  | [信任域与裁剪](docs/chapter07_ppo/trust-region-clipping.md)                    | 裁剪机制如何限制策略更新幅度？                                  |
| 7.4  | [GAE 与奖励模型](docs/chapter07_ppo/gae-reward-model.md)                       | GAE 如何平衡偏差与方差，并连接到奖励模型训练？                  |
| 总结 | [本篇小结](docs/summaries/part2-summary.md)                                    | 经典和现代强化学习中反复出现哪些算法模式？                      |

### 第三部分：大模型 RL

| 章节 | 主题                                                                                      | 核心问题                                                       |
| :--- | :---------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| 08   | [RLHF 全流程](docs/chapter08_rlhf/intro.md)                                               | 指令数据、奖励模型、PPO 训练、评估与扩展如何结合在一起？       |
| 8.1  | [为什么 base model 还不是 assistant](docs/chapter08_rlhf/base-model-to-assistant.md)      | 预训练模型和助手模型之间差在哪？                               |
| 8.2  | [标准 RLHF 流水线](docs/chapter08_rlhf/standard-rlhf-pipeline.md)                         | SFT、RM 和 RL 三阶段如何衔接？                                 |
| 8.3  | [SFT：教模型按指令回答](docs/chapter08_rlhf/imitation-learning-pipeline.md)               | 监督微调如何建立基础指令跟随能力？                             |
| 8.4  | [Reward Model：教一个裁判](docs/chapter08_rlhf/reward-function-design.md)                 | 奖励模型如何把人类偏好变成可优化信号？                         |
| 8.5  | [PPO-RLHF：按奖励练习](docs/chapter08_rlhf/ppo-rlhf-loop.md)                              | PPO 如何在 KL 约束下优化语言模型？                             |
| 8.6  | [评估：RLHF 到底有没有变好](docs/chapter08_rlhf/evaluation.md)                            | 如何判断对齐训练真的改善了模型？                               |
| 8.7  | [从小参数到大参数](docs/chapter08_rlhf/scaling-to-large-models.md)                        | 同一条 RLHF 流水线放大时会遇到哪些工程问题？                   |
| 8.8  | [扩展实战：Reward Hacking 与数据飞轮](docs/chapter08_rlhf/extended-practice.md)           | 如何识别奖励投机，并让数据迭代持续改进模型？                   |
| 09   | [后训练对齐](docs/chapter09_alignment/intro.md)                                           | DPO、GRPO、DeepSeek-R1、可验证奖励（RLVR）如何训练推理行为？   |
| 9.1  | [DPO、IPO 与 KTO](docs/chapter09_alignment/dpo-theory-and-family.md)                      | 偏好优化家族如何绕过显式奖励模型？                             |
| 9.2  | [动手：DPO 对齐实验](docs/chapter09_alignment/dpo-hands-on.md)                            | 如何跑通一个可检查的 DPO 训练实验？                            |
| 9.3  | [GRPO 实践与机制](docs/chapter09_grpo_rlvr/grpo-practice-and-mechanism.md)                | GRPO 如何用组内相对优势替代 Critic？                           |
| 9.4  | [DeepSeek-R1 与 DAPO](docs/chapter09_grpo_rlvr/deepseek-dapo.md)                          | 推理模型训练中有哪些新的 RL 经验？                             |
| 9.5  | [RLVR：可验证奖励](docs/chapter09_grpo_rlvr/rlvr.md)                                      | 规则可判定任务如何为 RL 提供稳定奖励？                         |
| 9.6  | [On-Policy Distillation](docs/chapter09_grpo_rlvr/on-policy-distillation.md)              | 如何把在线 RL 行为蒸馏回更可用的模型？                         |
| 10   | [Agentic RL](docs/chapter10_agentic_rl/intro.md)                                          | 多轮交互、工具调用、轨迹合成与智能体系统工程如何改变 RL 问题？ |
| 10.1 | [多轮交互与信用分配（含 ORM vs PRM 实验）](docs/chapter10_agentic_rl/multi-turn-rl.md)    | 多步任务中如何把最终结果分配回中间动作？                       |
| 10.2 | [工具调用、轨迹合成与 Agentic 工程](docs/chapter10_agentic_rl/tool-use-and-trajectory.md) | 工具执行结果如何进入 RL 轨迹和训练数据？                       |
| 10.3 | [工业实践、评测与 Badcase](docs/chapter10_agentic_rl/industrial-evaluation.md)            | Agentic RL 在工程评测中最常见的失败点是什么？                  |
| 10.4 | [项目一：多工具 Code Agent](docs/chapter10_agentic_rl/multi-tool-code-agent.md)           | 如何训练模型在搜索、写代码和测试之间切换？                     |
| 10.5 | [项目二：Deep Research Agent](docs/chapter10_agentic_rl/deep-research-agent.md)           | 深度研究型智能体如何组织搜索、引用和答案质量奖励？             |
| 10.6 | [延伸阅读](docs/chapter10_agentic_rl/extended-readings.md)                                | 继续深入 Agentic RL 应该读哪些资料？                           |
| 总结 | [本篇小结](docs/summaries/part3-summary.md)                                               | 什么让 LLM 的强化学习不同于经典环境的强化学习？                |

### 第四部分：前沿与高级系统

| 章节 | 主题                                                                                         | 核心问题                                                               |
| :--- | :------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| 11   | [VLM 强化学习](docs/chapter11_vlm_rl/intro.md)                                               | 视觉奖励、多模态框架以及视觉生成 RL 如何改变训练循环？                 |
| 11.1 | [动手：GRPO 训练 VLM](docs/chapter11_vlm_rl/vlm-grpo-hands-on.md)                            | 如何把 GRPO 训练扩展到视觉问答任务？                                   |
| 11.2 | [视觉奖励与幻觉](docs/chapter11_vlm_rl/vlm-challenges.md)                                    | 多模态奖励和视觉幻觉会带来哪些新问题？                                 |
| 11.3 | [Open-R1、R1-V 与 VLM-R1](docs/chapter11_vlm_rl/vlm-frameworks.md)                           | 前沿 VLM-RL 框架如何组织数据、奖励和训练？                             |
| 11.4 | [视觉生成 RL](docs/chapter11_vlm_rl/visual-generation-rl.md)                                 | 图像生成模型如何用偏好和奖励优化？                                     |
| 12   | [未来趋势](docs/chapter12_future_trends/intro.md)                                            | 具身智能、Model-Based RL、自我博弈、多智能体 RL 与离线 RL 的发展方向？ |
| 12.1 | [具身智能](docs/chapter12_future_trends/embodied-intelligence/index.md)                      | RL 如何进入机器人和物理世界？                                          |
| 12.2 | [Model-Based RL](docs/chapter12_future_trends/embodied-intelligence/model-based-rl/index.md) | 世界模型如何减少真实环境交互成本？                                     |
| 12.3 | [Self-Play 与自进化](docs/chapter12_future_trends/self-play-outlook/index.md)                | 自我博弈如何推动能力持续提升？                                         |
| 12.4 | [LLM 多智能体 RL](docs/chapter12_future_trends/llm-multi-agent-rl/index.md)                  | 多个语言智能体如何协作、竞争和共同学习？                               |
| 12.5 | [离线强化学习](docs/chapter12_future_trends/offline-rl/index.md)                             | 不能在线试错时，如何从固定数据中学习策略？                             |
| 12.6 | [RL Scaling 展望](docs/chapter12_future_trends/rl-scaling-outlook.md)                        | RL 的规模化训练还会走向哪里？                                          |
| 总结 | [本篇小结](docs/summaries/part4-summary.md)                                                  | 完成核心课程后，学习者应关注哪些方向？                                 |

### 附录

| 附录 | 主题                                                                             | 描述                                                   |
| :--- | :------------------------------------------------------------------------------- | :----------------------------------------------------- |
| A    | [训练调试指南](docs/appendix_common_pitfalls/intro.md)                           | 强化学习训练的失败模式、症状、根本原因和修复策略。     |
| B    | [RL 工程实践](docs/appendix_industrial_training/intro.md)                        | 训练系统底座、Agent 沙箱、评估基准及工业实战练习。     |
| B.1  | [训练系统底座](docs/appendix_industrial_training/rl-infrastructure.md)           | RL 训练系统需要哪些基础组件？                          |
| B.2  | [Agent 沙箱与工具调度](docs/appendix_industrial_training/agentic-rl-infra.md)    | 工具型智能体训练如何隔离执行环境？                     |
| B.3  | [RL 与 Agent Benchmark](docs/appendix_industrial_training/evaluation-badcase.md) | 如何设计评测并分析坏例？                               |
| B.4  | [训练指标词典](docs/appendix_industrial_training/metrics-glossary.md)            | 常见训练指标各自说明什么问题？                         |
| B.5  | [工业实战练习](docs/appendix_industrial_training/industrial-exercises.md)        | 如何把工程概念落到可练习任务里？                       |
| D    | [学习资料与项目推荐](docs/appendix_game_projects/intro.md)                       | 精选的学习资源和复现项目参考，用于扩展课程示例。       |
| E    | [数学基础](docs/appendix_math/intro.md)                                          | 强化学习相关的线性代数、概率统计、微积分优化和信息论。 |
| E.1  | [数学对象与线性代数](docs/appendix_math/linear-algebra.md)                       | 向量、矩阵和函数近似如何支撑 RL 表达？                 |
| E.2  | [概率、期望与随机估计](docs/appendix_math/probability-statistics.md)             | 回报、采样和轨迹估计依赖哪些概率工具？                 |
| E.3  | [微积分与优化](docs/appendix_math/calculus-optimization.md)                      | 梯度、链式法则和优化器如何驱动策略更新？               |
| E.4  | [信息论与分布距离](docs/appendix_math/information-theory.md)                     | 熵、交叉熵和 KL 如何解释探索与对齐约束？               |

## 实验代码

[`code/`](code/) 目录包含与各章节对齐的可运行示例。每章的代码都设计得足够精简，以便独立检查、运行和修改。

| 领域           | 代码路径                                                                                                           | 代表性实验                                                     |
| :------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| 经典控制       | [`code/chapter01_cartpole/`](code/chapter01_cartpole/)                                                             | 训练 CartPole，检查奖励和回合长度，比较 PPO 实现。             |
| 偏好微调       | [`code/chapter02_dpo/`](code/chapter02_dpo/)                                                                       | 生成偏好数据，使用 DPO 进行训练，比较微调前后的模型行为。      |
| MDP 与价值学习 | [`code/chapter03_mdp/`](code/chapter03_mdp/)                                                                       | 运行老虎机策略，求解网格世界，用数值方法验证贝尔曼更新。       |
| 深度 Q 学习    | [`code/chapter04_dqn/`](code/chapter04_dqn/)                                                                       | 实现经验回放、目标网络和 Double DQN 变体。                     |
| 策略梯度       | [`code/chapter05_policy_gradient/`](code/chapter05_policy_gradient/)                                               | 比较 REINFORCE、基线变体和 Actor-Critic 更新。                 |
| PPO            | [`code/chapter07_ppo/`](code/chapter07_ppo/)                                                                       | 训练 LunarLander，检查截断机制，可视化 GAE，并比较训练稳定性。 |
| RLHF           | [`code/chapter08_rlhf/`](code/chapter08_rlhf/)                                                                     | 走通 SFT、奖励模型训练和 PPO 风格对齐的完整流程。              |
| 对齐与 RLVR    | [`code/chapter09_alignment/`](code/chapter09_alignment/), [`code/chapter09_grpo_rlvr/`](code/chapter09_grpo_rlvr/) | 探索 DPO 奖励、GRPO 组优势和基于规则的可验证奖励。             |
| VLM 与智能体   | [`code/chapter10_agentic_rl/`](code/chapter10_agentic_rl/), [`code/chapter11_vlm_rl/`](code/chapter11_vlm_rl/)     | 构建工具调用智能体轨迹综合，实现多模态模型强化学习等。         |
| 高级主题       | [`code/chapter12_future_trends/`](code/chapter12_future_trends/)                                                   | 学习前沿方向包括多智能体强化学习、Model-Based RL等。           |

参见 [`code/README.md`](code/README.md) 获取代码索引和各章节的依赖说明。

## 推荐学习路径

本仓库的实用学习路径：

1. 阅读[课程简介](docs/preface/intro.md)并运行 CartPole 示例。
2. 尽早粗略阅读 DPO 章节（甚至在完整的理论之前），以锚定大模型后训练的动机。
3. 按顺序学习第 03-07 章；这是概念核心。
4. 在掌握策略梯度和 PPO 机制后，返回学习 RLHF、DPO、GRPO 和 RLVR。
5. 每当训练运行出现异常时，使用调试和工程附录。
6. 将前沿章节作为扩展学习：VLM 强化学习、Agentic RL、连续控制、多智能体系统和测试时推理。

## 快速开始

### 在线阅读

发布的课程网站地址：

```text
https://walkinglabs.github.io/hands-on-modern-rl/
```

### 本地运行文档网站

环境要求：

- Node.js >= 18.0.0
- npm

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

然后在浏览器中打开终端显示的本地 VitePress 服务地址，通常是：

```text
http://localhost:5173
```

### 验证网站

在提交更改文档结构、主题代码、导航、构建脚本或生成资产的 Pull Request 之前，请运行：

```bash
npm run verify
```

这会检查代码格式，Lint VitePress 主题，构建网站，并验证预期的构建产物。

### 运行课程代码

大多数代码示例基于 Python，并按章节组织。

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

对于较小的安装需求，建议使用章节特定的 requirements 文件：

```bash
pip install -r chapter01_cartpole/requirements.txt
python chapter01_cartpole/1-ppo_cartpole.py
```

某些章节可能需要额外的系统库、GPU 支持、模型下载或特定环境的设置。建议在运行涉及 LLM、VLM 或重度仿真器的示例前，先从第 01 章开始。

## 仓库结构

```text
hands-on-modern-rl/
├── docs/                      # VitePress 课程内容
│   ├── .vitepress/            # 网站配置、导航、主题覆盖
│   ├── public/                # 复制到构建后网站的静态资产
│   ├── preface/               # 课程简介和历史背景
│   ├── chapter*/              # 主要课程章节
│   ├── appendix*/             # 补充材料和参考文献
│   └── summaries/             # 部分级别的回顾和总结笔记
├── code/                      # 与章节对齐的可运行示例
├── scripts/                   # 维护和验证脚本
├── package.json               # 网站脚本和依赖
├── AGENTS.md                  # 仓库维护指南
└── README.md                  # 项目总览
```

## 开发命令

```bash
npm run dev           # 启动本地文档服务器
npm run build         # 构建静态网站
npm run preview       # 在本地预览构建后的网站
npm run format        # 使用 Prettier 格式化仓库文件
npm run format:check  # 检查代码格式
npm run lint          # Lint VitePress 主题代码
npm run verify        # 运行格式检查、Lint、构建和产物验证
```

## 参与贡献

所有的贡献都应旨在让课程更清晰、更准确、更易于复现或更易于导航。

优秀的贡献包括：

- 修复概念错误、公式、图表、失效链接或拼写错误；
- 在不改变预期学习路径的情况下改进解释说明；
- 添加能够阐明现有章节的、可复现的小型实验；
- 改进脚本、构建可靠性、导航或可访问性；
- 添加高质量论文、官方文档或广泛使用的开源实现的参考文献。

请保持 Pull Request 的聚焦。一个好的 PR 通常一次只修改一个章节、一个实验、一组图表或一个基础设施问题。

添加内容时：

1. 将课程资料放在 [`docs/`](docs/) 目录下。
2. 为新目录和文件使用 kebab-case（短横线分隔）命名。
3. 优先使用基于目录的路由（即 `index.md`）。
4. 添加可导航页面时，更新 [`docs/.vitepress/config.mjs`](docs/.vitepress/config.mjs)。
5. 当更改涉及配置、主题、脚本或生成的网站输出时，在请求 Review 之前运行 `npm run verify`。
6. 使用 Conventional Commits 规范，例如 `docs: clarify ppo clipping` 或 `fix: repair chapter link`。

有关特定于本仓库的维护规则，请参阅 [`AGENTS.md`](AGENTS.md)。

## 其他课程

我们的团队还制作了其他课程！请查看：

[![LEARN HARNESS ENGINEERING](https://img.shields.io/badge/LEARN_HARNESS_ENGINEERING-0052cc?style=for-the-badge)](https://github.com/walkinglabs/learn-harness-engineering)

## WeChat Group

有任何建议 / 反馈，欢迎扫码加入 WeChat Group 交流：

<img src="https://github.com/walkinglabs/.github/raw/main/profile/wechat.png" alt="WeChat Group" width="300" />

## 引用

如果您在教学材料、学习笔记或衍生非商业教育作品中使用本课程，请引用本仓库：

```bibtex
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```

## 开源协议

本课程资料在 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE) 下发布。

您可以出于非商业目的共享和修改本材料，前提是必须给出适当的署名，并且衍生作品也必须在相同的协议下分发。

---

<div align="center">
  <sub>由 WalkingLabs 及贡献者维护。</sub>
</div>
