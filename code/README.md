# 代码索引

本目录包含课程各章的配套代码。每章代码可独立运行。

## 快速开始

```bash
# 全局安装（推荐先创建虚拟环境）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装全局依赖（包含所有章节）
pip install -r requirements.txt

# 或者只安装某一章的依赖
pip install -r chapter01_cartpole/requirements.txt
```

## 章节代码一览

### Part 1: 极速入门

| 章节                    | 目录                  | 代码文件                        | 说明                                             |
| ----------------------- | --------------------- | ------------------------------- | ------------------------------------------------ |
| **Ch01** 传统 RL 初体验 | `chapter01_cartpole/` | `1-ppo_cartpole.py`             | SB3 版 CartPole 训练与演示                       |
|                         |                       | `2-ppo_cartpole_tensorboard.py` | 带 TensorBoard 日志的训练版本                    |
|                         |                       | `3-pytorch_from_scratch.py`     | 纯 PyTorch REINFORCE 实现（黑盒拆解）            |
|                         |                       | `requirements.txt`              | gymnasium, stable-baselines3, torch, tensorboard |
| **Ch02** 现代 RL 初体验 | `chapter02_dpo/`      | `0-download_model.py`           | 从 ModelScope 下载 Qwen2.5-0.5B-Instruct        |
|                         |                       | `1-generate_data.py`            | 偏好数据生成脚本（减少过度顺从）                 |
|                         |                       | `2-test_before.py`              | 微调前测试                                       |
|                         |                       | `3-train_dpo.py`                | DPO 训练                                         |
|                         |                       | `4-test_after.py`               | 微调后测试                                       |

### Part 2: 理论与方法

| 章节                             | 目录                         | 代码文件                     | 说明                                                 |
| -------------------------------- | ---------------------------- | ---------------------------- | ---------------------------------------------------- |
| **Ch03** MDP 与价值函数          | `chapter03_mdp/`             | `two_armed_bandit.py`        | 两臂老虎机：随机 / 贪心 / ε-贪心 / UCB 策略对比      |
|                                  |                              | `gridworld_q_learning.py`    | 4×4 GridWorld Q-Learning 与最优路径可视化            |
|                                  |                              | `bellman_equation_verify.py` | 贝尔曼方程数值验证：手动计算 vs 代码迭代对比         |
| **Ch04** 深度强化学习 DQN        | `chapter04_dqn/`             | `dqn_cartpole.py`            | 完整 DQN from scratch：Q-Network、经验回放、目标网络 |
|                                  |                              | `double_dqn_cartpole.py`     | Double DQN 改进：与普通 DQN 对比训练曲线             |
| **Ch05** 策略梯度与 Actor-Critic | `chapter05_policy_gradient/` | `reinforce_cartpole.py`      | REINFORCE 算法：策略梯度训练 CartPole                |
|                                  |                              | `reinforce_with_baseline.py` | 加入 baseline 的 REINFORCE：方差降低效果对比         |
|                                  |                              | `actor_critic_cartpole.py`   | Actor-Critic：双网络架构、TD Error 优势函数估计      |
| **Ch06** PPO 与奖励模型          | `chapter06_ppo/`             | `ppo_lunar_lander.py`        | SB3 PPO 训练 LunarLander-v3：回调监控与训练曲线      |
|                                  |                              | `ppo_from_scratch.py`        | 纯 PyTorch PPO：Clipping、GAE、多轮 epoch 更新       |
|                                  |                              | `gae_visualization.py`       | GAE 不同 λ/γ 值对比可视化：偏差-方差权衡             |

### Part 3: LLM 时代

| 章节                                     | 目录                   | 代码文件                 | 说明                                               |
| ---------------------------------------- | ---------------------- | ------------------------ | -------------------------------------------------- |
| **Ch07** 对齐方法族（DPO / KTO / SimPO） | `chapter07_alignment/` | `dpo_hands_on.py`        | DPO 对齐实战：毒性/讽刺偏好数据、β 调参对比        |
|                                          |                        | `dpo_math_reward.py`     | 数学推理场景 DPO：规则奖励、格式奖励、数据构造     |
| **Ch08** GRPO、DAPO 与 RLVR              | `chapter08_grpo_rlvr/` | `grpo_mechanism.py`      | GRPO 核心机制：组采样、组内归一化、优势计算可视化  |
|                                          |                        | `grpo_math_reasoning.py` | GRPO + GSM8K 数学推理：规则奖励函数、训练循环      |
|                                          |                        | `rule_based_reward.py`   | 可验证奖励函数：答案正确性、格式检查、推理质量打分 |

### Part 4: 进阶与前沿

| 章节                     | 目录                            | 代码文件                       | 说明                                              |
| ------------------------ | ------------------------------- | ------------------------------ | ------------------------------------------------- |
| **Ch09** 连续动作控制    | `chapter09_continuous_control/` | `sac_halfcheetah.py`           | SAC 训练 HalfCheetah-v4：熵正则化、自动温度调节   |
|                          |                                 | `ppo_td3_sac_comparison.py`    | PPO vs TD3 vs SAC 算法对比：性能曲线与特性分析    |
| **Ch10** RLHF 完整流水线 | `chapter10_rlhf/`               | `sft_pipeline.py`              | SFT 阶段：Self-Instruct 数据生成、指令微调训练    |
|                          |                                 | `reward_model_training.py`     | 奖励模型训练：Bradley-Terry 模型、偏好对评估      |
|                          |                                 | `rlhf_ppo_train.py`            | PPO 对齐训练：SFT 模型 + RM 联合训练              |
| **Ch11** VLM 强化学习    | `chapter11_vlm_rl/`             | `geometry_counting_dataset.py` | 几何图形计数数据集生成：随机形状图片 + 标注       |
|                          |                                 | `multi_modal_reward.py`        | 多维度奖励函数：正确性 + 推理质量 + 格式规范      |
|                          |                                 | `vlm_grpo_train.py`            | VLM GRPO 训练演示：多模态输入、视觉奖励函数       |
| **Ch12** Agentic RL      | `chapter12_agentic_rl/`         | `multi_turn_rl.py`             | 多轮交互 RL：ORM vs PRM 信用分配策略对比          |
|                          |                                 | `tool_use_agent.py`            | 工具调用智能体：搜索/计算器工具、REINFORCE 训练   |
| **Ch13** 未来趋势        | `chapter13_future_trends/`      | `tree_of_thought.py`           | Tree of Thought 推理：24点游戏、多路径搜索剪枝    |
|                          |                                 | `multi_agent_marl.py`          | 多智能体 RL：GridWorld 协作、独立学习 vs 参数共享 |

### 附录

| 附录                       | 目录                        | 代码文件                     | 说明                                |
| -------------------------- | --------------------------- | ---------------------------- | ----------------------------------- |
| **附录A** 常见坑与解法     | `appendix_common_pitfalls/` | `debug_reward_hacking.py`    | 奖励作弊复现与修复、奖励设计原则    |
|                            |                             | `debug_training_collapse.py` | 训练崩溃诊断：学习率/梯度裁剪/探索  |
| **附录B** 工业级训练与评测 | _(待补充)_                  | _(待补充)_                   | 集群训练与评测                      |
| **附录C** 算法速查         | `appendix_algorithm_guide/` | _(无代码)_                   | 算法选型指南 + 现代 RL 框架三层架构 |

## 统计

- 总计 **37** 个 Python 文件，覆盖 **13** 个章节 + **1** 个附录
- 每个文件包含完整中文注释，可独立运行
- 所有代码已通过 `py_compile` 语法检查
