# 附录 E：数学基础

> **本附录目标**：将全书涉及的核心数学公式按主题组织，每条公式附带一句话中文解释和章节引用，方便随时查阅。

## F.1 概率基础

> 相关章节：[第3章 MDP 形式化](/chapter03_mdp/formalism)

**条件概率**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

在已知事件 B 发生的条件下，事件 A 发生的概率。RL 中状态转移概率 $P(s' \mid s, a)$ 就是一个条件概率。

**贝叶斯定理**

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

通过先验概率 $P(A)$ 和似然 $P(B \mid A)$ 反推后验概率。在贝叶斯强化学习和偏好学习中广泛使用。

**数学期望**

$$\mathbb{E}[X] = \sum_{x} x \cdot P(x)$$

随机变量的加权平均值。RL 中价值函数 $V^\pi(s)$ 本质上就是回报的期望。

**方差**

$$\text{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

衡量随机变量的波动程度。策略梯度中引入 baseline 降低方差，就是为了减少梯度估计的噪声。

## F.2 优化基础

> 相关章节：[第4章 DQN](/chapter04_dqn/from-q-to-dqn)、[第6章 PPO 数学推导](/chapter06_ppo/ppo-math)

**梯度下降**

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$

沿损失函数梯度的反方向更新参数，$\alpha$ 是学习率。RL 中策略参数 $\theta$ 和价值网络参数都通过梯度下降优化。

**随机梯度下降 (SGD)**

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta; x_i)$$

用单个样本（或小批量）估计梯度，而非全量数据。训练效率更高，且噪声有一定正则化效果。

**Adam 更新规则**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta \leftarrow \theta - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam 结合了动量（一阶矩 $m_t$）和自适应学习率（二阶矩 $v_t$），是 RL 训练中最常用的优化器。$\hat{m}_t, \hat{v}_t$ 是偏差校正后的估计值。

## F.3 MDP 核心公式

> 相关章节：[第3章 MDP](/chapter03_mdp/formalism)、[第3章 贝尔曼方程](/chapter03_mdp/bellman-equation)

**状态价值函数**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s\right]$$

从状态 $s$ 出发、遵循策略 $\pi$ 能获得的期望累积折扣回报。$\gamma \in [0,1)$ 是折扣因子。

**动作价值函数 (Q 函数)**

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s, a_0 = a\right]$$

在状态 $s$ 执行动作 $a$ 后，再遵循策略 $\pi$ 的期望回报。DQN 学习的就是 $Q^*$。

**贝尔曼方程**

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[r(s,a,s') + \gamma V^\pi(s')\right]$$

价值函数的递归定义：当前状态的价值 = 即时奖励 + 下一状态价值的折扣期望。这是动态规划和 TD 学习的理论基础。

**贝尔曼最优方程**

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[r(s,a,s') + \gamma V^*(s')\right]$$

最优策略下的价值函数满足：在每个状态都选择让价值最大的动作。Q-Learning 和 DQN 的理论依据。

**TD Error**

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

时序差分误差：实际回报与预估的偏差。TD(0) 用它来更新价值估计，GAE 也基于它构建。

## F.4 策略梯度

> 相关章节：[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)

**策略梯度定理**

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot G_t\right]$$

策略性能对参数的梯度，等于 $\nabla \log \pi$ 与累积回报 $G_t$ 的期望乘积。这是 REINFORCE 和所有策略梯度方法的理论基础。

**对数导数技巧 (Log-derivative Trick)**

$$\nabla_\theta \pi_\theta(a \mid s) = \pi_\theta(a \mid s) \cdot \nabla_\theta \log \pi_\theta(a \mid s)$$

将策略梯度的计算转化为对数概率的梯度，数值上更稳定。这个技巧贯穿整个策略梯度方法。

**带 Baseline 的策略梯度**

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot (G_t - b(s))\right]$$

引入 baseline $b(s)$（通常用 $V(s)$）不改变梯度的期望值，但显著降低方差，加速收敛。

## F.5 PPO 与 GAE

> 相关章节：[第6章 PPO 数学推导](/chapter06_ppo/ppo-math)、[第6章 信任域与裁剪](/chapter06_ppo/trust-region-clipping)

**PPO 裁剪目标**

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ 是新旧策略的概率比。裁剪机制限制策略更新幅度，避免一步走太远。

**GAE（广义优势估计）**

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{k=0}^{T-t-1} (\gamma\lambda)^k \delta_{t+k}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD Error。$\lambda \in [0,1]$ 控制偏差-方差权衡：$\lambda=0$ 退化为 TD(0)（低方差高偏差），$\lambda=1$ 退化为蒙特卡洛（高方差低偏差）。

**TRPO 信任域约束**

$$\max_\theta\; \mathbb{E}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}\left[D_{KL}(\pi_{\theta_{old}} \| \pi_\theta)\right] \le \delta$$

PPO 的前身，通过硬性 KL 约束限制策略更新。PPO 用裁剪替代 KL 约束，实现更简单。

## F.6 DPO 与偏好学习

> 相关章节：[第7章 DPO 数学推导](/chapter07_alignment/dpo-math)

**Bradley-Terry 模型**

$$P(y_w \succ y_l) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

给定提示 $x$，回答 $y_w$ 优于 $y_l$ 的概率由两者的奖励差经 sigmoid 变换得到。这是偏好建模的基础。

**DPO 损失函数**

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)\right]$$

DPO 直接优化策略模型，绕过显式的奖励模型训练。$\beta$ 控制对偏好偏离参考策略的惩罚力度，$\pi_{ref}$ 是参考模型（通常是 SFT 后的模型）。

**DPO 隐式奖励**

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)$$

DPO 定义了一个隐式奖励函数。因为 $Z(x)$ 在 winning 和 losing 的差中被消掉，所以不需要显式计算配分函数。

## F.7 KL 散度与信息论

> 相关章节：[第6章 GAE 与奖励模型](/chapter06_ppo/gae-reward-model)、[第7章 DPO](/chapter07_alignment/dpo-math)

**KL 散度**

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

衡量分布 $P$ 与 $Q$ 的"距离"（不对称）。RLHF 中用来限制训练后的策略不能偏离参考策略太远，防止奖励黑客。

**交叉熵**

$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = H(P) + D_{KL}(P \| Q)$$

分类任务的标准损失函数。在 RL 中也用于策略蒸馏和模仿学习。

**信息熵**

$$H(\pi) = -\sum_{a} \pi(a) \log \pi(a)$$

策略的不确定性度量。熵越高，策略越随机（探索越多）。SAC 算法中通过最大化熵来鼓励探索。

## F.8 GRPO

> 相关章节：[第8章 GRPO 核心机制](/chapter08_grpo_rlvr/grpo-mechanism)

**组相对优势**

$$\hat{A}_i = \frac{r_i - \mu}{\sigma}$$

对于同一个提示生成的 $G$ 个回答，用组内奖励的均值 $\mu$ 和标准差 $\sigma$ 进行归一化。不需要训练价值网络，直接用组内统计量做 baseline。

**GRPO 目标函数**

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^{G} \min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

GRPO 将 PPO 的裁剪目标与组归一化优势结合，同时加入 KL 惩罚。省去了 Critic 网络，是 DeepSeek-R1 的核心训练方法。

## F.9 连续控制公式

> 相关章节：[第9章 DDPG/TD3](/chapter09_continuous_control/continuous-policy-ddpg-td3)、[第9章 SAC](/chapter09_continuous_control/sac-comparison)

**确定性策略梯度**

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^{\mu}(s, a)\big|_{a=\mu_\theta(s)}\right]$$

DDPG 的理论基础。与随机策略梯度不同，确定性策略直接输出动作，梯度通过 Q 网络反传。

**SAC 熵正则化目标**

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}\left[r(s_t, a_t) + \alpha \cdot H(\pi(\cdot \mid s_t))\right]$$

SAC 在标准 RL 目标中加入策略熵 $H(\pi)$ 的奖励，系数 $\alpha$ 控制探索程度。鼓励策略保持随机性，提升鲁棒性。

**Clipped Double Q-Learning (TD3)**

$$y = r + \gamma \min_{i=1,2} Q_{\theta'_i}(s', a')$$

TD3 用两个 Q 网络取最小值计算目标，防止 Q 值过估计。这是 TD3 三大技巧之一。

## F.10 公式速查索引

| 公式          | 符号                                           | 出处 |
| ------------- | ---------------------------------------------- | ---- |
| 状态价值      | $V^\pi(s)$                                     | Ch3  |
| 动作价值      | $Q^\pi(s,a)$                                   | Ch3  |
| 贝尔曼方程    | $V = R + \gamma PV$                            | Ch3  |
| 策略梯度      | $\nabla J = \mathbb{E}[\nabla\log\pi \cdot G]$ | Ch5  |
| PPO 裁剪      | $L = \min(rA, \text{clip}(r)A)$                | Ch6  |
| GAE           | $\hat{A} = \sum(\gamma\lambda)^k \delta_k$     | Ch6  |
| DPO           | $-\log\sigma(\beta\Delta\log\pi)$              | Ch7  |
| Bradley-Terry | $P(y_w \succ y_l) = \sigma(r_w - r_l)$         | Ch7  |
| KL 散度       | $D_{KL} = \sum P\log(P/Q)$                     | Ch7  |
| GRPO          | $\hat{A}_i = (r_i - \mu)/\sigma$               | Ch8  |

::: tip 使用建议
这个附录是查阅工具，不需要从头背到尾。遇到不熟悉的公式时，回来翻对应的章节链接，看完整的推导过程。
:::
