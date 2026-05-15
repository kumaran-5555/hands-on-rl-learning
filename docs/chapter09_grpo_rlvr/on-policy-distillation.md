# 9.6 在线策略蒸馏（OPD）——把 Teacher 变成密集奖励

上一节讨论了 RLVR 如何用规则验证器替代 RM，在数学、代码等有客观答案的领域给出精确的奖励信号。本节看另一条路线：**不让小模型从零探索，而是让强模型在它自己生成的轨迹上逐 token 给指导**。这条路线叫在线策略蒸馏（On-Policy Distillation, OPD）。

核心流程只有三步：student 先自己对 prompt 生成回答（即"策略"）；teacher 不重写答案，而是看 student 写出的每一步，判断每个 token 合不合理；这个判断作为密集的 token 级信号回传给 student，指导它调整。这里的关键术语对应：student 模型就是 RL 中的**策略（policy）**，每个被选出的 token 是**动作（action）**，teacher 对该 token 的 log-prob 就是**反馈信号**。

OPD 和前面方法的核心区别在于两个维度。一是**谁生成训练轨迹**：SFT / SeqKD 用 teacher 写的答案，GRPO 用 student 自己的探索，OPD 也用 student 自己的轨迹。二是**反馈有多密**：GRPO / RLVR 通常是结果级奖励——答案对了给 1 分、错了给 0 分，一个 2000 token 的解题过程只有最后那个 0 或 1；OPD 则几乎每个 token 都有信号，因为 teacher 对每个 token 都能给出 log-prob。

| 方法        | 谁生成训练轨迹 | 反馈来自哪里         | 反馈粒度            | 主要问题                  |
| ----------- | -------------- | -------------------- | ------------------- | ------------------------- |
| SFT / SeqKD | 人类或 teacher | 标准答案 token       | token-level         | student 不练自己的错误    |
| PPO / GRPO  | student        | RM 或规则验证器      | 多为 sequence-level | 奖励稀疏，采样昂贵        |
| DPO         | 离线偏好数据   | chosen / rejected 对 | sequence-pair-level | 不能在线探索              |
| **OPD**     | **student**    | **teacher log-prob** | **token-level**     | 需要好 teacher 和好初始化 |

一句话概括：**OPD 同时拿到 RL 的 on-policy 分布和蒸馏的密集监督。**

## OPD 的核心思想

### Teacher 在 OPD 中的角色

所有蒸馏方法里都有 teacher，区别在于 teacher 在训练中扮演什么角色。

|                     | teacher 的角色                                     | 训练轨迹来自谁 | student 学到什么         | 局限                                                    |
| ------------------- | -------------------------------------------------- | -------------- | ------------------------ | ------------------------------------------------------- |
| **SFT / SeqKD**     | 答案作者：生成完整答案                             | teacher        | 老师怎么写               | 只在 teacher 上下文里学，不练自己的错误                 |
| **Off-policy 蒸馏** | 专家示范者：展示完整解题轨迹                       | teacher        | 老师怎么想               | 轨迹仍是 teacher 走到的，student 自己卡住的地方没有信号 |
| **OPD**             | 在线评价器：看 student 前缀，逐 token 判断是否合理 | student        | 在自己走错的路上怎么纠偏 | 需要好的 teacher 和好的初始化                           |

从上到下，teacher 从**数据生产者**变成**奖励提供者**，student 从模仿 teacher 轨迹变成在自己的轨迹上被纠偏。OPD 的动机正是消除分布偏移：student 推理时看到的是自己生成的上下文，训练时直接在这些区域给信号，而不是只在 teacher 走过的路上铺数据。

### OPD 到底在优化什么

OPD 的训练循环很短：

1. 给定 prompt $x$，student $\pi_\theta$ 自己采样回答 $y \sim \pi_\theta(\cdot \mid x)$。
2. 对 student 生成的每个前缀 $c_t=(x,y_{<t})$，teacher $q$ 计算当前 token $y_t$ 的概率。
3. 用 teacher 对 student 自己 token 的评价更新 student。

Google DeepMind 的 GKD 是这条线的代表性起点：它不再只依赖固定 teacher 输出，而是让 student 在自生成序列上接受 teacher 反馈，并允许 forward KL、reverse KL、JSD 等不同散度。[^gkd]

如果采用最常见的 reverse KL 形式，目标可以写成：

$$
\mathcal{L}_{\text{OPD}}(\theta)
= \mathbb{E}_{y \sim \pi_\theta}
\left[
\sum_t
\log \frac{\pi_\theta(y_t \mid c_t)}{q(y_t \mid c_t)}
\right]
$$

这就是在 student 自己访问到的状态上最小化：

$$D_{\text{KL}}(\pi_\theta(\cdot \mid c_t) \| q(\cdot \mid c_t))$$

展开后，每个 token 都有一个很自然的训练信号：

$$
r_t
= \log q(y_t \mid c_t) - \log \pi_\theta(y_t \mid c_t)
$$

teacher 比 student 更认可这个 token，$r_t$ 就高；teacher 觉得这个 token 不像好策略，$r_t$ 就低。Thinking Machines Lab 的实现里，几乎就是把 RL 训练脚本里的 KL regularizer model 换成 teacher：采 student rollout，算 student log-prob，再让 teacher 对同一条轨迹算 log-prob，最后把负 reverse KL 当作 per-token advantage。[^tml_opd]

所以 OPD 和 RL 的对应关系非常直接：

| RL 概念    | OPD 中的对应物                                           |
| ---------- | -------------------------------------------------------- |
| 状态 $s_t$ | prompt + 已生成前缀 $c_t$                                |
| 动作 $a_t$ | 下一个 token $y_t$                                       |
| 策略       | student $\pi_\theta$                                     |
| 奖励       | teacher 给 student token 的相对认可度                    |
| 采样分布   | student 自己的分布                                       |
| 优势估计   | 常用 $\log q(y_t \mid c_t)-\log \pi_\theta(y_t\mid c_t)$ |

这里要注意一个细节：teacher 不是环境真理，它只是一个强策略。OPD 不是让模型"发现超过 teacher 的新策略"，而是把 teacher 在 student 自己状态上的行为压进 student。它更像有过程监督的模仿学习，而不是纯探索型 RL。

## 术语辨析：distillation, off-policy, on-policy

先把 **distillation**、**off-policy / offline** 和 **on-policy** 三个概念分清。它们经常被混在一起，但指的是不同层面的事情。

**蒸馏（distillation）** 描述 teacher-student 关系：一个强模型 $q$ 把能力迁移给一个较小或较便宜的模型 $\pi_\theta$。最朴素的蒸馏是让 teacher 生成答案，student 对这些答案做交叉熵训练：

$$
\mathcal{L}_{\text{hard KD}}
= -\sum_t \log \pi_\theta(y_t^T \mid x, y_{<t}^T)
$$

这里 student 看到的是 teacher 选出来的 token $y_t^T$。如果你能拿到 teacher 的完整概率分布，还可以做"软蒸馏"：不只告诉 student 正确 token 是什么，还告诉它其他 token 有多合理。

$$
\mathcal{L}_{\text{soft KD}}
= \sum_t D_{\text{KL}}\left(q(\cdot \mid c_t) \| \pi_\theta(\cdot \mid c_t)\right)
$$

软蒸馏的信息量更大。比如 teacher 可能认为 "therefore" 很好，"so" 也可以，"banana" 完全不行。硬标签只告诉你 teacher 最后选了哪个词，软标签告诉你 teacher 对整个动作空间的判断。

**policy（策略）** 在 LLM 里就是"给定上下文，输出下一个 token 分布"的模型：

$$
\pi_\theta(a_t \mid s_t)
\quad \Longleftrightarrow \quad
\pi_\theta(y_t \mid x, y_{<t})
$$

这里的状态 $s_t$ 是 prompt 加已生成前缀，动作 $a_t$ 是下一个 token。谁生成轨迹，谁就是行为策略（behavior policy）。这件事决定了训练数据的分布。

**Off-policy** 指的是：训练数据不是当前 student 自己生成的，而是来自另一个策略 $\mu$。这个 $\mu$ 可以是 teacher、旧 checkpoint、用户日志、历史模型，也可以是一批固定数据集。普通 teacher 蒸馏就是典型 off-policy：

$$
y \sim q(\cdot \mid x),
\quad
\text{update } \pi_\theta
$$

数据来自 teacher $q$，更新的是 student $\pi_\theta$。好处是便宜、可复用、稳定；坏处是 student 没有在自己的错误上下文里训练。

**Offline** 比 off-policy 更严格：不只是数据来自别的策略，而且训练时不再新采样，只使用一份固定数据集。SFT、DPO、离线偏好训练通常都是 offline。可以这样记：

| 概念       | 数据从哪里来            | 训练时还采新数据吗 | 例子                        |
| ---------- | ----------------------- | ------------------ | --------------------------- |
| offline    | 固定历史数据集          | 不采               | SFT、DPO、离线 SeqKD        |
| off-policy | 不是当前 student 的策略 | 可以采，也可以不采 | teacher 轨迹、旧模型 replay |
| on-policy  | 当前 student 自己生成   | 要采               | PPO、GRPO、OPD rollout      |

所以，**offline 在 LLM 后训练里通常也是 off-policy，但 off-policy 不一定 offline**。如果你每一轮都让 teacher 重新生成数据，再训练 student，它不是 offline，但仍然是 off-policy，因为行为策略还是 teacher，不是 student。

**On-policy** 指的是：用当前 student 自己生成的数据来更新当前 student。形式上是：

$$
y \sim \pi_\theta(\cdot \mid x),
\quad
\text{update } \pi_\theta
$$

它的优势是训练分布和推理分布一致。student 推理时会走到什么上下文，训练时就真的让它走过去，然后在那里给反馈。代价也很明显：每轮更新前都要重新 rollout，旧数据很快过期，样本效率低。

把这几个概念合起来看，OPD 的位置就很清楚了：

- 它是 **distillation**，因为反馈来自 teacher。
- 它是 **on-policy**，因为轨迹来自 student 自己。
- 它通常不是纯 **offline**，因为训练中要不断重新采 student rollout。
- 它和普通 off-policy 蒸馏的区别，不在于有没有 teacher，而在于 teacher 是在谁的轨迹上给信号。

## 为什么普通蒸馏不够

知识蒸馏（Knowledge Distillation, KD）的老问题是：大模型好用但贵，小模型便宜但弱。做法也很直接：让 teacher 生成数据，student 在这些数据上做监督学习。LLM 时代的 KD 综述通常把它分成几类：白盒蒸馏看 teacher logits，黑盒蒸馏只看 teacher 输出；也可以按能力分成推理、对齐、领域知识、工具使用等技能蒸馏。[^kd_survey_xu][^kd_survey_yang]

这条路非常有用。DeepSeek-R1 的蒸馏模型就是典型例子：先让强推理模型生成高质量轨迹，再把这些轨迹 SFT 到小模型上。对小模型来说，这往往比直接做 RL 更稳。

但它有一个根本缺口：**训练时 student 看到的是 teacher 的状态分布，推理时 student 走的是自己的状态分布。**

设 prompt 是 $x$，teacher 轨迹是：

$$y^{T} = (y_1^T, y_2^T, \dots, y_T^T) \sim q(\cdot \mid x)$$

普通蒸馏训练的是：

$$\mathcal{L}_{\text{off-policy}}(\theta) = -\sum_t \log \pi_\theta(y_t^T \mid x, y_{<t}^T)$$

这里的上下文 $x, y_{<t}^T$ 来自 teacher。可一旦 student 在第 3 步写错了，后面的上下文就变成了 $x, y_{<3}^{S}$。这个状态 teacher 可能从来不会走到，SFT 数据里也没有"从这里怎么救回来"的示范。错误会沿着自回归生成不断放大，这就是 exposure bias，也可以理解成模仿学习里的分布偏移。DAgger 早就指出过：要缓解这种问题，必须把 learner 自己访问到的状态也纳入训练。[^dagger]

OPD 就是把这个思想搬到 LLM 蒸馏里。

## Online OPD vs Offline OPD

既然 OPD 是 on-policy，[Lightning OPD](https://arxiv.org/html/2604.13010v1) 为什么又叫 offline on-policy distillation？两者并不矛盾。关键在于区分"轨迹是谁生成的"和"teacher 什么时候打分"。

**标准 OPD 是 online 的。** 每一轮训练都用当前 student 生成新回答，teacher 现场计算每个 token 的 log-prob，然后更新 student。下一轮 student 参数变了，重新生成、打分、更新。

```text
当前 student 生成回答
→ teacher 现场打分
→ 更新 student
→ 新 student 再生成回答
→ teacher 再现场打分
→ ...
```

这条路线最忠实于 on-policy 定义，但成本高：teacher 往往比 student 大很多，标准 OPD 需要一个 live teacher server 在训练期间持续运行。Lightning OPD 论文把这个称为标准 OPD 的基础设施瓶颈：teacher 要在每个梯度步上给新 rollout 计算 log-prob。[^lightning_opd]

**Lightning OPD 是 offline 近似。** 它先做一个 SFT student，然后用这个 SFT student 生成一批固定回答；teacher 只对这批回答打分一次，把每个 token 的 log-prob 存下来。真正训练 OPD 时，不再启动 teacher server，而是直接从缓存里读取 teacher 分数，同时在线计算当前 student 自己的 log-prob。

```text
预处理阶段：
SFT student 生成回答
→ teacher 打分一次
→ 存下 token 和 teacher log-prob

训练阶段：
读取固定回答和 teacher log-prob
→ 计算当前 student log-prob
→ 更新 student
```

Lightning OPD 仍然属于 on-policy，因为这些固定回答不是 teacher 写的，而是 student 系列模型生成的。teacher 仍然是在 student 轨迹上给反馈，保留了 OPD 最关键的东西：**teacher 批改的是 student 会写出来的内容**。

但 Lightning OPD 不再是严格的 online OPD，因为训练过程中 student 参数变了，rollout 却不刷新。它成立依赖一个经验观察：OPD / RL 后训练里的 student 通常不会离 SFT 初始化太远，所以用 SFT student 的 rollout 可以近似后续 student 的 rollout。论文还强调一个条件：**teacher consistency**——生成 SFT 数据的 teacher 和 OPD 打分的 teacher 最好是同一个，否则缓存轨迹和后续打分会产生系统性偏差。[^lightning_opd]

可以这样选：

| 方案                    | 优点                                  | 代价                                   | 适合什么时候                         |
| ----------------------- | ------------------------------------- | -------------------------------------- | ------------------------------------ |
| 标准 online OPD         | 最贴近定义，训练分布最新              | 要 live teacher，成本高                | 研究机制、追求最稳、资源充足         |
| Lightning / offline OPD | 便宜，容易复现，不需要 teacher server | rollout 固定，依赖 teacher consistency | 快速验证、资源有限、student 漂移较小 |
| 普通 offline KD         | 最简单，直接 SFT                      | 只学 teacher 轨迹                      | 冷启动、先把 student 拉到合理区域    |

所以更准确的说法是：**标准 OPD 是 online on-policy distillation；Lightning OPD 是把标准 OPD 离线化的一种工程近似；普通 teacher SFT 是 off-policy / offline distillation。**

## 为什么是 reverse KL

蒸馏里最容易混淆的是 KL 方向。

经典 KD 常用 forward KL：

$$D_{\text{KL}}(q \| \pi_\theta)$$

它要求 student 覆盖 teacher 的概率质量。对分类任务这很自然：teacher 说猫 0.7、狗 0.2、狐狸 0.1，student 也应该学这个软标签。但对长文本生成来说，覆盖所有 teacher 可能说的话会让 student 概率分布变平滑：许多低概率、边缘的 token 也被抬高，生成会发散。

MiniLLM 的核心判断是：生成式 LLM 蒸馏更适合 reverse KL：

$$D_{\text{KL}}(\pi_\theta \| q)$$

reverse KL 是 mode-seeking：它鼓励 student 集中到 teacher 高概率的少数模式上，而不是覆盖 teacher 的所有可能模式。MiniLLM 还把这个目标落成 on-policy 优化，从而减少长文本生成里的 exposure bias。[^minillm]

但 reverse KL 有一个现实限制：**它只能在 student 当前会采到的区域里修正策略。** 如果某个关键 token 在 student 初始化里概率几乎为 0，student 根本采不到它，teacher 再懂也没机会给它高分。这就是为什么很多实用 recipe 都不是"直接 OPD"，而是：

1. 先用 off-policy SFT / SeqKD 做冷启动，把 student 拉到 teacher 支持集附近。
2. 再用 OPD 在 student 自己轨迹上做精修。

Thinking Machines Lab 的复现实验也是这样：先做 off-policy reasoning distillation，再用 OPD 做后训练提升。2026 年的 OPD 机制分析把这件事讲得更彻底：OPD 成功不只取决于 teacher 分数，还取决于 teacher 和 student 在 student 当前状态附近能不能形成可优化的局部重叠。[^rethinking_opd]

还有一条工程化路线是调散度本身。DistiLLM 使用 skew KL 和 adaptive off-policy 策略，目标是在 teacher 信号和 student 可学习性之间找更平滑的折中。[^distillm]

## 为什么强 teacher 也会翻车

直觉上，做 OPD 就是"找一个更强的 teacher，把它的能力压进 student"。但 OPD 真正迁移的不是排行榜分数，而是 **teacher 在 student 自己前缀上的局部偏好场**。

具体来说，teacher 只有在 student 真实走到的状态 $c_t=(x,y_{<t})$ 上，能把 student 已经认真考虑的候选 token 重新排序，这个信号才会变成有用梯度。如果 teacher 很强，但它的高概率 token 和 student 的高概率 token 基本不重叠，reverse KL 看到的是两个互相错开的支持集：student 采不到 teacher 想要的关键 token，teacher 也只能对 student 已经写出的奇怪 token 给低分。训练变成每一步都被批评，但没有可吸收的方向。

这就是 "thinking-pattern consistency" 的具体含义：teacher 和 student 是否在用相近的中间语言思考同一道题。一个数学 teacher 可能习惯先写完整推导，一个 student 可能习惯先猜公式；一个 reasoning teacher 可能会显式拆分子问题，而一个 base student 只会接着 prompt 补一句短答案。它们最终都可能给出正确答案，但中间路径不一样，逐 token 蒸馏时就会互相冲突。

2026 年的机制分析论文把 OPD 从"能力迁移"重新定义为"局部可教性"问题。[^rethinking_opd] teacher 分数高，只说明它自己会做题；teacher 可教，才说明它能在 student 当前的位置给出能被吸收的方向。在 SFT 里这两件事经常被混在一起，因为 SFT 直接把 student 拉到 teacher 轨迹上；但在 OPD 里，student 先走自己的路，teacher 只能在 student 走到的地方说话。于是核心问题从"teacher 有多强"变成了"teacher 的偏好能不能在 student 当前分布上形成梯度"。

**高分不等于新知识。** 如果 teacher 只是同一条训练 pipeline 里更大的 sibling，它可能只是把同一种数据和 recipe 吃得更充分，对 student 来说局部偏好并不提供新的方向，只是在已有分布上更自信。经过额外 RL post-training、数据扩展或任务专门训练的 teacher，哪怕参数不夸张，也可能带来 student 训练中没见过的决策边界。

**OPD 不是把 teacher 的答案压缩进 student，而是把 teacher 的"局部取舍方式"压缩进 student。** 一个只是"更大但同质"的 teacher，给 student 的可能是更强的确认偏差；经历过新的 RL、看过新的失败样本、形成了新的推理习惯的 teacher，才更可能提供新的可迁移结构。

这也解释了为什么 OPD 比普通蒸馏更"挑老师"。off-policy SFT 可以强行把 student 拉去看 teacher 轨迹；OPD 则是在 student 自己的地形上做局部导航——如果当前位置附近没有通向 teacher 模式的梯度，再大的 teacher 也只是站在远处报答案。

### Overlap token 才是主战场

论文里最有启发的实验是把 token 集合拆开：只在 student 和 teacher 都认为高概率的 overlap tokens 上优化，效果几乎不掉；只看 non-overlap tokens，几乎帮不上忙。这解释了为什么 top-$k$ OPD 往往已经够用，也解释了为什么失败 run 的 loss 看起来还在动，能力却没有涨。

语言模型的下一个 token 分布非常尖——大多数概率质量集中在少数候选上。如果这些候选 token 师生共享，teacher 的 log-prob 就像一个细粒度排序器，告诉 student "你已经在正确候选集里了，把权重挪到更好的选项上"。如果候选集不共享，teacher 的反馈主要是在否定 student 采到的 token，却没有把 student 带进新的高概率区域。

**密集监督不等于每个 token 都同样有用。** 真正有用的是那些 student 已经能想到、teacher 又能进一步排序的候选。overlap token 像一座桥：一头接 student 当前能力，另一头接 teacher 的更好偏好。non-overlap token 更像远处的灯塔——能说明方向错了，但不告诉 student 下一步怎么走。

论文还观察到一个反直觉现象：失败 teacher 的全局 reward 也能区分正确/错误 rollout，但这个信息没有形成局部可利用的优化几何。**全局会打分，不等于局部能教学。** 这解释了为什么有些 OPD run 看起来 reward 没坏、teacher 也不弱，最后却学不动。

### 长链条里的免费午餐会变质

OPD 的诱人之处是每个 token 都有 reward，但长思维链会暴露一个弱点：越往后，student 前缀越可能偏离 teacher 熟悉的分布，teacher 在这些陌生前缀上给出的 log-prob 越像噪声。论文在长响应里观察到一种从后向前扩散的熵坍塌：suffix 先变得高熵和不稳定，然后这种不稳定逐步传回更早位置。

这说明 OPD 的 scaling bottleneck 不只是 teacher forward 太贵，而是 **密集监督的可靠性随轨迹深度下降**。在短数学题或格式化回答里，teacher 批改 student 前缀通常可靠；在 15K token 的长推理、工具调用、多轮 agent 轨迹里，teacher 很可能已经不在自己训练时熟悉的状态分布上。更稳的做法是分段蒸馏、混入序列级验证器、限制每段 horizon，或只在高置信 overlap 区域施加 token-level loss。

## 动手：OPD 打分最小实现

前面把 OPD 的机制和坑理清了。现在用代码实现 OPD 最核心的一步：student 生成回答，teacher 在 student 的轨迹上逐 token 打分。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

student_name = "Qwen/Qwen2.5-0.5B-Instruct"
teacher_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(student_name)
student = AutoModelForCausalLM.from_pretrained(
    student_name, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
)
student.eval()
teacher.eval()

prompt = "Solve: if x + 3 = 7, what is x? Show your work."
inputs = tokenizer(prompt, return_tensors="pt").to(student.device)
prompt_len = inputs["input_ids"].shape[1]

# Step 1: student 生成回答
with torch.no_grad():
    output_ids = student.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
    )

full_ids = output_ids

# Step 2: 计算 student 和 teacher 的逐 token log-prob
def next_token_logps(model, input_ids):
    """计算模型对给定序列中每个 token 的 log probability。

    logps[:, i] 是位置 i 的 logits 预测的 token i+1 的 log-prob。
    """
    logits = model(input_ids).logits
    logps = torch.log_softmax(logits[:, :-1], dim=-1)
    next_ids = input_ids[:, 1:]
    return logps.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)

with torch.no_grad():
    student_logps = next_token_logps(student, full_ids)
    teacher_logps = next_token_logps(teacher, full_ids.to(teacher.device)).to(student_logps.device)

# Step 3: 计算逐 token reward（teacher 认可度 - student 自信度）
gen_mask = torch.zeros_like(student_logps, dtype=torch.bool)
gen_mask[:, prompt_len - 1 :] = True  # 只看生成部分

token_rewards = teacher_logps - student_logps
generated_ids = full_ids[:, 1:][gen_mask]
generated_rewards = token_rewards[gen_mask]

for tok_id, reward in zip(generated_ids[:32], generated_rewards[:32]):
    token = tokenizer.decode([tok_id.item()])
    print(f"{token!r:12s} reward={reward.item():+.3f}")
```

设计要点：

- `next_token_logps()` 同时用于 student 和 teacher，计算每个位置的 log probability。这个函数和 GRPO 里计算 log prob 的逻辑一样——OPD 的工程基础设施和 RL 高度复用。
- `token_rewards = teacher_logps - student_logps`：这就是 OPD 的 per-token reward。正值表示 teacher 比 student 更认可这个 token，负值表示 teacher 不认可。
- 这段代码只做了"打分"部分。如果要变成训练循环，核心只差三件事：batch rollout、reward 归一化/裁剪、policy gradient 更新。

训练循环的伪代码：

```python
for prompts in dataloader:
    # Step 1: student rollout
    trajectories = student.rollout(prompts)
    student_logps = student.logprobs(trajectories)
    teacher_logps = teacher.logprobs(trajectories)

    # Step 2: 计算 per-token advantage
    advantages = teacher_logps - student_logps
    advantages = normalize_and_mask(advantages, trajectories.response_mask)

    # Step 3: 策略梯度更新
    loss = policy_gradient_loss(
        new_logps=student.logprobs(trajectories),
        old_logps=student_logps.detach(),
        advantages=advantages.detach(),
    )
    loss.backward()
    optimizer.step()
```

真实系统还会加 KL 到 reference、长度控制、重复惩罚、prompt 难度采样和 eval gating。OPD 不是"一个公式就完事"，它是把 teacher log-prob 接到现有 RL 训练基础设施里。

## OPD 和相邻路线的关系

### 和 SFT / SeqKD 的关系

SFT 是必要的基础。它便宜、稳定、能快速把 student 拉到合理输出区域。OPD 不是替代 SFT，而是在 SFT 之后解决 student 自己轨迹上的错误修正。

可以简单类比：

- SFT / SeqKD：把路铺到 teacher 经常走的区域
- OPD：student 自己开车，teacher 坐副驾逐步纠偏

### 和 GRPO / RLVR 的关系

GRPO / RLVR 的奖励通常来自外部验证器：答案对不对、代码能不能跑、格式是否合格。这类奖励客观，但常常稀疏。一个 2000-token 的解题过程，最后只得到 0 或 1。

OPD 的奖励来自 teacher，每个 token 都能给信号。它不需要标准答案，也不需要 RM，但上限受 teacher 限制。

所以两者适合组合：数学题可以用规则奖励告诉模型最终答案是否正确，再用 teacher log-prob 给中间过程更密的 shaping 信号。Thinking Machines Lab 也把"蒸馏式 per-token reward + sequence-level environment reward"列为值得继续探索的方向。[^tml_opd]

### 和 DPO 的关系

DPO 的漂亮之处是：语言模型自己可以被看成隐式奖励模型。OPD 更进一步：直接把一个强 teacher 当作逐 token reward model。

但 DPO 是离线偏好优化，适合已有 chosen / rejected 数据的场景；OPD 是在线采样，适合你有 teacher、但缺少偏好数据或验证器的场景。

### 和黑盒蒸馏的关系

上面的 OPD 默认你能拿到 teacher log-prob，这是白盒或至少 logprob-access teacher。现实里很多 teacher 只有 API 文本输出，没有 logits。黑盒 OPD 就要换信号：例如 2025 年的 GAD 把 student 当 generator，训练 discriminator 区分 teacher 和 student 的回答，再用 discriminator 作为随 student 演化的 on-policy reward model。[^blackbox_opd]

这条线很有实践价值，但工程复杂度也高：你又引入了一个会被 hack、会漂移、需要稳定训练的判别器。

还有 teacher-free 的变体：Self-Distilled Reasoner 让同一个模型在不同上下文下同时扮演 teacher 和 student，用自蒸馏减少对外部强 teacher 的依赖。[^self_distilled_reasoner]

## 什么时候该用 OPD

OPD 最适合这些场景：

| 场景                     | 为什么 OPD 合适                                    |
| ------------------------ | -------------------------------------------------- |
| 小模型继承大模型推理能力 | 小模型探索弱，teacher 能给密集过程信号             |
| 没有规则验证器           | 不需要标准答案，只需要 teacher 能评估 token        |
| 领域模型后训练           | 可以用强 teacher 恢复 instruction following 或格式 |
| 已经有不错的 SFT 初始化  | student 能采到 teacher 支持集附近的 token          |
| 想降低 RL 成本           | teacher forward 比完整 RL 探索和稀疏奖励更高效     |

不适合这些场景：

| 场景                            | 风险                                          |
| ------------------------------- | --------------------------------------------- |
| 想明显超过 teacher              | OPD 本质是压缩 teacher 行为，不负责发现新策略 |
| student 太弱，采不到关键 token  | reverse KL 没法强化零概率附近的行为           |
| teacher 和 student 推理风格冲突 | 逐 token 信号可能互相拉扯，训练不稳定         |
| 长程任务只靠局部 token reward   | 局部对齐不一定等于全局任务成功                |
| 只有黑盒 teacher API            | 需要额外 reward/discriminator 设计            |
| teacher 只是同 pipeline 放大版  | 可能分数更高，但不给 student 新决策边界       |

2026 年的 OPD survey 把这个领域整理成三个维度：反馈信号（logit-based、outcome-based、self-play）、teacher access（white-box、black-box、teacher-free）和 loss granularity（token-level、sequence-level、hybrid）。[^opd_survey] 这个分类很好用：一看到一个"OPD 新方法"，先问这三个问题，就不会被名字绕晕。

## 实战指南

### 一个实用 recipe

实际跑 OPD 时，可以按以下步骤操作：

**第一步，选 teacher。** teacher 不一定越大越好，关键是它在目标任务上比 student 强、能提供 student 还没学会的能力，并且和 student 的 tokenizer、输出风格、推理格式尽量兼容。同族模型通常更省心，但"同族更大"不是充分条件。最好做一个小对照：同 pipeline 放大版 teacher vs 额外 RL / 数据增强后的 teacher。如果后者明显更好，说明 OPD 需要的是新决策边界，不只是参数规模。

**第二步，做 off-policy 冷启动。** 用 teacher 生成一批高质量答案，先 SFT student。目标不是一步到位，而是让 student 进入 teacher 的支持集附近。如果初始 overlap ratio 很低，直接 OPD 往往启动不了；先在 teacher rollout 上做轻量 SFT，等 student 能采到相似思路，再切到 on-policy。

**第三步，选 prompt 分布。** prompt 不只是任务数据，也是在选择 student 会进入哪些状态。尽量使用 teacher 训练时熟悉的模板、系统提示和题型，让早期 OPD 有足够 overlap；同时混入一部分 OOD prompt，避免 student 只学会 teacher 的固定口癖和低熵模板。

**第四步，采 student rollout。** 每个 prompt 采 2-8 条回答，保留 token、log-prob、mask、长度、停止原因。这里和 PPO / GRPO 的 rollout 基础设施基本相同。

**第五步，teacher 打分。** 对 student 生成的完整上下文做 teacher forward，拿生成 token 的 log-prob。白盒 teacher 可以直接算 logits；黑盒 teacher 需要另设 reward 近似。这里不要把 teacher 当成绝对裁判，而要把它当成一种局部偏好：它正在告诉 student，在 student 自己写出的前缀里，哪些选择更像 teacher 会保留的路径。

**第六步，更新 student。** 用 teacher_logp - student_logp 做 per-token advantage，再接 PPO-style loss 或 importance-sampling loss。实践中要监控 entropy、KL、response length 和重复率，避免 student 过早塌缩。长回答任务可以只对前若干 token、分段窗口或高置信 overlap token 上 loss，别默认整条 10K token 轨迹都同样可靠。

**第七步，和任务奖励混合。** 如果任务有验证器，不要浪费它。可以把 final reward 用于序列级方向，把 OPD reward 用于 token-level shaping。这样能弥补 OPD 的一个盲点：teacher 局部认可的 token，不一定保证全局答案正确。

**第八步，做 eval gating。** OPD 很容易把 teacher 的风格也压进 student。除了目标 benchmark，还要测通用能力、格式、拒答、安全和长度分布。

### 快速测试方案

第一次验证 OPD，不建议直接上大规模训练。先用小实验回答三个问题：

1. teacher 给的 token-level 信号是不是有意义？
2. offline 近似能不能先跑出正向变化？
3. online rollout 是否比 offline 缓存明显更好？

#### 0. 先问 teacher 是否真的"可学"

这是最快的 sanity check，几十分钟就能做。重点是形成判断：这个 teacher 是否在 student 当前能力附近说话。

选 50-100 个 prompt，让 student 每题生成 2-4 个回答。teacher 不生成答案，只对 student 的回答逐 token 算 log-prob。然后人工检查：teacher 给高分的 token，是否是 student 本来就有可能写出的合理延续？teacher 给低分的地方，是否确实对应推理分叉、格式偏移或过早下结论？如果高低分主要在惩罚长度、惩罚某种模板，或者把所有 student token 都压低，说明 teacher 和 student 不在同一个思维空间里。

这一步能直接验证论文的核心 insight：OPD 的失败常常不是 teacher"不会评分"，而是 teacher 的评分没有落在 student 能调整的位置上。全局上有信息的 reward，如果在每个局部 token 选择处都给不出可吸收的方向，训练仍然会原地打滑。

#### 1. Lightning OPD smoke test

第二步先跑 offline 版本，因为它便宜、稳定、容易复现。

数据规模可以很小：

- 训练 prompt：200-1000 条
- 验证 prompt：50-200 条
- student：0.5B-1.5B 小模型
- teacher：同族更大模型，或你要蒸馏的强模型
- 训练方式：LoRA 即可，100-500 steps 先看趋势

流程如下：

```text
1. 用 SFT student 生成固定回答
2. 用 teacher 预计算每个生成 token 的 log-prob
3. 训练 student，让它提高 teacher 认可的 token 概率
4. 在 held-out prompt 上重新生成答案
5. 比较训练前后：任务分数、长度、重复率、teacher score
```

最小可接受结果不是"teacher score 上升"，而是：

| 观察项            | 期望                                     |
| ----------------- | ---------------------------------------- |
| held-out 任务分数 | 小幅上升，或至少不下降                   |
| 平均回答长度      | 不明显爆炸，也不极端变短                 |
| 重复率            | 不上升                                   |
| teacher score     | 上升，但不能靠变短、套模板或重复来刷分   |
| 人工样例          | 至少 20 个样例里，多数看起来更像 teacher |

如果 teacher score 上升但任务分数下降，说明 student 可能在学 teacher 的局部口癖，而不是学能力。此时优先检查 mask、长度归一化、prompt 分布和 teacher consistency。

#### 2. Online OPD 小规模对照

如果 offline smoke test 有正向信号，再做一个小规模 online 对照。只跑 2-3 轮：

```text
Round 1: 当前 student 生成 rollout → teacher 打分 → 训练 50-100 steps
Round 2: 更新后的 student 重新生成 rollout → teacher 重新打分 → 再训练
Round 3: 可选，观察是否继续提升
```

对照组是 Lightning OPD：同样 prompt、同样 teacher、同样训练步数，但 rollout 固定不刷新。看 online 是否明显优于 offline。

| 如果结果是...               | 结论                                           |
| --------------------------- | ---------------------------------------------- |
| offline 已经接近 online     | 用 Lightning OPD 更划算，没必要上 live teacher |
| online 明显更好             | student 分布漂移较大，刷新 rollout 有价值      |
| online 不稳定，offline 更稳 | teacher 信号可能噪声大，或 online 采样质量太差 |
| 两者都没提升                | 优先回到 SFT 数据、teacher 选择和任务评测      |

这个测试能帮你决定工程路线：如果 offline 就够好，先走 Lightning；如果 online 明显赢，再考虑搭 teacher server。

#### 3. 最小实验报告模板

每次跑 OPD，至少记录这张表：

| 项目              | 内容                                             |
| ----------------- | ------------------------------------------------ |
| student / teacher | 模型名、参数量、是否同族                         |
| 数据              | prompt 来源、数量、是否和评测去重                |
| rollout           | 每题采样数、temperature、max tokens              |
| reward            | teacher log-prob 是否长度归一化、是否裁剪        |
| 训练              | online 还是 offline、steps、LoRA rank            |
| 评测              | 任务分数、长度、重复率、人工样例                 |
| insight 记录      | teacher 学到了什么新东西，student 是否真的吸收了 |
| 结论              | 继续 online、用 Lightning、还是回去做 SFT        |

这张表比单看 loss 重要得多。OPD 的 loss 下降只能说明 student 更像 teacher，不自动说明模型真的更会做题。

## 开源框架支持

本节列出主流训练框架对 OPD 和 OPSD 的支持情况与使用入口。所有信息均基于各框架仓库当前代码与官方文档的直接查证。

### 术语速查

| 缩写 | 含义                                                                                                                                                                  |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OPD  | On-Policy Distillation，学生自己生成 rollout，由**独立的外部教师模型**逐 token 打分。                                                                                 |
| OPSD | On-Policy Self-Distillation，**同一个模型**同时扮演 teacher 和 student，teacher 通过 privileged context（如 ground-truth 答案、"简洁"前缀等）获得额外信息来指导学生。 |

### OPD 框架详细说明

#### 1. slime（THUDM）

slime 是 GLM-4.5 / 4.6 / 4.7 的后训练框架，OPD 作为第一等特性内建，设计为**可加在任意 RL 算法上的 KL 惩罚项**。

**支持模式：**

- **SGLang 模式**：teacher 跑在独立 SGLang server 上，通过 `--rm_url` 在 rollout 阶段拉取 token-level log-probs。适合 teacher 架构不同或显存放不下的场景。
- **Megatron 模式**：teacher 直接加载进 Megatron 训练进程，通过 `--opd-teacher-load` 指定 checkpoint。要求 teacher 与 policy/ref 同架构。

**最小启动命令（Megatron 模式）：**

```bash
python train.py \
  --use-opd \
  --opd-type megatron \
  --opd-kl-coef 1.0 \
  --opd-teacher-load /path/to/teacher_ckpt \
  --adv_estimator grpo   # 也可换 ppo / reinforce_plus_plus
```

**关键设计：**

- OPD 与 advantage estimator 正交，只是给 advantage 加一个 reverse KL 惩罚。
- `slime/rollout/on_policy_distillation.py` 实现了 SGLang 模式下的 reward_func：对每条 sample 调用 teacher server，trim 到 response span 后写回 Sample。
- 官方示例用 Qwen3-8B student + Qwen3-32B teacher，在 DAPO-Math-17k 上 Math500 从 76% 提升到 94%。

**不支持 OPSD**：README 明确要求 "use a different (stronger) model as the teacher"，无 privileged context 机制。

---

#### 2. veRL（ByteDance Seed）

veRL 是目前社区活跃度最高的分布式 RL 框架之一，OPD 作为独立 trainer 提供。

**使用入口：** `examples/on_policy_distillation_trainer/`

**最小启动脚本：**

```bash
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
```

**核心配置（Hydra YAML）：**

```yaml
distillation:
  enabled: True
  teacher_models:
    teacher_model:
      model_path: 'Qwen/Qwen3-32B' # HF 路径或本地路径
  distillation_loss:
    loss_mode: 'k3' # 可选 k1 / k3 / forward_kl_topk 等
    use_policy_gradient: True # 是否与 GRPO PG loss 联合训练
    topk: 64 # teacher 只传 top-k logits，省显存
```

**关键设计：**

- teacher 通过**独立 Ray 集群** serving，与 student 训练进程解耦。
- 支持 `topk` sparse logits：teacher 只返回 top-64 的 logit 值和索引，student 用这些稀疏值算 KL，避免传输全词表 logits。
- 支持 FSDP 和 Megatron 两种训练后端，以及 vLLM 推理后端。
- 官方示例同时提供了纯文本和 VLM（`run_qwen3_vl_8b_fsdp.sh`）的 OPD 训练。

**不支持 OPSD**：官方 `on_policy_distillation_trainer` 的 `teacher_models` 配置只接受外部模型路径。第三方仓库 `HJSang/OPSD_OnPolicyDistillation` 虽然基于 veRL 构建，但其 README 明确标注 **"TODO: Add OPSD support. Currently only OPD is included"**。

---

#### 3. NeMo RL（NVIDIA）

NeMo RL 是 NVIDIA 的工业级后训练框架，OPD 以原生算法模块形式存在。

**使用入口：** `nemo_rl/algorithms/distillation.py`

**启动方式：**

```bash
python examples/run_distillation_math.py
```

**核心配置（YAML）：**

```yaml
teacher:
  model_path: 'nvidia/Nemotron-4-340B' # teacher 独立配置
  tensor_parallel_size: 4 # teacher 可用不同 TP
distillation:
  topk_logits_k: 64 # sparse top-k teacher logits
loss_fn:
  kl_type: 'reverse' # forward / reverse / mixed
```

**关键设计：**

- **两阶段执行**：Phase 1 加载 teacher 算完所有 micro-batch 的 logits，缓存到 CPU 后卸载 teacher；Phase 2 加载 student 做 forward + backward。避免 teacher 和 student 同时占显存。
- **独立并行策略**：teacher 和 student 可以使用不同的 Tensor Parallelism、Context Parallelism 配置。
- 支持 multi-turn rollout，通过 `max_rollout_turns` 配置。
- 与 NeMo 生态深度集成，适合已有 Megatron 训练基础设施的团队。

**不支持 OPSD**：`teacher: PolicyConfig` 是独立模型配置，代码中无 privileged context 或同一模型双角色逻辑。

---

#### 4. TRL（HuggingFace）

TRL 拥有最丰富的实验性 OPD trainer 集合，特点是**依赖轻、上手快**。

**使用入口：** `trl/experimental/gkd/`

**最小代码示例：**

```python
from trl import GKDTrainer, GKDConfig

trainer = GKDTrainer(
    model="Qwen/Qwen3-8B",           # student
    teacher_model="Qwen/Qwen3-32B",  # teacher（独立模型）
    args=GKDConfig(
        kl_type="reverse",           # "forward" / "reverse" / "jsd"
        temperature=1.0,
        per_device_train_batch_size=4,
    ),
    train_dataset=dataset,
)
trainer.train()
```

**关键设计：**

- `GKDTrainer` 继承自 `DPOTrainer`，架构成熟。
- 支持 forward KL、reverse KL、JSD 三种散度。
- 基于 Accelerate，可无缝切换单卡、DeepSpeed、FSDP。
- 实验性目录下还有 `minillm/`、`gold/`、`online_dpo/` 等变体，覆盖不同 OPD 算法。

**TRL 同时是以下框架的底层依赖：**

- **ms-swift**：`examples/train/rlhf/gkd/` 直接调用 TRL `GKDTrainer`，额外封装了多模态和 Megatron 适配。
- **LLaMA-Factory**：通过 TRL 集成支持 OPD，无原生独立实现。

---

#### 5. 其他原生支持 OPD 的框架

| 框架                             | 定位            | 使用入口                                        | 特点                                                                 |
| -------------------------------- | --------------- | ----------------------------------------------- | -------------------------------------------------------------------- |
| **rLLM**（UC Berkeley Sky）      | 轻量 OPD + OPSD | `rllm/trainer/distill/`                         | 单 GPU 用 tinker，多 GPU 接 verl 后端。AwesomeOPD 记录有 OPSD 支持。 |
| **AReaL**（AntGroup / Tsinghua） | 大规模 RL 框架  | `examples/distillation/gsm8k_grpo_distill.yaml` | 与蚂蚁内部训练平台对齐。                                             |
| **ROLL**（Alibaba）              | 多模态 RL 框架  | `roll/pipeline/distill/`                        | 原生支持 VLM，内置多种 divergence 库。                               |
| **SkyRL**（UC Berkeley NovaSky） | RL 研究框架     | `skyrl-train/examples/on_policy_distillation/`  | NovaSky 实验室出品，与 Sky 计算实验室生态对齐。                      |
| **KDFlow**（BJTU）               | KD-first 框架   | `examples/on_policy_kd/`                        | SGLang teacher + FSDP2 student 解耦，原生支持跨 tokenizer 和 VLM。   |

---

#### 6. 不支持 OPD 的框架

**OpenRLHF** 被 AwesomeOPD 明确排除。其架构虽然支持 rollout / teacher / update 分离，但：

- teacher 作为 remote Ray worker，传输 full logits 跨进程开销极大。
- 无原生 on-policy distillation 实现，现有蒸馏路径是离线固定语料。
- 社区多次讨论后仍未合并 OPD 原生支持。

---

### OPSD 框架详细说明

OPSD（On-Policy Self-Distillation）比 OPD 更稀缺：它要求**同一个模型**在训练过程中同时扮演 teacher 和 student，teacher 通过 privileged context（ground-truth 答案、额外指令、更长上下文等）获得学生看不到的信息，然后对学生的 rollout 逐 token 打分。

#### 1. TRL（HuggingFace）—— 目前唯一官方实验性实现

**使用入口：** `trl/experimental/self_distillation/`

**核心机制：**

- `SelfDistillationMixin._split_prompt_and_privileged_context()` 从 batch 里分离出 `prompt` 和 `privileged_context`。
- 同一 model 跑两次 forward：
  - student forward：`prompt + completion`
  - teacher forward：`prompt + privileged_context + completion`（teacher 看到更多信息）
- 计算 reverse KL：`KL(teacher || student)`，只对学生生成部分算 loss。

**关键类：**

- `BaseSelfDistillationTrainer`：在线自蒸馏基类，支持 vLLM rollout。
- `SelfDistillationMixin`：公共 loss 计算，支持 `grpo`、`bnpo`、`dr_grpo`、`dapo` 等 loss type。
- `SDPO`（Self-Distillation Policy Optimization）：具体 trainer 实现。

**数据格式要求：**
数据集必须包含 `prompt` 和 `privileged_context` 两列。例如数学题中，`privileged_context` 可以是 ground-truth 解答或推导提示。

**代码片段：**

```python
from trl import SDPOTrainer, SelfDistillationConfig

trainer = SDPOTrainer(
    model="Qwen/Qwen3-8B",
    args=SelfDistillationConfig(
        kl_type="reverse",
        loss_type="grpo",      # 或 bnpo / dapo 等
    ),
    train_dataset=dataset,     # 需包含 "prompt" 和 "privileged_context"
)
trainer.train()
```

**支持的散度：**

- `alpha=0`：reverse KL（$D_{KL}(teacher || student)$）
- `alpha=1`：forward KL（$D_{KL}(student || teacher)$）
- `0 < alpha < 1`：JSD 混合

**限制：** 实验性代码，API 可能变动；目前只支持单模型架构，无多 teacher。

---

#### 2. rLLM（UC Berkeley Sky）

AwesomeOPD 记录 rLLM 在 `examples/math_distill/` 下有 OPSD 实现（含 `opsd/` 子目录）。该框架定位轻量，适合：

- 单 GPU 快速验证（tinker 后端）
- 多 GPU 扩展（verl 后端）

但当前 GitHub 路径可能已迁移，建议以最新仓库为准。

---

#### 3. 明确不支持 OPSD 的框架

| 框架                         | 原因                                                                        |
| ---------------------------- | --------------------------------------------------------------------------- |
| **slime**                    | 架构上要求 `--opd-teacher-load` 指向独立模型，无 privileged context 接口。  |
| **veRL**                     | 官方配置 `teacher_models.teacher_model.model_path` 只接受外部模型路径。     |
| **NeMo RL**                  | `teacher: PolicyConfig` 为独立模型配置块，代码无同一模型双角色逻辑。        |
| **ms-swift / LLaMA-Factory** | 通过 TRL GKDTrainer 间接支持 OPD，TRL 的 self_distillation 模块尚未被封装。 |
| **OpenRLHF**                 | 无原生 OPD，更无 OPSD。                                                     |

### 选型建议

- **需要 OPD + 大规模分布式训练**（SGLang / Megatron / vLLM teacher server）：选 **slime**、**veRL** 或 **NeMo RL**。三者都是生产级实现，slime 和 veRL 的社区活跃度更高。
- **需要 OPSD（自己教自己）**：目前只有 **TRL** 有官方实验性实现。如果已在 TRL 生态内，可直接复用 `self_distillation/` 模块。
- **已有 SWIFT / ModelScope 或 LLaMA-Factory 工作流**：OPD 可通过 TRL GKDTrainer 间接使用，但 OPSD 暂无支持。
- **只是想快速验证 OPD 机制**：**TRL** 的 `GKDTrainer` 或 **veRL** 的 `on_policy_distillation_trainer` 都是最小依赖的起步选择。

## 本章总结

OPD 的核心是一个训练范式选择：

- off-policy 蒸馏有密集 token 监督，但不训练 student 自己会犯的错。
- RL 是 on-policy，但奖励常常稀疏，样本效率低。
- OPD 把两者接起来：student 自己采样，teacher 逐 token 给密集反馈。

它特别适合小模型、专用模型和后训练能力迁移。但 OPD 不是 RL 的替代品，也不是万能压缩器——上限来自 teacher，稳定性来自初始化，价值来自"在 student 自己状态上给密集信号"。

2026 年机制分析论文最值得带走的 insight，是把 OPD 的核心问题从"teacher 是否更强"换成"teacher 是否更可学"。更强的模型可能只是会在自己的轨迹上拿高分；更可学的 teacher 则能在 student 当前轨迹上提供新知识、相近思维路径和可吸收的局部偏好。这个视角会改变你设计整条蒸馏 pipeline 的方式：先冷启动让两者说同一种语言，再用 teacher-aligned prompt 让 teacher 站到熟悉的分布上，最后用任务奖励保证局部偏好没有背离全局目标。

从本章主线看，DPO、GRPO、RLVR 和 OPD 都在回答同一个问题：**当我们不想完整跑传统 RLHF 时，训练信号还能从哪里来？** DPO 用偏好对，RLVR 用验证器，OPD 用 teacher。理解这三类信号的边界，才是真正能迁移到新项目里的能力。

## 参考文献

[^kd_survey_xu]: Xu X, Li M, Tao C, et al. [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116), arXiv 2024.（从 algorithm、skill、verticalization 三个角度整理 LLM KD）

[^kd_survey_yang]: Yang C, Lu W, Zhu Y, et al. [Survey on Knowledge Distillation for Large Language Models: Methods, Evaluation, and Application](https://arxiv.org/abs/2407.01885), arXiv 2024.（按 white-box / black-box KD、评测和应用整理）

[^opd_survey]: Song M, Zheng M. [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626), arXiv 2026.（把 OPD 统一到 f-divergence 框架，并按反馈信号、teacher access、loss 粒度分类）

[^dagger]: Ross S, Gordon G, Bagnell D. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.（DAgger：让 learner 访问到的状态进入训练集）

[^gkd]: Agarwal R, Vieillard N, Zhou Y, et al. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), ICLR 2024.（GKD：在 student 自生成序列上用 teacher 反馈缓解分布偏移）

[^minillm]: Gu Y, Dong L, Wei F, Huang M. [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), ICLR 2024.（用 reverse KL 和 on-policy 优化做生成式 LLM 蒸馏）

[^distillm]: Ko J, Kim S, Chen T, Yun S. [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://proceedings.mlr.press/v235/ko24c.html), ICML 2024.（用 skew KL 和 adaptive off-policy 提升 LLM 蒸馏效率）

[^tml_opd]: Lu K, Thinking Machines Lab. [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.（工程化 OPD 复现与 Tinker 实现，包含 Qwen3 对比和 personalization 实验）

[^rethinking_opd]: Li Y, Zuo Y, He B, et al. [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016), arXiv 2026.（分析 OPD 成功条件、token-level 机制和失败恢复策略）

[^lightning_opd]: Shi Z, Zhang J, Jiang W, et al. [Lightning OPD: Cost-effective On-Policy Distillation](https://arxiv.org/html/2604.13010v1), arXiv 2026.（把标准 OPD 离线化：预计算 teacher log-prob，避免训练时 live teacher server）

[^self_distilled_reasoner]: Zhao S, Xie Z, Liu M, et al. [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734), arXiv 2026.（单模型在不同上下文下同时扮演 teacher 和 student）

[^blackbox_opd]: Ye T, Dong L, Chi Z, et al. [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643), arXiv 2025.（GAD：没有 teacher logits 时，用 discriminator 提供 on-policy 奖励）
