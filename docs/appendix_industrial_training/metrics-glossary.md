# B.8 大模型 RL 训练指标词典

> 你第一次打开训练日志或监控面板的时候，可能会被几十个指标吓一跳——`actor/pg_clipfrac`、`critic/advantages/mean`、`timing_s/gen`、`perf/mfu/actor`……这么多数字，到底看哪个？
>
> 别着急。想象你在观察一个学生准备考试：**最终考试成绩**是你最关心的；**每天作业的正确率**告诉你他有没有在进步；**每天学习多长时间**告诉你效率高不高；**错题本的分布**告诉你他哪里薄弱。训练指标也是一样的道理——每个指标都在讲述训练过程的一个侧面。
>
> 这份词典把这些指标按"你最先该看什么"的顺序组织。适用于 veRL、OpenRLHF、TRL 等主流框架，实际字段名可能略有差异（文末附对照表）。

---

## 1. 验证指标 — "期末考试考了多少分"

无论你用什么算法、调了什么超参，最终只有一个问题真正重要：**模型变好了吗？**

验证指标就是回答这个问题的。你可以在训练过程中定期跑一次验证集，看模型在没见过的数据上表现如何——就像模拟考试不能代替期末考试，但能给你一个靠谱的估计。

| 指标                      | 它在说什么                                                                    |
| ------------------------- | ----------------------------------------------------------------------------- |
| `val-core/*/acc/mean@1`   | 验证集准确率——模型答对了多少。这是你最应该盯的数字                            |
| `val-aux/*/reward/mean@1` | 验证集的平均奖励。对于规则奖励（比如 GSM8K 答案对不对），通常和准确率高度相关 |
| `val-aux/num_turns/*`     | 验证阶段的对话轮数统计                                                        |

其中 `val-core` 是主指标（准确率），`val-aux` 是辅助指标（奖励、轮数等）。先看 `val-core`，再看 `val-aux`。

**怎么看**：理想情况下，`val-core` 的准确率应该随训练步数稳步上升。如果它开始下降或者长时间不动，你就需要往回翻——去看是训练过程出了问题（第 2 节），还是数据本身有问题（第 4 节）。

---

## 2. Actor 指标 — "每天的作业和复习情况"

确认了模型在变好（或没变好）之后，下一步是理解**训练过程本身健不健康**。Actor 指标就像学生的日常学习记录：作业正确率、做题速度、注意力集中程度——它们共同告诉你学习过程是否正常。

### 2.1 loss：模型在努力缩小"预测"和"目标"的差距

| 指标                 | 它在说什么                                                                      |
| -------------------- | ------------------------------------------------------------------------------- |
| `actor/loss`         | Actor 的总损失，一般是 `pg_loss + 正则项`。这是反向传播时真正在优化的那个数     |
| `actor/pg_loss`      | 策略梯度（policy gradient）的核心损失。可以理解为"这一步策略优化本身在优化什么" |
| `actor/entropy_loss` | 熵正则项的损失。用来防止模型太快变得"太确定"                                    |
| `actor/kl_loss`      | KL 散度损失项（需要 `use_kl_loss=True` 才会启用）。用来限制策略偏离参考模型太远 |
| `actor/kl_coef`      | KL 损失的权重系数                                                               |

**怎么看**：`actor/loss` 总体应该呈下降趋势。如果突然跳升，通常意味着学习率太大、数据分布突变、或者策略在剧烈震荡。`pg_loss` 不一定单调下降（策略梯度的损失曲线本身就比较嘈杂），但持续飙升肯定有问题。

### 2.2 梯度范数：更新幅度有没有失控

| 指标              | 它在说什么                                       |
| ----------------- | ------------------------------------------------ |
| `actor/grad_norm` | 梯度的 L2 范数。可以理解为"这一步参数想改变多大" |

**怎么看**：想象你在爬山，梯度就是山的坡度。`grad_norm` 就是坡度的绝对大小——坡度太大你会滚下去（梯度爆炸），坡度太小你走不动（梯度消失）。训练稳定时 `grad_norm` 应该保持在合理范围内波动。如果突然飙升到正常值的 10 倍以上，就该警觉了。

### 2.3 熵：模型有多"纠结"

| 指标            | 它在说什么           |
| --------------- | -------------------- |
| `actor/entropy` | 策略输出分布的平均熵 |

想象一个学生做选择题：如果他对每个选项都犹豫不决（均匀分配概率），熵就高；如果他胸有成竹地锁定一个选项（概率集中在某一个），熵就低。

- **训练初期，熵高是正常的**：模型还在探索，对各种回答都愿意试试
- **训练后期，熵低是正常的**：模型已经形成了比较确定的策略
- **熵突然骤降要警惕**：模型可能在几步之内就"锁死"到了一个狭窄的输出空间，这是策略坍缩的信号，往往伴随着 reward hacking（模型找到了"骗"奖励的捷径）
- **熵一直不降**：模型可能根本没学到东西

### 2.4 KL 散度：策略偏离起点有多远

| 指标           | 它在说什么                                                 |
| -------------- | ---------------------------------------------------------- |
| `actor/ppo_kl` | 当前策略和参考策略（通常是训练前的模型）之间的 KL 散度近似 |

想象你让学生在原有知识基础上学习新内容。KL 散度衡量的是"新学的东西和原来差别有多大"。差别太小说明没学到新东西，差别太大说明可能学偏了。

**怎么看**：KL 应该缓慢、稳定地增长。如果突然飙升（比如从 0.02 跳到 0.15），说明策略在急转弯——这通常不妙。一般 PPO 训练中会把 KL 控制在一个合理范围（比如 0.01~0.1），超过这个范围就该考虑调低学习率或加大 KL 惩罚。

### 2.5 Clip Fraction：有多少更新被"限速"了

| 指标                      | 它在说什么                                           |
| ------------------------- | ---------------------------------------------------- |
| `actor/pg_clipfrac`       | PPO 裁剪机制被触发的比例——有多少比例的更新撞到了上界 |
| `actor/pg_clipfrac_lower` | 撞到下界的比例                                       |

PPO 用裁剪（clipping）来限制每次策略更新的幅度，就像汽车的限速器——你踩油门踩到底，但速度不会超过上限。clipfrac 就是"有多少次踩到了限速"。

**怎么看**：正常范围大概在 0.1~0.3。如果持续高于 0.3，说明策略每步都想迈很大的步子但被 clip 拉回来了——这时候应该降低学习率或减小 batch size。如果 clipfrac 长期接近 0，可能说明策略已经收敛或学习率太小。

### 2.6 学习率

| 指标       | 它在说什么          |
| ---------- | ------------------- |
| `actor/lr` | 当前 actor 的学习率 |

**怎么看**：确认学习率是否按你设定的 schedule 在走（比如 warmup + cosine decay）。有时候配置写错了，学习率全程不变或者突然归零，看这个指标能第一时间发现。

---

## 3. Critic / 奖励统计 — "老师打的分"

训练 RL 就像训练一个学生：Actor 是学生本人（负责做决策），Critic 是老师（负责评估"这个决策有多好"）。不过，PPO 和 GRPO 的情况不太一样。

### PPO 场景：有独立的"老师"

PPO 有一个独立的 Critic 网络，它学习预测每个状态的期望回报。

| 指标                     | 它在说什么                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| `critic/v_loss`          | Critic 的预测误差——老师估分准不准                                                                        |
| `critic/vpred/mean`      | Critic 预测的 value 均值                                                                                 |
| `critic/vpred/var`       | Critic 预测的方差                                                                                        |
| `critic/score/mean`      | 序列级别分数的统计                                                                                       |
| `critic/rewards/mean`    | 序列级别奖励的统计——这批回答整体好不好                                                                   |
| `critic/advantages/mean` | 优势值（由 GAE 计算）。这个值直接决定 Actor 怎么更新——正值表示"比预期好，多做"，负值表示"比预期差，少做" |
| `critic/returns/mean`    | 回报统计                                                                                                 |

**怎么看**：`critic/v_loss` 应该稳步下降（老师在越来越准）。`critic/rewards/mean` 应该稳步上升（模型在越做越好）。如果 rewards 在涨但 `v_loss` 不降，说明 Critic 跟不上 Actor 的变化速度，可能需要调高 Critic 的学习率。

### GRPO 场景：没有"老师"，只有"同学互评"

GRPO 不训练 Critic 网络（你可能会在日志里看到 `Disabled critic as algorithm.adv_estimator != gae`）。但日志里 `critic/rewards/mean`、`critic/advantages/mean` 这些字段仍然会出现——此时它们只是 batch 统计量，反映的是当前一批样本的奖励和优势分布，**不代表有一个 Critic 网络在训练**。

---

## 4. 长度指标 — "回答写得太长还是太短"

长度指标常常被忽视，但它们能告诉你很多关于训练状态的信息。

想象你让学生写作文：如果所有人都卡着字数上限交，说明要么题目要求他们写很多，要么字数上限设太低了；如果作文越来越短但分数越来越高，说明学生可能找到了"凑分"的技巧。

### 回复长度

| 指标                            | 它在说什么                                                  |
| ------------------------------- | ----------------------------------------------------------- |
| `response_length/mean`          | 生成回复的平均 token 数                                     |
| `response_length/max`           | 最长回复                                                    |
| `response_length/min`           | 最短回复                                                    |
| `response_length/clip_ratio`    | 有多少比例的回复撞到了 `max_response_length` 上限——被截断了 |
| `response_length_non_aborted/*` | 排除异常样本后的回复长度                                    |
| `response/aborted_ratio`        | 有多少比例的回复被中断或无效                                |

### Prompt 长度

| 指标                       | 它在说什么                                                    |
| -------------------------- | ------------------------------------------------------------- |
| `prompt_length/mean`       | 输入 prompt 的平均 token 数                                   |
| `prompt_length/max`        | 最长 prompt                                                   |
| `prompt_length/min`        | 最短 prompt                                                   |
| `prompt_length/clip_ratio` | 有多少比例的 prompt 撞到了 `max_prompt_length` 上限——被截断了 |

### 怎么看

- **`clip_ratio` 高**：很多回复或 prompt 被截断了。对于 response，考虑增大 `max_response_length`；对于 prompt，检查输入数据是不是太长了
- **`response_length/mean` 在下降 + reward 在上升**：模型可能在 reward hacking——学会用更短的回答"骗"到奖励，而不是真正提升回答质量
- **`response/aborted_ratio` 高**：很多样本生成失败。可能是采样参数设置有问题，或者模型输出了 EOS 以外的终止信号

---

## 5. DPO / KTO / SimPO 指标 — "偏好学习的成绩单"

前面几节主要讲 PPO 和 GRPO——它们都需要模型自己生成回答（on-policy）。DPO 家族不一样：它们直接从人类标注的"好回答 vs 坏回答"对中学习偏好，不需要在线采样。所以指标体系也不同。

### DPO 核心指标

| 指标               | 它在说什么                                          |
| ------------------ | --------------------------------------------------- |
| `loss/dpo`         | DPO 的总损失。应该稳步下降                          |
| `rewards/chosen`   | 对"好回答"的隐式奖励。应该为正且逐步增大            |
| `rewards/rejected` | 对"坏回答"的隐式奖励。应该为负或明显低于 chosen     |
| `rewards/margins`  | chosen 和 rejected 的奖励差——模型区分好坏回答的能力 |
| `rewards/accuracy` | 模型正确偏好 chosen 的比例                          |
| `logps/chosen`     | 对 chosen 回复的 log 概率                           |
| `logps/rejected`   | 对 rejected 回复的 log 概率                         |

**怎么看**：`rewards/margins` 是 DPO 最核心的指标——它衡量模型在多大程度上能把好回答和坏回答区分开。这个值应该稳步增大。但如果它持续增大、同时验证集效果不涨，说明模型可能在**过拟合训练对**——记住了这些特定的偏好对，而不是学到了通用的偏好判断。

### KTO 和 SimPO

- **KTO**：不需要成对偏好数据，只需要知道每条回答是"好"还是"坏"。额外关注 `kl_estimate`（策略偏离参考模型的程度）和 `loss/kl`（KL 正则损失）
- **SimPO**：不需要参考模型，指标类似 DPO 但没有 reference 相关字段，多了 length-normalized reward 的统计

---

## 6. Reward Model 训练指标 — "裁判的培训记录"

如果你在做 PPO 训练，通常需要先训练一个 Reward Model（RM）——它扮演"裁判"的角色，给模型的回答打分。RM 本身的训练也有指标需要关注。

| 指标                      | 它在说什么                                    |
| ------------------------- | --------------------------------------------- |
| `rm/loss`                 | RM 的偏好预测损失（交叉熵）。应该稳步下降     |
| `rm/accuracy`             | RM 正确判断"哪个回答更好"的准确率             |
| `rm/reward_margin`        | RM 给好回答和坏回答的分数差距——裁判有多"果断" |
| `rm/reward_chosen/mean`   | RM 给好回答的平均分                           |
| `rm/reward_rejected/mean` | RM 给坏回答的平均分。应该明显低于 chosen      |
| `rm/grad_norm`            | RM 的梯度范数                                 |

### 怎么判断 RM 是不是在正常工作

一个好的裁判应该满足两个条件：**判断准确**（accuracy 高）和**果断**（margin 大，好就是好、坏就是坏）。

但如果出现以下情况，说明裁判可能出了问题：

- **accuracy 很高，但下游 PPO 评测不涨**：RM 过拟合了训练数据中的偏好模式，但没学到真正通用的判断标准
- **margin 趋近 0**：RM 变成了"和稀泥"的裁判，无法区分好坏回答
- **用旧版 RM 给当前策略打分，和用新版 RM 差异很大**：RM 可能被当前策略"污染"了——如果 RM 的训练数据中混入了当前策略的输出，RM 就在"自己给自己打分"

---

## 7. 多轮交互指标 — "对话进行了几轮"

| 指标                  | 它在说什么           |
| --------------------- | -------------------- |
| `num_turns/min`       | 样本中最少的对话轮数 |
| `num_turns/max`       | 样本中最多的对话轮数 |
| `num_turns/mean`      | 平均对话轮数         |
| `val-aux/num_turns/*` | 验证阶段的轮数统计   |

**怎么看**：对于普通的 GSM8K 单轮问答，每个样本通常是 2 轮（1 个 user + 1 个 assistant）。如果你在做 Agentic RL（多轮工具调用），轮数会高很多。如果 `num_turns/max` 突然变得非常高，可能说明某些样本陷入了死循环。

---

## 8. 负载均衡指标 — "活儿分得均不均"

前面几节关注的是"模型学得好不好"。从这节开始，我们关注的是"训练跑得快不快"。

`global_seqlen/*` 系列指标衡量的是**多卡训练时，工作负载分配是否均衡**。它表示的是每个 GPU（partition / rank）分到的总 token 工作量，不是某一条句子的长度。

| 指标                         | 它在说什么                                  |
| ---------------------------- | ------------------------------------------- |
| `global_seqlen/min`          | 工作量最轻的那张卡，分到了多少 token        |
| `global_seqlen/max`          | 工作量最重的那张卡，分到了多少 token        |
| `global_seqlen/minmax_diff`  | `max - min`。这个值越小，说明活儿分得越均匀 |
| `global_seqlen/balanced_min` | 做完负载均衡后最轻的卡                      |
| `global_seqlen/balanced_max` | 做完负载均衡后最重的卡                      |
| `global_seqlen/mean`         | 每张卡的平均工作量                          |

**怎么看**：`minmax_diff` 是最值得关注的。想象一个团队做项目：如果一个人做了 80% 的工作，其他人闲着等他，整体进度就被拖慢了。多卡训练也是一样——`minmax_diff` 太大意味着各卡之间负载不均衡，快的卡要等慢的卡，整体效率就低了。

---

## 9. 时间指标 — "每一步的时间花在了哪里"

| 指标                      | 它在说什么                                                           |
| ------------------------- | -------------------------------------------------------------------- |
| `timing_s/step`           | 整个 step 的总耗时                                                   |
| `timing_s/gen`            | 生成回复花了多少秒。通常是时间大头                                   |
| `timing_s/reward`         | 计算 reward 花了多少秒。规则奖励（如答案对不对）很快，用模型打分就慢 |
| `timing_s/old_log_prob`   | 计算旧策略的 log probability 和 entropy 花了多少秒                   |
| `timing_s/ref`            | 参考模型计算 log probability 花了多少秒                              |
| `timing_s/adv`            | 计算优势值花了多少秒                                                 |
| `timing_s/update_actor`   | 更新 Actor 参数花了多少秒                                            |
| `timing_s/update_critic`  | 更新 Critic 参数花了多少秒（PPO 才有，GRPO 没有）                    |
| `timing_s/update_weights` | 把更新后的权重同步给生成引擎花了多少秒                               |

**怎么看**：先看 `timing_s/step` 了解整体节奏，再逐项看哪个环节是瓶颈。大部分情况下 `timing_s/gen`（生成）是大头。如果你想优化训练速度，先优化占用时间最长的环节。

### 更细粒度的时间

对于 Agentic 场景，还有更细的时间拆分：

- `timing_s/agent_loop/generate_sequences/min|max|mean`：单样本生成时间的分布
- `timing_s/agent_loop/tool_calls/min|max|mean`：工具调用的耗时
- `timing_s/agent_loop/slowest/*`：最慢那条样本的细节

### 按 token 归一化的时间

`timing_per_token_ms/*` 把时间除以 token 数，得到"每个 token 花多少毫秒"。这组指标适合**跨 run 比较**——比如 PPO vs GRPO 谁的生成效率更高，用这个指标比直接比 `timing_s/gen` 更公平（因为两次 run 可能生成了不同数量的 token）。

常见字段：`timing_per_token_ms/gen`、`timing_per_token_ms/ref`、`timing_per_token_ms/adv`、`timing_per_token_ms/update_actor`。

---

## 10. 性能指标 — "GPU 用得满不满"

| 指标                    | 它在说什么                                  |
| ----------------------- | ------------------------------------------- |
| `perf/total_num_tokens` | 这一步总共处理了多少 token                  |
| `perf/time_per_step`    | 一步的总耗时                                |
| `perf/throughput`       | 每秒每卡处理多少 token——最核心的效率指标    |
| `perf/mfu/actor_infer`  | Actor 在推理阶段的 MFU（模型 FLOPs 利用率） |
| `perf/mfu/actor`        | Actor 在训练更新阶段的 MFU                  |

### MFU 是什么

MFU 衡量的是"GPU 的计算能力实际被用上了多少"。你可以把它理解成工厂的产能利用率——如果工厂能产 100 件/小时但只产了 40 件，利用率就是 40%。

正常训练中 MFU 通常在 50%~60% 左右。低于 30% 说明 GPU 在大量等待（可能是通信瓶颈、数据加载瓶颈、或者负载不均衡），有优化空间。

---

## 11. 框架差异提示

不同框架对同一个指标可能用不同的字段名。核心含义是一样的，只是"叫法"不同。

| 含义       | veRL / OpenRLHF        | TRL                             |
| ---------- | ---------------------- | ------------------------------- |
| 策略熵     | `actor/entropy`        | `policy/approx_kl` 或 `entropy` |
| KL 散度    | `actor/ppo_kl`         | `objective/kl`                  |
| 梯度范数   | `actor/grad_norm`      | `loss/total` 里的梯度监控       |
| 回复长度   | `response_length/mean` | `response_length`               |
| 生成耗时   | `timing_s/gen`         | 通常不单独记录                  |
| 奖励均值   | `critic/rewards/mean`  | `rewards`                       |
| DPO 奖励差 | `rewards/margins`      | `rewards/margins`（一致）       |

如果你用的是 TRL 的 `SFTTrainer` + `DPOTrainer`，指标会自动打到 wandb / tensorboard，字段名参考 TRL 文档的 Logging 章节。

---

## 12. 快速参考：按场景看什么指标

### PPO / GRPO 训练

1. **先看效果**：`val-core/*/acc/mean@1`（准确率）、`critic/rewards/mean`（奖励趋势）
2. **再看稳定性**：`actor/loss`、`actor/grad_norm`、`actor/ppo_kl`、`actor/pg_clipfrac`
3. **再看效率**：`perf/throughput`、`timing_s/gen`
4. **最后看数据**：`response_length/mean`、`response_length/clip_ratio`、`global_seqlen/minmax_diff`

### DPO / KTO 训练

1. **先看效果**：`rewards/margins`（区分能力）、`rewards/accuracy`（偏好准确率）
2. **再看稳定性**：`loss/dpo`（损失趋势）
3. **注意过拟合**：`rewards/accuracy` 到 99%+ 但评测不涨 → 过拟合了

### Reward Model 训练

1. **先看效果**：`rm/accuracy`（偏好预测准确率）、`rm/reward_margin`（区分能力）
2. **再看泛化**：验证集 accuracy 是否跟训练集一起涨

### 训练出问题了

| 现象                 | 先查                            | 再查                                           |
| -------------------- | ------------------------------- | ---------------------------------------------- |
| Reward 暴跌          | `actor/entropy` 是否骤降        | `actor/pg_clipfrac` 是否飙升                   |
| KL 飙升              | `actor/ppo_kl` 趋势             | 学习率 schedule 是否正确                       |
| 评测不涨但 reward 涨 | `response_length/mean` 是否在变 | `response/aborted_ratio` 是否偏高              |
| 训练很慢             | `timing_s/step` 各子项          | `perf/throughput`、`global_seqlen/minmax_diff` |
| 显存 OOM             | `global_seqlen/max` 是否过大    | `response_length/max`、`prompt_length/max`     |
