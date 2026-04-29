# 知识蒸馏——从大模型到小模型的知识迁移

上一节我们讨论了 RL Scaling 的三个维度和训练范式的选择。但你可能已经注意到一个盲区：**三种训练范式（DPO / GRPO / Iterative DPO）的训练信号要么来自人类偏好，要么来自可验证奖励。如果既没有偏好数据，也没有标准答案，只有一个现成的强模型，能不能直接用它来训练小模型？**

这个问题的答案是**知识蒸馏（Knowledge Distillation）**。更具体地说，2025-2026 年工业界大规模采用的**在线策略蒸馏（On-Policy Distillation, OPD）**已经成为了 LLM 后训练的第四条核心路线。Qwen3 用它训练了 0.6B 到 30B 的全部小模型，DeepSeek-R1 的蒸馏版超过了直接 RL 训练的小模型，GLM-5 用它防止多阶段遗忘，MiMo-V2-Flash 用多个 teacher 同时蒸馏。这一节我们就来拆解 OPD 到底在做什么、为什么它本质上就是 RL、以及工业界是怎么用的。

## 从一个类比说起：三种学习方式

想象你在学一门新课程。有三种学习方式：

**方式一：抄笔记（SFT）**。老师写了一份完美的课堂笔记，你逐字抄一遍。好处是快、简单；坏处是你只学会了老师的表面写法，遇到新题还是不会。这就是 SFT——直接在 teacher 模型的输出上做监督微调。

**方式二：做老师做过的题（Off-policy 蒸馏）**。老师把自己做过的题和答案给你，你对照着学。比抄笔记好一些——你在做题中学，但题目是老师选的，不是你自己遇到困难的题。这就是 off-policy 蒸馏——在 teacher 生成的轨迹上训练 student。

**方式三：自己做题，老师逐题批改（On-policy 蒸馏）**。你自己去做题，做完了老师告诉你每一步写得好不好、应该怎么改。你学的是**你自己遇到的问题**，老师的反馈是**针对你的错误**的。这就是 OPD——student 在自己生成的文本上学习，teacher 逐 token 地给出反馈。

OPD 的核心区别就一个：**student 在自己生成的文本上学习**。这为什么重要？因为 student 在推理时看到的是自己生成的文本，如果训练时只见过 teacher 的文本，就会遇到**分布偏移（distribution shift）**——就像"只在柏油路上练车，考试却要开山路"。OPD 通过让 student 在自己的轨迹上学习，直接消除了这个 gap。

## OPD 的数学本质——它就是 RL

前面几章我们反复看到，很多看似不同的方法本质上都是 RL。OPD 也不例外。把它的目标函数拆开看：

$$J_{\text{OPD}}(\theta) = -\mathbb{E}_{y \sim \pi_\theta}\left[\underbrace{\log q(y|x)}_{\text{teacher 认可度}} + \underbrace{\mathcal{H}(\pi_\theta)}_{\text{熵正则}}\right]$$

其中 $\pi_\theta$ 是 student 模型，$q$ 是 teacher 模型。翻译成我们在第 5-6 章学过的 RL 语言：

| RL 概念                          | OPD 对应                                                |
| -------------------------------- | ------------------------------------------------------- |
| 策略 $\pi_\theta$                | Student 模型                                            |
| 状态 $s_t$                       | 当前 prompt + 已生成的 token（上下文 $c_t$）            |
| 动作 $a_t$                       | 下一个 token $y_t$                                      |
| 奖励 $r_t$                       | $\log q(y_t \mid c_t)$——teacher 认为这个 token 有多合理 |
| 熵正则 $\mathcal{H}(\pi_\theta)$ | 防止 student 坍缩，保持生成多样性                       |

**奖励函数不是人设计的，也不是 RM 学的，而是直接用 teacher 的 log-prob。** 这和 GRPO 的核心区别在于奖励的密度——GRPO 的奖励是稀疏的（只在回答结束后才有），OPD 的奖励是密集的（每个 token 都有）。不需要训练 Reward Model，不需要标准答案，只要 teacher 能做一次 forward，你就有了完整的训练信号。

到这里，我们可以把 LLM 后训练的训练信号来源放在一张全景图里：

| 方法        | 训练信号                    | 需要什么               | 类比                 |
| ----------- | --------------------------- | ---------------------- | -------------------- |
| RLHF / PPO  | 人类偏好 Reward Model       | 标注偏好数据 + 训练 RM | 请人类评委打分       |
| DPO         | 人类偏好数据（隐式 RM）     | 标注偏好数据           | 给评委一对一对地比   |
| GRPO / RLVR | 可验证奖励（答案对不对）    | 有标准答案的任务       | 自动阅卷             |
| **OPD**     | **Teacher 模型的 log-prob** | **一个现成的强模型**   | **请优等生当小老师** |

OPD 的门槛最低——只需要一个现成的强模型。Qwen3 的实验表明，对小模型来说，**蒸馏的效果等同甚至超过 RL，但只需要 1/10 的 GPU 时间**。

## KL 散度的方向为什么重要

知识蒸馏的核心是比较 teacher 和 student 的概率分布。但 KL 散度有两个方向，选哪个会产生完全不同的行为。这个选择在 2024 年之前很少有人关注，直到 MiniLLM（ICLR 2024）系统地论证了：**LLM 蒸馏应该用 reverse KL，而不是 forward KL**。

先回顾一下两个方向的区别。KL 散度 $D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$ 是不对称的：

**Forward KL**（$D_{\text{KL}}(q \| \pi_\theta)$，teacher 分布在前）要求 student 覆盖 teacher 的所有高概率区域。这就像学生想把老师的每一句话都记住，包括老师随口说的无关内容。结果 student 会给一些其实不太需要的 token 也分配概率，导致生成质量下降。这就是经典的 KD 方法（Hinton 2015）在分类任务上的做法，对 BERT 蒸馏效果很好，但对 LLM 生成长文本时会出问题——student 会生成很多低质量的长尾内容。

**Reverse KL**（$D_{\text{KL}}(\pi_\theta \| q)$，student 分布在前）只要求 student 集中在 teacher 的某个高概率区域。学生只抓住老师的核心思路，不纠结细枝末节。对于 LLM 来说这是更好的选择——student 不会去模仿 teacher 的低概率 token，生成更精确、更聚焦。

| KL 方向        | 行为                     | 适合场景            | 代表方法              |
| -------------- | ------------------------ | ------------------- | --------------------- |
| Forward KL     | Mode-covering（覆盖）    | 分类任务、BERT 蒸馏 | 经典 KD (Hinton 2015) |
| **Reverse KL** | **Mode-seeking（聚焦）** | **生成式 LLM 蒸馏** | **MiniLLM, GKD**      |

但 reverse KL 有一个代价：它的期望是关于 student 分布的（$\mathbb{E}_{y \sim \pi_\theta}$），需要用策略梯度（类似 PPO/REINFORCE）来优化，方差较高。MiniLLM 用 PPO 式的优化来处理这个问题，GKD（Google DeepMind, ICLR 2024）则提出了更灵活的散度选择（可以用 reverse KL、forward KL 或 JSD），并引入了 on-policy 和 off-policy 数据的混合策略。DistiLLM（ICML 2024）进一步用 skewed KL 降低了梯度方差。这三篇构成了 OPD 的理论基础。

## 最简单的 OPD：动手试试

理解了原理之后，我们用一个最简化的例子来实际感受 OPD 的训练信号。思路只有三步：student 生成回答 → teacher 对每个 token 打分 → 计算 reward。

```python
# ==========================================
# 最简 OPD：感受 teacher-student log-prob 差
# ==========================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 student（小）和 teacher（大）
student_name = "Qwen/Qwen2.5-0.5B-Instruct"
teacher_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(student_name)
student = AutoModelForCausalLM.from_pretrained(
    student_name, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher.eval()

# Step 1: student 自己生成回答（on-policy 采样）
prompt = "Solve: if x + 3 = 7, what is x? Show your work."
inputs = tokenizer(prompt, return_tensors="pt").to(student.device)

with torch.no_grad():
    output_ids = student.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
print("Student:", tokenizer.decode(gen_ids, skip_special_tokens=True))

# Step 2: teacher 和 student 分别计算每个 token 的 log-prob
full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
with torch.no_grad():
    s_logits = student(full_ids).logits[0]   # [seq_len, vocab]
    t_logits = teacher(full_ids).logits[0]

# Step 3: 只在生成的 token 上计算 reward
gen_start = inputs["input_ids"].shape[1]
for t in range(gen_start, full_ids.shape[1] - 1):
    tok = full_ids[0, t + 1]
    s_logp = torch.log_softmax(s_logits[t], dim=-1)[tok].item()
    t_logp = torch.log_softmax(t_logits[t], dim=-1)[tok].item()
    reward = t_logp - s_logp  # teacher 更认可 → 正 reward
    print(f"  token={tokenizer.decode([tok])!r:10s}  "
          f"teacher={t_logp:+.3f}  student={s_logp:+.3f}  reward={reward:+.3f}")
```

运行这段代码，你会看到每个 token 的 teacher 和 student log-prob，以及它们的差作为 reward。关键观察：

1. **大部分 token 的 reward 是负的**——student 的概率通常高于 teacher（因为 student 更"自信"但不够好），所以 $\log \pi_\theta - \log q > 0$，即 reward = $\log q - \log \pi_\theta < 0$
2. **少数 token 是正的**——这些是 teacher 比 student 更认可的 token，也就是 student 需要强化学习的方向
3. **这就是 OPD 的全部训练信号**——不需要 RM，不需要标准答案，只要 teacher 的一次 forward

在实际训练中，你会用这些 reward 做 policy gradient 更新（和第 5 章的 REINFORCE 或第 6 章的 PPO 一样的框架），让 student 逐步逼近 teacher 的分布。完整的训练循环需要加入裁剪（PPO clip）、entropy bonus 等稳定化技巧，感兴趣可以参考 Thinking Machines Lab 提供的[端到端复现代码](https://thinkingmachines.ai/blog/on-policy-distillation/)。

## 工业界怎么用 OPD

OPD 已经不只是学术概念。2025-2026 年，几乎所有主流大模型的后训练流程都包含了蒸馏。但每家的用法各有特色，值得逐个拆解。

### DeepSeek-R1：蒸馏胜过 RL

DeepSeek-R1 做了一个经典实验，直接回答了"蒸馏和 RL 谁更强"的问题。他们用 R1（大模型，经过大规模 RL 训练）生成了约 80 万条推理轨迹，然后通过 SFT 蒸馏到 Qwen2.5-1.5B / 7B / 14B / 32B。结果令人震惊：

- 蒸馏后的 Qwen2.5-32B 在多个基准上接近甚至超过 OpenAI o1-mini
- **同样的算力预算，直接在小模型上做 RL，效果不如蒸馏**

这说明了一个关键事实：**大模型 RL 训练发现的推理模式，对小模型来说是最宝贵的知识**。小模型自己通过 RL 很难发现这些模式（探索空间太大），但通过蒸馏可以直接学过来。

### Qwen3：两阶段蒸馏

Qwen3 对小模型（0.6B-30B）采用了两阶段蒸馏流程，比直接做 on-policy 蒸馏更稳健：

1. **Phase 1 - Off-policy 蒸馏**：先让 teacher 生成大量数据（包括 `/think` 和 `/no_think` 两种模式），student 在这些数据上做 SFT。这一步让 student 学到基本能力——会推理、会回答、会切换模式
2. **Phase 2 - On-policy 蒸馏**：student 自己生成轨迹，teacher 给密集的逐 token 反馈。这一步解决分布偏移——让 student 在自己的错误上学习

两阶段的逻辑很清晰：先用 off-policy 给 student 打好基础（避免一上来就 on-policy 导致训练不稳定），再用 on-policy 做精细校正。Qwen3 报告的关键数字是：**蒸馏只需要 RL 的 1/10 GPU 时间，但效果相当甚至更好**。

### GLM-5：用蒸馏防遗忘

GLM-5 的后训练是多个阶段串联的：先做 Reasoning RL，再做 Agentic RL，最后做 General RL。每个阶段都在优化不同能力，但问题是——学了新的容易忘旧的（灾难性遗忘）。

GLM-5 的解决方案很巧妙：**用前面阶段的 checkpoint 当 teacher，通过 on-policy 蒸馏做"跨阶段蒸馏"**。具体来说，最终阶段会同时用 SFT checkpoint、Reasoning RL checkpoint 和 General RL checkpoint 三个 teacher 的 logit 差作为训练信号。这样 student 在学习新能力的同时，被旧的 teacher "拉住"，不会忘记前面学的东西。

### MiMo-V2-Flash：多 Teacher 同时蒸馏

MiMo-V2-Flash（小米，309B 总参数 / 15B 激活参数的 MoE 模型）做得更极致。他们先用 RL 分别训练了多个领域专用 teacher：一个数学专家、一个代码专家、一个逻辑专家。然后让 student **同时接收所有 teacher 的信号**——每个 token 的 reward 是对应领域 teacher 的 log-prob。

这种多 teacher 蒸馏（他们称为 MOPD）有两个关键好处。第一是**算力节省**：分别训练多个小 teacher 再蒸馏，比训练一个全能大模型再蒸馏节省约 50 倍算力。第二是**防止能力冲突**：如果用一个 teacher 教所有领域，训练时经常出现"学了数学就忘代码"的问题。多 teacher 让每个领域有自己的信号通道，互不干扰。

<details>
<summary>思考题：为什么蒸馏对小模型特别有效，但对大模型可能不如 RL？</summary>

这个问题触及了 RL 和蒸馏的本质区别。小模型的探索能力有限——参数少，表示能力弱，通过 RL 很难自己发现高质量的推理模式。蒸馏相当于直接把这些模式"喂"给它，省去了探索过程。

大模型则不同。它的参数足够多，探索空间足够大，RL 可以帮助它发现**teacher 都不知道的**新策略。这时候蒸馏反而可能限制了大模型的上限——你只能学到 teacher 的水平，很难超越。

这也是为什么 Qwen3 的策略是：大模型用 RL 训练，小模型用蒸馏训练。各取所长。

</details>

## 本章小结

回顾整个第 8 章，我们已经覆盖了 LLM 后训练的核心路线：

| 路线               | 核心思想                 | 章节     |
| ------------------ | ------------------------ | -------- |
| DPO 及家族         | 绕过 RM 的离线偏好对齐   | 第 8 章  |
| GRPO / DAPO / RLVR | 省掉 Critic + 可验证奖励 | 第 8 章  |
| RLHF 工程流水线    | SFT → RM → RL 全流程     | 第 7 章  |
| **知识蒸馏 / OPD** | **用教师模型当密集奖励** | **本节** |

这四条路线不是互斥的。工业界最常见的做法是**先用 RL 训练大模型，再用蒸馏把能力迁移到小模型**——DeepSeek-R1 和 Qwen3 都是这么做的。从另一个角度看，这四条路线的训练信号分别来自人类偏好、可验证奖励和教师模型——理解了这三类信号各自的适用场景，你就掌握了 LLM 后训练的全局图景。

从第 4 章的策略梯度定理，到第 5 章的 PPO 裁剪机制，到第 8 章的 DPO 数学推导，再到第 8 章的 GRPO、RLVR——你已经掌握了现代 RL 训练从理论到实践的核心知识。上一章的 RLHF 工程流水线和本章的知识蒸馏则把理论转化为了可落地的系统。接下来的篇章将带你进入更前沿的领域：Agentic RL、VLM 强化学习、连续控制与具身智能。
