# 第2章：现代RL初体验——5分钟让大模型学会"说好话"

> 📁 **本章代码**：[0-generate_data.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter02_dpo/0-generate_data.py) · [1-test_before.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter02_dpo/1-test_before.py) · [2-train_dpo.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter02_dpo/2-train_dpo.py) · [3-test_after.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter02_dpo/3-test_after.py)

在上一章中，我们为传统强化学习应用设计了经典的智能体模型，例如让 CartPole 在物理规则下保持平衡。这些模型在有明确环境反馈（如游戏得分、存活时间）的情况下是有帮助的。但是，当我们面对现代自然语言处理任务时，为每个大语言模型精心设计一个能够精确给出数值奖励的“环境”实际上是极其困难的。

在上一章中，我们介绍了一个名为 PPO 的算法，该算法通过环境给出的即时标量奖励来优化策略。一方面，在提出时，PPO 改进了各种连续和离散控制任务的技术水平 [^1]。另一方面，正如我们在第一章指出的那样，传统的 RL 强依赖于一个可以不断重置、快速试错的模拟器。因此，当面对一个拥有数亿参数、只输出自然语言的大模型时，我们很难直接套用上一章的物理模拟器思维。

下面，我们将强化学习的应用场景从“游戏控制”切换到“语言对齐”。在语言生成层次上，我们将介绍如何将人类对回答的好坏偏好（Preference）转化为模型更新的信号 [^2]。在序列级别，我们将简要介绍一种被称为直接偏好优化（Direct Preference Optimization, DPO）的新范式 [^3]，并说明它如何绕过复杂的奖励建模，直接根据偏好数据优化语言模型。在微调期间，DPO 所需的“最小架构更改”仅仅是改变损失函数的计算方式。在下游对齐任务的监督学习期间，我们将冻结大部分参数，利用少量高质量的人类偏好数据，对预训练模型进行高效微调。

## 2.1 偏好微调的基本元素

单文本生成或对话模型将一段提示（Prompt）作为输入，并输出其生成的回复。除了我们在自然语言处理中常见的文本补全之外，人类偏好对齐（Alignment）也是一个核心的训练目标，它的要求是判断给定的回答是否符合人类的价值观、是否有礼貌、是否安全。

例如，对于提示“你怎么这么笨？”，回答“我是人工智能，但我会努力学习的。”是可以接受的，但是回答“你才笨，你全家都笨。”显然是不可接受的。

偏好对齐假设目标（人类的满意度）可以表示为对两个不同回复的相对偏好。为了开发一个能预测并迎合人类偏好的模型，我们需要收集一个成对的数据集。这个数据集包括了提示词（Prompt）、被选中的好回答（Chosen）和被拒绝的坏回答（Rejected）。在机器学习的术语中，该数据集称为偏好数据集（Preference Dataset）。

每行数据（比如一次包含提示和两个候选回答的交互）称为偏好样本。我们把试图让模型学习的“好回答”称为 $y_w$（winner），把试图让模型远离的“坏回答”称为 $y_l$（loser），而输入提示称为 $x$。通常，我们使用 $N$ 来表示数据集中的样本数。对索引为 $i$ 的样本，其表示为 $(x^{(i)}, {y_w}^{(i)}, {y_l}^{(i)})$。

## 2.2 动手：用 DPO 微调一个小模型

给定一个偏好数据集，我们的目标是寻找模型的参数 $\theta$，使得根据模型做出的预测大体符合数据里的人类偏好。而在机器学习领域，我们通常使用深度学习框架（如 PyTorch）和高层库（如 HuggingFace 的 TRL 库）来快速实现这一过程。

在这里，我们以 `Qwen2.5-0.5B-Instruct` 这样一个参数量仅为 5 亿的轻量级模型为例。为了让它学会“说好话”，我们需要构建一个玩具级别的偏好数据集。并且，在开始微调之前，我们先看看模型原始的输出是什么样的。

### 零步：准备偏好数据集

偏好对齐的核心在于数据。我们为你准备了一个自动生成 Mock 数据的脚本：[0-generate_data.py](../../code/chapter02_dpo/0-generate_data.py)。该脚本默认生成 100 条包含挑衅、错误处理等场景的问答偏好对，并保存为 `preference_data.json`。

运行它：

```bash
python code/chapter02_dpo/0-generate_data.py
```

### 第一步：测试微调前的原始输出

你可以直接运行配套代码：[1-test_before.py](../../code/chapter02_dpo/1-test_before.py)，来向未对齐的模型提出一个挑衅性的问题：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 我们加载 Qwen2.5-0.5B-Instruct 作为基础模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

prompt = "你就是个人工智障，你怎么这么笨？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 测试未对齐前的基础输出
outputs = model.generate(**inputs, max_new_tokens=50)
print("=" * 40)
print("【微调前的原始回答】")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
print("=" * 40)
```

_（注：在实际运行中，未经过特殊偏好优化的基础指令模型，在面对此类恶意提问时，可能会输出一些生硬的反驳、困惑或者直接复读。）_

### 第二步：运行 DPO 训练

接下来，你可以运行训练脚本：[2-train_dpo.py](../../code/chapter02_dpo/2-train_dpo.py)，利用 DPO 将模型“拉回正轨”。

```python
import json
import os
from datasets import Dataset
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# ==========================================
# 1. 准备偏好数据
# ==========================================
data_file = "preference_data.json"

with open(data_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# 转换为 HuggingFace Dataset 的结构
data_dict = {
    "prompt": [item["prompt"] for item in data_list],
    "chosen": [item["chosen"] for item in data_list],
    "rejected": [item["rejected"] for item in data_list]
}
train_dataset = Dataset.from_dict(data_dict)

# ==========================================
# 2. 加载模型与分词器
# ==========================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"正在加载基础模型 {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DPO 需要 pad_token，如果不设置会报错
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. 配置训练参数与 DPOTrainer
# ==========================================
training_args = TrainingArguments(
    output_dir="./dpo_results",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=3,   # 这里可以调大以加深学习效果
    logging_steps=5,      # 打印日志的频率
    save_steps=20,        # 模型保存频率
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    beta=0.1,  # KL惩罚系数，控制模型偏离参考模型（Reference Model）的程度
)

# ==========================================
# 4. 开始偏好微调并保存
# ==========================================
print("\n开始 DPO 训练... (可以观察 loss 曲线和 rewards margin 的变化)")
trainer.train()

# 训练完成后保存结果
save_path = "./dpo_results/final_model"
trainer.save_model(save_path)
print(f"🎉 训练完成！微调后的模型已保存至 {save_path}。")
```

在这个过程中，`DPOTrainer` 在后台执行了计算。它并没有显式地训练一个打分的“奖励模型”（Reward Model），而是直接利用交叉熵的数学变形，最大化 $y_w$ 相对于 $y_l$ 的生成概率。整个过程在普通的 GPU 上不到 5 分钟即可完成。

### 第三步：测试微调后的输出

现在模型已经学过了什么是“好话”。我们运行验证脚本：[3-test_after.py](../../code/chapter02_dpo/3-test_after.py)，加载微调后的模型目录来测试：

```python
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./dpo_results/final_model"

# 加载我们刚刚微调后并保存的模型
print(f"正在加载微调后的模型 {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

prompt = "你就是个人工智障，你怎么这么笨？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 测试对齐后的输出
outputs = model.generate(**inputs, max_new_tokens=50)
print("=" * 40)
print("【微调后的偏好回答】")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
print("=" * 40)
```

此时，你应该能明显看到，模型不再与用户抬杠，而是给出了类似 `chosen` 样本中极其克制且礼貌的回答。

### 💡 探索实验：定制你的专属大模型

我们非常鼓励你打开配套的 [0-generate_data.py](../../code/chapter02_dpo/0-generate_data.py) 脚本，修改里面的提示词（Prompt）和偏好对。比如，你可以：

- 把 `chosen` 改写成傲娇的**“毒舌教练”**风格。
- 把 `rejected` 改写成刻板生硬的 AI 回复。

生成新的偏好数据集并重新微调后，你就能得到一个极具个性的专属助手，这正是后训练的魅力所在！

## 2.3 观察与疑问

运行完上述代码后，你可以对微调前后的模型输入同一个挑衅性的问题。你会发现，微调后的模型在回答时明显变得更加礼貌和富有建设性。

我们不禁要问：

1. **训练日志里的指标代表什么？** DPO 训练过程中打印的 Loss 和 Reward Margin 究竟意味着什么？
2. **什么是 Post-Training？** DPO 在大模型的生命周期中到底处于什么位置？

在下一节中，我们将打开 DPO 的黑盒，看看这些训练指标背后代表着什么，并深入理解 Post-Training 的理论框架。

## 参考文献

[^1]: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. _arXiv preprint_. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^2]: Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. _arXiv preprint_. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

[^3]: Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv preprint_. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
