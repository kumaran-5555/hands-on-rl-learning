# 11.1 动手：用 GRPO 训练 VLM 回答视觉问题

第 8 章我们跑过 GRPO 训练纯文本模型做数学推理——给模型一道数学题，让它生成多个推理路径，用规则奖励（答案对不对）来计算组内相对优势，然后更新策略。现在我们要做一件更酷的事：给模型一张图片和一个关于图片的问题，让它先"看"再"想"再"答"。

这个实验的核心区别在于输入：纯文本 GRPO 的输入是一串 token，而 VLM GRPO 的输入是**视觉 token（图像编码）+ 文本 token（问题）**。奖励函数和优化算法本身没有变化——GRPO 的核心代码完全一样，只是模型的输入多了一个图像维度。

## 11.1.1 数据集：几何图形计数

我们选择一个简单的视觉问答任务——几何图形计数。这个任务的好处是有客观标准答案，可以用规则奖励来评估，不需要训练额外的 RM。

每张图片包含若干个基本几何图形（三角形、圆形、正方形），问题形如"图中有几个圆形？"。模型的理想回答流程是：先描述看到了什么（"我看到图片中有 3 个三角形、2 个圆形和 1 个正方形"），然后推理出答案（"所以圆形的数量是 2"）。

```python
# ==========================================
# 数据集：几何图形计数
# ==========================================
from datasets import Dataset
import random

def generate_shape_image(num_triangles, num_circles, num_squares, seed=None):
    """生成包含指定数量几何图形的图片"""
    from PIL import Image, ImageDraw

    if seed is not None:
        random.seed(seed)

    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)

    # 随机放置三角形
    for _ in range(num_triangles):
        x, y = random.randint(20, 236), random.randint(20, 236)
        size = random.randint(15, 35)
        draw.polygon([(x, y - size), (x - size, y + size), (x + size, y + size)],
                     fill='red', outline='darkred')

    # 随机放置圆形
    for _ in range(num_circles):
        x, y = random.randint(20, 236), random.randint(20, 236)
        r = random.randint(10, 25)
        draw.ellipse([(x - r, y - r), (x + r, y + r)],
                     fill='blue', outline='darkblue')

    # 随机放置正方形
    for _ in range(num_squares):
        x, y = random.randint(20, 236), random.randint(20, 236)
        s = random.randint(12, 28)
        draw.rectangle([(x - s, y - s), (x + s, y + s)],
                       fill='green', outline='darkgreen')

    return img


def generate_dataset(num_samples=500):
    """生成几何图形计数数据集"""
    data = []
    for i in range(num_samples):
        # 随机生成 1-5 个各种图形
        n_tri = random.randint(1, 5)
        n_cir = random.randint(1, 5)
        n_sqr = random.randint(1, 5)

        img = generate_shape_image(n_tri, n_cir, n_sqr, seed=i)

        # 随机选择一个问题
        questions = [
            f"图中有几个三角形？",
            f"图中有几个圆形？",
            f"图中有几个正方形？",
        ]
        answers = [str(n_tri), str(n_cir), str(n_sqr)]
        q_idx = random.randint(0, 2)

        data.append({
            'image': img,
            'question': questions[q_idx],
            'answer': answers[q_idx],
            'ground_truth': {
                'triangles': n_tri,
                'circles': n_cir,
                'squares': n_sqr,
            }
        })

    return Dataset.from_list(data)

# 生成训练集和验证集
train_dataset = generate_dataset(500)
val_dataset = generate_dataset(100)
```

## 11.1.2 奖励设计：三维评估

这个任务的奖励函数包含三个维度，每个维度都有明确的评分标准：

| 奖励维度              | 分值 | 评估标准                         | 类型     |
| --------------------- | ---- | -------------------------------- | -------- |
| 正确性（Correctness） | +1.0 | 最终答案与 ground truth 一致     | 规则奖励 |
| 推理质量（Reasoning） | +0.5 | 回答中包含对图片内容的描述       | 规则奖励 |
| 格式规范（Format）    | +0.2 | 回答遵循"描述 → 推理 → 答案"格式 | 规则奖励 |

这个奖励设计背后的思路是：正确答案最重要（+1.0），但我们不只想要"猜对答案"的模型——我们想要"看图 → 描述 → 推理 → 答案"的完整链路。所以推理质量（+0.5）和格式规范（+0.2）作为辅助奖励，引导模型形成正确的推理习惯。

```python
# ==========================================
# 奖励函数：三维评估
# ==========================================
import re

def compute_reward(response, ground_truth, target_shape):
    """
    计算三维奖励分数
    - response: 模型生成的回答
    - ground_truth: {'triangles': n, 'circles': n, 'squares': n}
    - target_shape: 本次问题的目标图形 ('triangles'/'circles'/'squares')
    """
    reward = 0.0

    # 1. 正确性奖励：提取最终答案，检查是否正确
    correct_answer = str(ground_truth[target_shape])
    # 尝试从回答末尾提取数字
    numbers = re.findall(r'\d+', response)
    if numbers and numbers[-1] == correct_answer:
        reward += 1.0

    # 2. 推理质量奖励：检查是否描述了图片内容
    shape_keywords = {
        'triangles': ['三角形', '红色', '三角'],
        'circles': ['圆形', '蓝色', '圆'],
        'squares': ['正方形', '绿色', '方块'],
    }
    has_description = any(kw in response for kw in shape_keywords[target_shape])
    if has_description:
        reward += 0.5

    # 3. 格式规范奖励：检查是否包含推理关键词
    reasoning_keywords = ['所以', '因此', '总共', '数量是', '答案是']
    has_reasoning = any(kw in response for kw in reasoning_keywords)
    if has_reasoning:
        reward += 0.2

    return reward
```

## 11.1.3 训练前后对比

训练前，模型的典型回答是"瞎猜"——因为它没有学会"先看图再推理"的策略。训练后，模型学会了先描述图片内容，然后基于描述推导答案。让我们看看 GRPO 训练过程是怎么实现这个转变的。

```python
# ==========================================
# VLM GRPO 训练循环
# ==========================================
def vlm_grpo_train(model, tokenizer, dataset, num_epochs=3, group_size=4, lr=1e-6):
    """
    用 GRPO 训练 VLM
    - group_size: 每个 prompt 生成多少个回答（组内比较）
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    normalizer = RewardNormalizer()

    for epoch in range(num_epochs):
        for batch in DataLoader(dataset, batch_size=8):
            all_log_probs = []
            all_rewards = []

            for prompt_img, prompt_text, ground_truth, target_shape in batch:
                # 对每个 prompt 生成 group_size 个回答
                group_responses = []
                group_log_probs = []
                group_rewards = []

                for _ in range(group_size):
                    # VLM 前向传播：输入图像 + 文本
                    response, log_prob = model.generate_with_log_prob(
                        image=prompt_img,
                        text=prompt_text,
                        max_new_tokens=128,
                        temperature=0.8
                    )

                    # 计算奖励
                    reward = compute_reward(response, ground_truth, target_shape)

                    group_responses.append(response)
                    group_log_probs.append(log_prob)
                    group_rewards.append(reward)

                all_log_probs.append(group_log_probs)
                all_rewards.append(group_rewards)

            # GRPO 核心：计算组内相对优势
            # 回顾第 8 章：Advantage = (R_i - mean) / std
            rewards_tensor = torch.tensor(all_rewards)
            mean_r = rewards_tensor.mean(dim=-1, keepdim=True)
            std_r = rewards_tensor.std(dim=-1, keepdim=True) + 1e-8
            advantages = (rewards_tensor - mean_r) / std_r

            # 策略梯度损失
            log_probs_tensor = torch.stack([torch.stack(lp) for lp in all_log_probs])
            loss = -(log_probs_tensor * advantages.detach()).mean()

            # 加入 KL 惩罚（回顾第 10 章）
            kl_penalty = compute_kl_penalty(model, ref_model, batch)
            loss = loss + 0.05 * kl_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

训练前后的对比非常直观。训练前，模型对"图中有几个圆形？"的回答可能是：

> **训练前**："3。"（纯粹瞎猜，没有看图的过程）

训练后，模型的回答变成：

> **训练后**："我看到图片中有 2 个红色三角形、**3 个蓝色圆形**和 1 个绿色正方形。问题是关于圆形的数量。所以圆形的数量是 3。"

模型学会了先描述视觉内容，再基于描述推导答案。这正是我们通过推理质量奖励（+0.5）和格式规范奖励（+0.2）引导的行为。

## 11.1.4 训练指标分析

训练 VLM 时，除了第 10 章提到的标准指标（奖励、KL 散度、响应长度）之外，还有几个多模态特有的指标值得关注：

**注意力热力图变化。** VLM 的注意力机制决定了模型在"看"图片的哪些区域。训练前，注意力可能分散在整个图片上；训练后，注意力应该集中在与问题相关的图形上。你可以通过可视化注意力热力图来验证这一点——如果问"有几个圆形"，注意力应该集中在蓝色圆形区域。

**推理长度与准确率的关系。** 统计回答中推理部分的长度和最终答案准确率的关系。理想情况是一个倒 U 形曲线——适度的推理长度效果最好。太短意味着模型没有仔细看图（猜答案），太长可能意味着模型在"过度思考"甚至产生视觉幻觉。

**跨泛化测试。** 在训练集之外的新图形组合上测试模型的表现。如果模型真的学会了"看图计数"的能力，它应该在从未见过的图形组合上也能正确回答——比如训练时最多 5 个图形，测试时给 7 个。

<details>
<summary>思考题：为什么 VLM GRPO 的学习率（1e-6）比纯文本 GRPO（通常 5e-7 到 1e-5）的范围更窄？</summary>

VLM 包含两个组件——视觉编码器（ViT）和文本解码器（Transformer）。如果学习率太大，RL 的梯度可能会破坏视觉编码器已经学到的特征（图像理解能力），导致模型"失明"——虽然还在输出文字，但已经"看不懂"图片了。如果学习率太小，文本解码器的策略更新太慢，训练效率极低。

实践中，一个常见的做法是对视觉编码器和文本解码器使用不同的学习率——视觉编码器用更小的学习率（比如文本解码器的 1/10），甚至完全冻结视觉编码器。这样可以在保持视觉理解能力的同时，让文本生成部分通过 RL 充分优化。下一节会详细讨论这个策略选择。

</details>

这个实验虽然简单，但展示了 VLM RL 的核心流程：和纯文本 RL 的算法完全一样（GRPO），但输入多了图像，奖励评估需要考虑视觉理解的维度。接下来我们要深入讨论的是，当输入从纯文本变成"图像 + 文本"时，会出现哪些纯文本场景中根本不存在的新挑战——[VLM RL 的特殊挑战](./vlm-challenges)。
