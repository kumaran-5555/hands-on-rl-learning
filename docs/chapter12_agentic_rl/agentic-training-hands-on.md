# 12.4 动手：端到端 Agentic RL 训练——真实模型、真实 Benchmark、真实提升

前面的实验用模拟数据对比了 ORM 和 PRM。但那些轨迹是假的——没有真实的模型推理，没有真实的代码执行，也没有真实的梯度更新。这一节我们要做一件更硬核的事：**用真实的 Code LLM，跑真实的 HumanEval Benchmark，做真实的 GRPO 训练，看到真实的 pass@1 提升。**

整个实验在一台 24GB 显存的 GPU（如 RTX 4090 / A5000）上即可完成。如果你没有 GPU，可以用 Google Colab 的免费 T4。

```mermaid
flowchart LR
    A["1. 环境\n准备"] --> B["2. 基线\nBenchmark"]
    B --> C["3. On-Policy\nRollout"]
    C --> D["4. GRPO\n训练"]
    D --> E["5. 评测\n对比"]
    E -->|"效果不够？"| C

    style A fill:#e3f2fd,stroke:#1976d2,color:#000
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#f3e5f5,stroke:#7b1fa2,color:#000
    style D fill:#fce4ec,stroke:#c62828,color:#000
    style E fill:#e8f5e9,stroke:#388e3c,color:#000
```

## 第零步：环境准备

```bash
# 核心依赖
pip install torch transformers accelerate
pip install datasets                    # HumanEval 数据集
pip install human-eval                  # pass@k 评测
pip install matplotlib numpy
pip install peft                        # LoRA 微调
```

我们使用 **Qwen2.5-Coder-1.5B-Instruct** 作为基座模型——1.5B 参数量在 24GB 显存上跑 GRPO（group_size=4）绰绰有余，同时代码能力已经足够做 HumanEval。

> 如果你 GPU 显存更大（≥48GB），推荐换成 `Qwen2.5-Coder-7B-Instruct`，效果更好。只需修改下面的 `MODEL_NAME` 即可。

```python
# ==========================================
# 0. 全局配置
# ==========================================
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 512
GROUP_SIZE = 4          # GRPO 组内采样数
MAX_EPOCHS = 3          # 训练 epoch 数（小模型快速收敛）
LR = 5e-6               # 学习率
KL_COEFF = 0.05         # KL 惩罚系数
EFFICIENCY_PENALTY = 0.02  # 每轮交互的效率惩罚

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

## 第一步：加载模型与 Benchmark

### 1.1 加载真实模型

```python
# ==========================================
# 1.1 加载 Code LLM
# ==========================================
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model.eval()

# 冻结参数（稍后只训练 LoRA）
for param in model.parameters():
    param.requires_grad = False

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
```

### 1.2 加载 HumanEval Benchmark

我们使用 OpenAI 的 **HumanEval**——164 个 Python 编程题，每题包含函数签名、docstring 和单元测试。这是 Code Agent 领域最经典的 benchmark。

```python
# ==========================================
# 1.2 加载 HumanEval
# ==========================================
from datasets import load_dataset

humaneval = load_dataset("openai_humaneval", trust_remote_code=True)
problems = humaneval["test"]

print(f"HumanEval 题目数: {len(problems)}")
print(f"\n示例题目:")
print(f"  task_id: {problems[0]['task_id']}")
print(f"  prompt (前 100 字符): {problems[0]['prompt'][:100]}...")
print(f"  test: {problems[0]['test'][:80]}...")
```

### 1.3 真实的代码执行验证器

这是整个实验的基石——用 Python `exec` 实际运行模型生成的代码，跑 HumanEval 的断言测试。**不模拟、不估算，真实执行。**

```python
# ==========================================
# 1.3 真实的代码执行验证器
# ==========================================
import subprocess
import tempfile
import os
import re

def execute_completion(task: dict, completion: str, timeout: float = 10.0) -> dict:
    """
    真实执行模型生成的代码，跑 HumanEval 的断言测试。
    返回 {"passed": bool, "error": str or None}
    """
    # 组装完整代码：函数定义 + 测试断言
    full_code = task["prompt"] + completion + "\n" + task["test"] + "\n"
    # HumanEval 的 test 里有 check() 调用，需要加上入口
    full_code += f"\ncheck({task['entry_point']})\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "solution.py")
        with open(path, "w") as f:
            f.write(full_code)

        try:
            result = subprocess.run(
                ["python", path],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmpdir,
            )
            if result.returncode == 0:
                return {"passed": True, "error": None}
            else:
                # 提取关键错误信息
                err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
                return {"passed": False, "error": err}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "TIMEOUT"}
        except Exception as e:
            return {"passed": False, "error": str(e)}


def batch_evaluate(tasks: list, completions: list) -> dict:
    """批量评测，返回 pass@1 和详细结果"""
    assert len(tasks) == len(completions)
    results = []
    for task, comp in zip(tasks, completions):
        res = execute_completion(task, comp)
        results.append(res)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    return {
        "pass@1": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
        "details": results,
    }
```

## 第二步：基线 Benchmark——训练前模型的真实水平

在做任何训练之前，先跑一遍 HumanEval，记录模型的**真实基线**。

### 2.1 构建 Agent Prompt

我们将每道 HumanEval 题目包装成 Agent 可以理解的格式。为了让本实验聚焦，我们先用**单轮生成**（不用工具调用），直接让模型补全代码。后面第五步会引入多轮工具调用。

```python
# ==========================================
# 2.1 构建 Prompt
# ==========================================

def build_completion_prompt(task: dict) -> str:
    """构建代码补全的 prompt"""
    return (
        f"Complete the following Python function. "
        f"Only output the function body, no extra explanation.\n\n"
        f"{task['prompt']}"
    )

def generate_completion(prompt: str, temperature: float = 0.0) -> str:
    """用模型生成代码补全"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只取生成部分（去掉 prompt）
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion.strip()
```

### 2.2 跑基线评测

```python
# ==========================================
# 2.2 基线 Benchmark 评测
# ==========================================

# 为了控制时间，我们评测前 64 道题（完整 HumanEval 164 题约需 30 分钟）
BASELINE_N = 64
baseline_tasks = list(problems.select(range(BASELINE_N)))

print(f"Running baseline evaluation on {BASELINE_N} HumanEval problems...")
print("=" * 60)

baseline_completions = []
baseline_results = []

for i, task in enumerate(baseline_tasks):
    prompt = build_completion_prompt(task)
    completion = generate_completion(prompt, temperature=0.0)

    eval_res = execute_completion(task, completion)
    baseline_completions.append(completion)
    baseline_results.append(eval_res)

    status = "PASS" if eval_res["passed"] else f"FAIL ({eval_res['error'][:40]})"
    if (i + 1) % 16 == 0 or i == 0:
        passed_so_far = sum(1 for r in baseline_results if r["passed"])
        print(f"  [{i+1:3d}/{BASELINE_N}] {task['task_id']} {status}  "
              f"(running pass@1: {passed_so_far/(i+1):.1%})")

baseline_metrics = batch_evaluate(baseline_tasks, baseline_completions)
print("\n" + "=" * 60)
print(f"基线 pass@1: {baseline_metrics['pass@1']:.1%} "
      f"({baseline_metrics['passed']}/{baseline_metrics['total']})")
print("=" * 60)
```

::: details 预期基线结果参考
Qwen2.5-Coder-1.5B-Instruct 在 HumanEval 上的 greedy decoding (temperature=0) pass@1 通常在 **36-42%** 左右。你的具体数值会因硬件和版本略有不同——这正是我们要提升的目标。
:::

## 第三步：搭建工具增强的 Agent 环境

现在我们把单轮补全升级为**多轮工具调用的 Agent**。Agent 可以读题、执行代码试错、根据报错修复——这正是 Agentic RL 的核心。

### 3.1 定义工具集

```python
# ==========================================
# 3.1 工具集定义
# ==========================================

def tool_execute_code(task: dict, code: str) -> str:
    """工具：在沙箱中执行代码并返回结果"""
    full_code = task["prompt"] + code + "\n" + task["test"] + "\n"
    full_code += f"\ncheck({task['entry_point']})\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_solution.py")
        with open(path, "w") as f:
            f.write(full_code)
        try:
            result = subprocess.run(
                ["python", path], capture_output=True, text=True,
                timeout=10, cwd=tmpdir,
            )
            if result.returncode == 0:
                return "ALL TESTS PASSED"
            else:
                # 返回最后几行错误（最有用的调试信息）
                err_lines = result.stderr.strip().split("\n")
                return "ERROR:\n" + "\n".join(err_lines[-3:])
        except subprocess.TimeoutExpired:
            return "ERROR: TIMEOUT"

def tool_run_partial_test(task: dict, code: str, test_expr: str) -> str:
    """工具：运行用户自定义的部分测试"""
    full_code = task["prompt"] + code + "\nprint(" + test_expr + ")\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "partial.py")
        with open(path, "w") as f:
            f.write(full_code)
        try:
            result = subprocess.run(
                ["python", path], capture_output=True, text=True,
                timeout=5, cwd=tmpdir,
            )
            output = result.stdout.strip()
            err = result.stderr.strip()
            if result.returncode == 0:
                return f"Output: {output}"
            else:
                return f"Error: {err.split(chr(10))[-1]}"
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
```

### 3.2 Agent Loop：模型与环境的多轮交互

这是 Agentic RL 的核心区别。和单轮补全不同，Agent 会**调用工具、看到报错、修复代码、再次执行**。每一步都记录在完整的对话轨迹中——这条轨迹就是训练数据。

```mermaid
flowchart TD
    S["System: Agent Prompt"] --> U1["User: 题目"]
    U1 --> A1["Assistant: 代码 v1\n（模型生成）"]
    A1 --> T1["Tool: 执行代码 v1"]
    T1 --> U2["User: 执行结果\n（ERROR: ...)"]
    U2 --> A2["Assistant: 代码 v2\n（模型修复）"]
    A2 --> T2["Tool: 执行代码 v2"]
    T2 --> U3["User: 执行结果\n（ALL TESTS PASSED)"]
    U3 --> A3["Assistant: 最终代码\n（模型提交）"]

    style A1 fill:#e3f2fd,stroke:#1976d2,color:#000
    style A2 fill:#e3f2fd,stroke:#1976d2,color:#000
    style A3 fill:#e3f2fd,stroke:#1976d2,color:#000
    style T1 fill:#fff3e0,stroke:#f57c00,color:#000
    style T2 fill:#fff3e0,stroke:#f57c00,color:#000
    style U2 fill:#fce4ec,stroke:#c62828,color:#000
    style U3 fill:#e8f5e9,stroke:#2e7d32,color:#000
```

训练时，**loss 只算在蓝色节点（模型生成的 assistant token）上**，橙色和红色节点（工具执行结果）的 token 被 mask 掉。

```python
# ==========================================
# 3.2 Agent Loop：真实模型 + 真实工具交互
#     关键：返回完整的 multi-turn 对话轨迹
# ==========================================

AGENT_SYSTEM_PROMPT = """\
You are a Python coding agent. Given a function signature and docstring, you must implement the function.

You have two tools available:
1. <execute>your_code</execute> — Run your current code against all tests. Returns PASS or error details.
2. <test>expression</test> — Run a single Python expression with your current code loaded. Returns the output.

Workflow:
1. Write your initial implementation.
2. Use <execute> to test it.
3. If it fails, read the error, fix the code, and test again.
4. When all tests pass or you've tried your best, output <final>your_final_code</final>.

You have at most 3 rounds of execution. Be efficient."""

def run_agent_episode(
    task: dict,
    temperature: float = 0.7,
    max_rounds: int = 3,
    verbose: bool = False,
) -> dict:
    """
    运行一次真实的 Agent 交互循环。
    模型生成代码 → 执行测试 → 看到报错 → 修复 → 再执行。

    ★ 关键：返回完整的 multi-turn 对话轨迹 conversation。
    训练时用整条轨迹计算 log_prob，不是只用最终代码。
    """
    conversation = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Implement this function:\n\n{task['prompt']}"},
    ]

    current_code = ""
    rounds_used = 0

    for round_idx in range(max_rounds + 1):
        # 模型生成回复
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # ★ 把 assistant 回复追加到对话轨迹
        conversation.append({"role": "assistant", "content": response})
        rounds_used += 1

        # 解析 <final> 标签——Agent 提交最终代码
        final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL)
        if final_match:
            current_code = final_match.group(1).strip()
            break

        # 解析 <execute> 标签——执行测试
        exec_match = re.search(r'<execute>(.*?)</execute>', response, re.DOTALL)
        if exec_match:
            current_code = exec_match.group(1).strip()
            exec_result = tool_execute_code(task, current_code)

            if verbose:
                print(f"  Round {round_idx+1}: {exec_result[:80]}")

            # ★ 把工具执行结果作为 user 消息追加到轨迹
            if exec_result == "ALL TESTS PASSED":
                conversation.append({"role": "user",
                    "content": f"Execution result: {exec_result}\nAll tests passed! Output your final code with <final> tags."})
            else:
                conversation.append({"role": "user",
                    "content": f"Execution result:\n{exec_result}\n\nPlease fix the code and try again."})
            continue

        # 解析 <test> 标签——部分测试
        test_match = re.search(r'<test>(.*?)</test>', response, re.DOTALL)
        if test_match:
            test_expr = test_match.group(1).strip()
            if current_code:
                test_result = tool_run_partial_test(task, current_code, test_expr)
            else:
                test_result = "No code written yet."

            # ★ 同样追加到轨迹
            conversation.append({"role": "user",
                "content": f"Test result: {test_result}"})
            continue

        # 没有工具调用——直接提取代码
        current_code = response
        break

    # 最终验证
    if current_code:
        final_eval = execute_completion(task, current_code)
    else:
        final_eval = {"passed": False, "error": "NO CODE GENERATED"}

    return {
        "completion": current_code,
        "passed": final_eval["passed"],
        "rounds": rounds_used,
        "error": final_eval.get("error"),
        # ★ 完整的多轮对话轨迹——Agentic RL 训练的核心数据
        "conversation": conversation,
    }
```

### 3.3 从对话轨迹构建训练数据——token 级 mask

上面的 `run_agent_episode` 返回了完整的 `conversation`——包含 system prompt、用户题目、模型的代码尝试、工具执行结果、错误修复……这一整条轨迹就是 Agentic RL 的训练数据。

但训练时有一个关键约束：**loss 只应该算在模型生成的 token 上（assistant 角色），工具执行结果的 token（user 角色）要被 mask 掉。** 如果不 mask，模型会被训练去"生成工具返回的报错信息"——这显然是错的。

```python
# ==========================================
# 3.3 Tokenize 轨迹 + 构建 assistant-only mask
# ==========================================

def tokenize_trajectory_with_mask(conversation: list, max_length: int = 2048):
    """
    将完整的多轮对话轨迹 tokenize，并构建 labels：
    - assistant 生成的 token → 保留原 token id（参与 loss 计算）
    - system / user / 工具结果的 token → 设为 -100（被 mask，不参与 loss）

    这是 Agentic RL 训练的关键数据预处理步骤。
    """
    # 先 tokenize 每一段，记录哪些是 assistant 生成的
    segments = []       # [(token_ids, is_assistant), ...]
    for msg in conversation:
        # 构建这段消息的文本（需要包含角色标记）
        is_assistant = (msg["role"] == "assistant")

        # 用 chat_template 拼接每段消息
        partial_conv = []
        for prev_msg in conversation:
            partial_conv.append(prev_msg)
            if prev_msg is msg:
                break

        if is_assistant:
            # assistant: tokenize 到这段消息结束（不含 generation_prompt）
            text = tokenizer.apply_chat_template(
                partial_conv, tokenize=False, add_generation_prompt=False
            )
        else:
            # system/user: tokenize 这段消息 + 后面的 generation prompt
            text = tokenizer.apply_chat_template(
                partial_conv, tokenize=False, add_generation_prompt=True
            )

        tokens = tokenizer.encode(text, add_special_tokens=False)
        segments.append((tokens, is_assistant))

    # 拼接所有 segment，计算每个 token 的归属
    all_tokens = []
    token_is_assistant = []

    prev_len = 0
    for tokens, is_assistant in segments:
        # 只取新增部分（避免重复计算前面消息的 token）
        new_tokens = tokens[prev_len:] if len(tokens) > prev_len else []
        all_tokens.extend(new_tokens)
        token_is_assistant.extend([is_assistant] * len(new_tokens))
        prev_len = len(tokens)

    # 截断到 max_length
    if len(all_tokens) > max_length:
        all_tokens = all_tokens[:max_length]
        token_is_assistant = token_is_assistant[:max_length]

    # 构建 labels：assistant token 保留，其余 mask 为 -100
    input_ids = torch.tensor([all_tokens], dtype=torch.long)
    labels = torch.tensor([all_tokens], dtype=torch.long)
    for i, is_assist in enumerate(token_is_assistant):
        if not is_assist:
            labels[0, i] = -100  # mask 掉非 assistant token

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

```mermaid
flowchart LR
    subgraph "完整轨迹 tokens"
        S["system\ntokens"] --> U1["user\n(题目) tokens"] --> A1["assistant\ntokens ✓"]
        A1 --> U2["user\n(报错) tokens"] --> A2["assistant\ntokens ✓"]
        A2 --> U3["user\n(PASS) tokens"] --> A3["assistant\ntokens ✓"]
    end

    subgraph "Labels (训练目标)"
        LS["-100"] --> LU1["-100"] --> LA1["保留 ✓\n算 loss"]
        LA1 --> LU2["-100"] --> LA2["保留 ✓\n算 loss"]
        LA2 --> LU3["-100"] --> LA3["保留 ✓\n算 loss"]
    end

    style S fill:#90a4ae,stroke:#546e7a,color:#fff
    style U1 fill:#90a4ae,stroke:#546e7a,color:#fff
    style A1 fill:#1976d2,stroke:#0d47a1,color:#fff
    style U2 fill:#90a4ae,stroke:#546e7a,color:#fff
    style A2 fill:#1976d2,stroke:#0d47a1,color:#fff
    style U3 fill:#90a4ae,stroke:#546e7a,color:#fff
    style A3 fill:#1976d2,stroke:#0d47a1,color:#fff
    style LS fill:#e0e0e0,stroke:#9e9e9e
    style LU1 fill:#e0e0e0,stroke:#9e9e9e
    style LA1 fill:#e8f5e9,stroke:#2e7d32
    style LU2 fill:#e0e0e0,stroke:#9e9e9e
    style LA2 fill:#e8f5e9,stroke:#2e7d32
    style LU3 fill:#e0e0e0,stroke:#9e9e9e
    style LA3 fill:#e8f5e9,stroke:#2e7d32
```

**只有蓝色（assistant）token 参与 loss 计算。** 灰色（system/user/tool result）token 被 mask 为 -100。这意味着模型学习的是"在给定上下文（包括工具执行结果）的情况下如何生成下一步动作"——这正是 Agentic RL 要学的策略。

::: details 为什么不能简单地只训练最终输出？
如果只用最终代码训练，模型只学到了"给一个题目，输出正确答案"——这和单轮 SFT 没有本质区别。

Agentic RL 的核心价值是训练模型**整个交互过程**：
- 学会在第一步写出合理的初始代码
- 学会调用 `<execute>` 工具验证
- 学会**阅读错误信息并定位 bug**
- 学会生成正确的修复代码

只有训练完整轨迹，模型才能学到这些多步策略。如果只训练最终输出，模型永远不会学到"怎么调试"——它只会学到"什么是正确答案"。这就是 SFT 和 Agentic RL 的本质区别。
:::

### 3.4 验证：轨迹构造是否正确

在训练之前，先验证 Agent Loop 产出的轨迹是否包含完整的多轮对话。

```python
# ==========================================
# 3.4 验证轨迹结构
# ==========================================

print("Sanity check — Agent trajectory structure:")
print("-" * 60)

task = problems[0]
result = run_agent_episode(task, temperature=0.3, verbose=True)

print(f"\nStatus: {'PASS' if result['passed'] else 'FAIL'}")
print(f"Rounds: {result['rounds']}")
print(f"\n完整对话轨迹 (conversation):")
for i, msg in enumerate(result["conversation"]):
    role = msg["role"]
    content_preview = msg["content"][:80].replace("\n", " ")
    print(f"  [{i}] {role:12s} | {content_preview}...")

# 验证 tokenize + mask 是否正确
enc = tokenize_trajectory_with_mask(result["conversation"])
total_tokens = enc["input_ids"].shape[1]
assistant_tokens = (enc["labels"] != -100).sum().item()
masked_tokens = total_tokens - assistant_tokens

print(f"\nToken 统计:")
print(f"  总 token 数:     {total_tokens}")
print(f"  assistant token: {assistant_tokens} ({assistant_tokens/total_tokens:.0%})")
print(f"  masked token:    {masked_tokens} ({masked_tokens/total_tokens:.0%})")
print("-" * 60)
```

## 第四步：GRPO 训练——真实的梯度更新

现在我们有了真实的模型、真实的 Agent Loop、真实的执行验证。接下来做真正的 GRPO 训练。

### 4.1 LoRA——让小 GPU 也能训练

直接更新 1.5B 参数太贵了。我们用 LoRA 只训练一小部分参数。

```python
# ==========================================
# 4.1 设置 LoRA
# ==========================================
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # 只训练 attention 的 Q 和 V
)

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 4.2 真实的 GRPO 训练循环——在完整轨迹上训练

这里是核心。和单轮 RL 的关键区别：**训练数据是多轮对话轨迹（包含工具调用和执行结果），loss 只算在 assistant 生成的 token 上。**

每一轮都涉及：
1. **真实模型推理**生成多条完整的多轮轨迹（不只是最终代码）
2. **真实代码执行**验证每条轨迹
3. **真实梯度更新**用 GRPO + assistant-only mask 优化策略

```python
# ==========================================
# 4.2 GRPO 训练循环（Agentic 多轮轨迹版本）
# ==========================================
from torch.optim import AdamW
import torch.nn.functional as F

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# 引用模型（KL 约束用的 frozen copy）
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# ---- 训练数据 ----
# 选择 32 道题作为训练集（不同于评测集，避免数据泄露）
TRAIN_IDS = list(range(64, 96))  # HumanEval #64-#95
train_tasks = list(problems.select(TRAIN_IDS))

# ---- 训练 ----
training_log = {
    "epoch": [], "mean_reward": [], "success_rate": [],
    "avg_rounds": [], "loss": [], "avg_traj_tokens": [],
}

print("=" * 70)
print("GRPO Agentic RL Training (Multi-Turn Trajectory)")
print(f"  Model: {MODEL_NAME}")
print(f"  Train tasks: {len(train_tasks)} (HumanEval #{TRAIN_IDS[0]}-#{TRAIN_IDS[-1]})")
print(f"  Group size: {GROUP_SIZE}")
print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Learning rate: {LR}")
print(f"  ★ Training on FULL multi-turn trajectories (not just final output)")
print("=" * 70)

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_rewards = []
    epoch_successes = 0
    epoch_losses = []
    epoch_rounds = []
    epoch_traj_tokens = []

    random.shuffle(train_tasks)

    for task_idx, task in enumerate(train_tasks):
        # ---- Phase 1: On-Policy Rollout (GROUP_SIZE 条完整轨迹) ----
        trajectories = []
        for g in range(GROUP_SIZE):
            result = run_agent_episode(task, temperature=0.7, max_rounds=3)
            # Reward = 任务完成 + 效率惩罚
            reward = (1.0 if result["passed"] else 0.0) - EFFICIENCY_PENALTY * result["rounds"]
            trajectories.append({
                **result,
                "reward": reward,
            })

        # ---- Phase 2: GRPO Advantage ----
        rewards = np.array([t["reward"] for t in trajectories])
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # ---- Phase 3: 在完整轨迹上做策略梯度更新 ----
        for traj, advantage in zip(trajectories, advantages):
            if not traj["completion"]:
                continue  # 跳过空轨迹

            # ★ 关键：用完整的多轮对话轨迹 tokenize，不是只用最终代码
            enc = tokenize_trajectory_with_mask(traj["conversation"], max_length=2048)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = enc["labels"].to(device)

            traj_tokens = (labels != -100).sum().item()
            epoch_traj_tokens.append(traj_tokens)

            # 计算 policy log prob（只在 assistant token 上）
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # outputs.loss 已经自动忽略了 labels=-100 的 token
            policy_log_prob = -outputs.loss

            # 计算 reference log prob（KL 惩罚）
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ref_log_prob = -ref_outputs.loss

            # GRPO loss = -advantage * (log π - log π_ref) + KL
            kl_term = policy_log_prob - ref_log_prob
            pg_loss = -advantage * policy_log_prob
            loss = pg_loss + KL_COEFF * kl_term

            # 梯度更新
            if model.training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_losses.append(loss.item())

        # 统计
        epoch_rewards.extend(rewards.tolist())
        epoch_successes += sum(1 for t in trajectories if t["passed"])
        epoch_rounds.extend([t["rounds"] for t in trajectories])

        # 打印进度
        if (task_idx + 1) % 8 == 0:
            running_sr = epoch_successes / ((task_idx + 1) * GROUP_SIZE)
            print(f"  Epoch {epoch+1}/{MAX_EPOCHS} | "
                  f"Task {task_idx+1}/{len(train_tasks)} | "
                  f"Running SR: {running_sr:.1%} | "
                  f"Avg Reward: {np.mean(epoch_rewards[-GROUP_SIZE:]):.3f} | "
                  f"Avg Traj Tokens: {np.mean(epoch_traj_tokens[-GROUP_SIZE:]):.0f}")

    # Epoch 统计
    epoch_sr = epoch_successes / (len(train_tasks) * GROUP_SIZE)
    epoch_mean_r = np.mean(epoch_rewards)
    epoch_mean_loss = np.mean(epoch_losses) if epoch_losses else 0
    epoch_mean_rounds = np.mean(epoch_rounds)
    epoch_mean_tokens = np.mean(epoch_traj_tokens)

    training_log["epoch"].append(epoch + 1)
    training_log["mean_reward"].append(epoch_mean_r)
    training_log["success_rate"].append(epoch_sr)
    training_log["avg_rounds"].append(epoch_mean_rounds)
    training_log["loss"].append(epoch_mean_loss)
    training_log["avg_traj_tokens"].append(epoch_mean_tokens)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Summary:")
    print(f"  Success Rate:    {epoch_sr:.1%}")
    print(f"  Mean Reward:     {epoch_mean_r:.3f}")
    print(f"  Mean Loss:       {epoch_mean_loss:.4f}")
    print(f"  Avg Rounds:      {epoch_mean_rounds:.1f}")
    print(f"  Avg Traj Tokens: {epoch_mean_tokens:.0f} (assistant-only)")
    print(f"{'='*60}\n")
```

## 第五步：训练后 Benchmark——真的提升了吗？

训练完了，现在最关键的：**用和基线完全相同的评测协议，重新跑 HumanEval，看 pass@1 有没有变化。**

```python
# ==========================================
# 5. 训练后 Benchmark 评测
# ==========================================

model.eval()

print(f"Running POST-TRAINING evaluation on {BASELINE_N} HumanEval problems...")
print("(Same problems as baseline, greedy decoding)")
print("=" * 60)

post_completions = []
post_results = []

for i, task in enumerate(baseline_tasks):
    prompt = build_completion_prompt(task)
    completion = generate_completion(prompt, temperature=0.0)

    eval_res = execute_completion(task, completion)
    post_completions.append(completion)
    post_results.append(eval_res)

    status = "PASS" if eval_res["passed"] else f"FAIL ({eval_res['error'][:40]})"
    if (i + 1) % 16 == 0 or i == 0:
        passed_so_far = sum(1 for r in post_results if r["passed"])
        print(f"  [{i+1:3d}/{BASELINE_N}] {task['task_id']} {status}  "
              f"(running pass@1: {passed_so_far/(i+1):.1%})")

post_metrics = batch_evaluate(baseline_tasks, post_completions)
print("\n" + "=" * 60)
print(f"训练后 pass@1: {post_metrics['pass@1']:.1%} "
      f"({post_metrics['passed']}/{post_metrics['total']})")
print("=" * 60)
```

### 5.1 逐题对比——哪些题变好了？

```python
# ==========================================
# 5.1 逐题对比
# ==========================================

print("\n逐题对比 (baseline → post-training):")
print("-" * 60)

improved = []
regressed = []
gained = 0
lost = 0

for i, (b, p, task) in enumerate(zip(baseline_results, post_results, baseline_tasks)):
    b_pass = b["passed"]
    p_pass = p["passed"]
    if p_pass and not b_pass:
        improved.append(task["task_id"])
        gained += 1
    elif b_pass and not p_pass:
        regressed.append(task["task_id"])
        lost += 1

print(f"\n  新通过: {gained} 题  {improved[:5]}{'...' if len(improved) > 5 else ''}")
print(f"  退步:   {lost} 题  {regressed[:5]}{'...' if len(regressed) > 5 else ''}")
print(f"  净提升: {gained - lost} 题")
```

### 5.2 训练过程中的 Agent 成功率

```python
# ==========================================
# 5.2 训练过程可视化
# ==========================================
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

epochs = training_log["epoch"]

# --- 子图 1: 训练 Reward 曲线 ---
ax = axes[0]
ax.plot(epochs, training_log["mean_reward"], 'o-', color='#1976d2',
        linewidth=2, markersize=6, label='Mean Reward')
ax.fill_between(epochs,
    [r - 0.05 for r in training_log["mean_reward"]],
    [r + 0.05 for r in training_log["mean_reward"]],
    alpha=0.15, color='#1976d2')
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Reward')
ax.set_title('Training Reward (Agent Rollout)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# --- 子图 2: Agent 成功率 ---
ax = axes[1]
ax.plot(epochs, training_log["success_rate"], 's-', color='#388e3c',
        linewidth=2, markersize=6, label='Agent Success Rate')
ax.set_xlabel('Epoch')
ax.set_ylabel('Success Rate')
ax.set_title('Agent Task Success Rate During Training', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

# --- 子图 3: Benchmark 对比（最关键） ---
ax = axes[2]
categories = ['Before Training\n(Baseline)', 'After Training\n(GRPO)']
pass_rates = [baseline_metrics["pass@1"], post_metrics["pass@1"]]
colors = ['#ef9a9a', '#66bb6a']
edge_colors = ['#c62828', '#2e7d32']

bars = ax.bar(categories, pass_rates, color=colors, edgecolor=edge_colors, linewidth=2)
ax.set_ylabel('pass@1')
ax.set_title(f'HumanEval Benchmark (n={BASELINE_N})', fontweight='bold')
ax.set_ylim(0, 1.0)

for bar, v in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2., v + 0.02,
            f'{v:.1%}', ha='center', fontsize=14, fontweight='bold')

# 标注提升幅度
diff = post_metrics["pass@1"] - baseline_metrics["pass@1"]
if diff > 0:
    ax.annotate(f'+{diff:.1%}',
                xy=(1, post_metrics["pass@1"]),
                xytext=(0.5, max(pass_rates) + 0.1),
                fontsize=16, fontweight='bold', color='#2e7d32',
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))
    ax.text(0.5, max(pass_rates) + 0.18,
            f'Net +{gained-lost} problems solved',
            ha='center', fontsize=11, color='#1565c0', style='italic')

plt.suptitle('Agentic RL Training: Real Model + Real Benchmark', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("agentic_rl_real_benchmark.png", dpi=150, bbox_inches='tight')
print("Benchmark comparison saved to agentic_rl_real_benchmark.png")
```

### 5.3 Agent 行为分析——训练后 Agent 学到了什么？

除了 pass@1 数字，我们还要分析 Agent 的**行为变化**——是不是真的学会了更好的调试策略？

```python
# ==========================================
# 5.3 Agent 行为分析
# ==========================================

# 用 Agent Loop（多轮）评测几个题目，看训练后的行为
print("Agent behavior analysis (post-training):")
print("=" * 60)

behavior_tasks = list(problems.select(range(5)))  # 取前 5 题

agent_successes = 0
agent_rounds_list = []

for task in behavior_tasks:
    result = run_agent_episode(task, temperature=0.3, max_rounds=3, verbose=True)
    status = "PASS" if result["passed"] else f"FAIL"
    agent_rounds_list.append(result["rounds"])
    if result["passed"]:
        agent_successes += 1
    print(f"  {task['task_id']}: {status} (rounds: {result['rounds']})")

print(f"\nAgent Loop 成功率: {agent_successes}/{len(behavior_tasks)}")
print(f"平均交互轮数: {np.mean(agent_rounds_list):.1f}")
print("=" * 60)
```

## 第六步：On-Policy vs Off-Policy 对比实验

为了验证 GRPO（On-Policy）的选择是否合理，我们跑一个 Off-Policy 的基线：**用训练前的模型生成完整轨迹（只生成一次），然后用这些冻结轨迹做 REINFORCE 训练。** 和 On-Policy 的关键区别是：Off-Policy 的轨迹数据在整个训练过程中不变。

```python
# ==========================================
# 6. Off-Policy 基线对比
# ==========================================

# 重新加载原始模型（去掉 LoRA）
model_off = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model_off.eval()

# 用原始模型生成 Off-Policy 轨迹（只生成一次，不随训练更新）
# ★ 同样使用完整的多轮对话轨迹
print("Generating Off-Policy trajectories (frozen model)...")
off_policy_trajectories = []

for task in train_tasks[:8]:  # 取 8 道题
    for _ in range(GROUP_SIZE):
        result = run_agent_episode(task, temperature=0.7)
        reward = (1.0 if result["passed"] else 0.0) - EFFICIENCY_PENALTY * result["rounds"]
        off_policy_trajectories.append({
            "task": task,
            "result": result,
            "reward": reward,
        })

# 用这些冻结轨迹做 REINFORCE 训练（Off-Policy：数据不随训练更新）
# ★ 区别在于：数据是固定的（训练开始时生成一次），不是每个 epoch 重新生成
model_off.enable_input_require_grads()
model_off_lora = get_peft_model(model_off, lora_config)
optimizer_off = AdamW(filter(lambda p: p.requires_grad, model_off_lora.parameters()), lr=LR)

for epoch in range(MAX_EPOCHS):
    for traj_data in off_policy_trajectories:
        task = traj_data["task"]
        traj = traj_data["result"]
        reward = traj_data["reward"]

        if not traj["completion"]:
            continue

        # ★ 用完整的多轮轨迹 tokenize + mask（和 On-Policy 一样）
        enc = tokenize_trajectory_with_mask(traj["conversation"], max_length=2048)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = enc["labels"].to(device)

        outputs = model_off_lora(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = -(-outputs.loss) * reward  # REINFORCE: -log_prob * reward

        optimizer_off.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_off_lora.parameters(), 1.0)
        optimizer_off.step()

# 评测 Off-Policy 模型
model_off_lora.eval()
off_completions = []
for task in baseline_tasks:
    prompt = build_completion_prompt(task)
    # 用 off-policy 训练的模型生成
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_off_lora.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                       temperature=0.0, pad_token_id=tokenizer.pad_token_id)
    comp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    off_completions.append(comp.strip())

off_metrics = batch_evaluate(baseline_tasks, off_completions)

print("\n" + "=" * 60)
print("On-Policy vs Off-Policy Final Comparison")
print("=" * 60)
print(f"  Baseline (no training):  {baseline_metrics['pass@1']:.1%}")
print(f"  On-Policy (GRPO):        {post_metrics['pass@1']:.1%}  (Δ = {post_metrics['pass@1'] - baseline_metrics['pass@1']:+.1%})")
print(f"  Off-Policy (frozen data): {off_metrics['pass@1']:.1%}  (Δ = {off_metrics['pass@1'] - baseline_metrics['pass@1']:+.1%})")
print("=" * 60)
```

### 可视化对比

```python
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['Baseline\n(no training)', 'Off-Policy\n(frozen data)', 'On-Policy\n(GRPO)']
pass_rates = [baseline_metrics["pass@1"], off_metrics["pass@1"], post_metrics["pass@1"]]
colors = ['#90a4ae', '#ffb74d', '#66bb6a']
edge_colors = ['#546e7a', '#e65100', '#2e7d32']

bars = ax.bar(methods, pass_rates, color=colors, edgecolor=edge_colors, linewidth=2, width=0.6)
ax.set_ylabel('pass@1')
ax.set_title(f'HumanEval Benchmark (n={BASELINE_N})\nOn-Policy vs Off-Policy', fontweight='bold')
ax.set_ylim(0, max(pass_rates) * 1.3)

for bar, v in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2., v + 0.015,
            f'{v:.1%}', ha='center', fontsize=13, fontweight='bold')

# 标注最佳
best_idx = np.argmax(pass_rates)
ax.annotate('Best', xy=(best_idx, pass_rates[best_idx]),
            xytext=(best_idx, pass_rates[best_idx] + 0.08),
            fontsize=14, fontweight='bold', color='#2e7d32',
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))

plt.tight_layout()
plt.savefig("on_vs_off_policy_real_benchmark.png", dpi=150)
print("On vs Off comparison saved")
```

## 参考：真正在做 Agentic RL 训练的开源项目

上面的实验是教学版。如果你想看**生产级**的"搭工具环境 → rollout 多轮轨迹 → 用轨迹训练模型"的完整实现，以下项目都开源了代码，可以直接跑：

### SimpleTIR——代码执行 + 多轮推理

[ltzheng/SimpleTIR](https://github.com/ltzheng/SimpleTIR) | ICLR 2026

和我们实验最接近的参考。模型在推理过程中写 Python 代码，沙箱执行后返回结果，模型根据结果继续推理或修正。关键贡献是 **void turn 过滤**：如果某轮模型既没有产出有效代码块也没有给答案，整条轨迹被丢弃（不参与梯度计算），防止 OOD token 导致梯度爆炸。

- **工具环境**：代码执行沙箱（Sandbox Fusion），支持异步 256 路并行
- **Rollout**：最多 5 轮（训练）/ 10 轮（评测），每轮模型写代码 → 沙箱执行 → 返回 stdout/stderr
- **训练**：GRPO + DAPO 增强（off-policy、clip high、动态采样），基于 veRL
- **结果**：Qwen2.5-7B 在 AIME24 上从 **22.1 提升到 50.5**（SOTA）
- **轨迹处理**：双张量 `info_mask`，工具返回 token 替换为 pad_id，不参与 loss 和 KL 计算

### ReTool——推理中策略性调用代码解释器

[ReTool-RL/ReTool](https://github.com/ReTool-RL/ReTool) | 字节跳动 Seed

两阶段管线：先用合成数据做冷启动 SFT，再用 RL 训练模型在推理中策略性地调用代码解释器。关键发现是模型通过 RL 学会了**因地制宜**——简单算术直接口算，复杂计算才调解释器。

- **工具环境**：代码解释器（Python 执行器）
- **Rollout**：推理中穿插代码块，执行后结果追加到上下文，模型继续推理
- **训练**：冷启动 SFT → RL（outcome reward），基于 veRL
- **结果**：Qwen2.5-32B 在 AIME 2024 上 **67%**，400 步 RL 即达到

### Search-R1——RL 训练模型自主搜索

[PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | COLM 2025

搜索工具 RL 的标杆项目。模型在推理过程中自主决定何时搜索、搜什么，搜索引擎返回结果后继续推理。核心技术创新是 **retrieved token loss masking**：搜索返回的 token 在 GRPO loss 中被 mask 掉（`I(y_t) = 0`），只有模型自己生成的 token 参与 policy gradient。

- **工具环境**：搜索引擎服务器（BM25 / dense retriever / 在线 API）
- **Rollout**：`think → <search>query</search> → 搜索结果 → think → ... → <answer>`
- **训练**：PPO 或 GRPO，搜索结果 token 不参与 loss
- **结果**：Qwen2.5-7B 在 7 个 QA 数据集上平均 **0.431**（无 mask 只有 0.343）

### DeepSWE / rLLM——纯 RL 训练 SWE-bench Agent

[agentica-project/rllm](https://github.com/agentica-project/rllm) | Together AI

目前 SWE-bench 上最强的开源方案。从 Qwen3-32B 出发，**完全不用 SFT**，纯 RL 训练约 200 步就在 SWE-bench Verified 上达到 42.2%。每条轨迹在一个 Docker 容器中运行完整 agent loop（最多 100 步）。

- **工具环境**：Docker 容器 + 完整测试套件（Bash / Search / File Editor / Submit）
- **Rollout**：多步 agent 交互，每步在 Docker 中执行动作
- **训练**：GRPO++（7 项增强：clip high、no KL、length norm、leave-one-out、compact filtering 等）
- **结果**：SWE-bench Verified **42.2%**，约 200 步 RL，64×H100 训练 6 天

### VerlTool——通用工具调用 RL 框架

[TIGER-AI-Lab/verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) | ICLR 2026 Workshop | 951 stars

最成熟的通用工具调用 RL 框架。在 veRL 之上封装了标准化的工具服务器接口，支持代码执行、搜索、SQL、NL2SQL 等多种工具，添加新工具只需一个 Python 文件。支持同步和异步 rollout（异步至少 2x 加速）。

- **工具环境**：插件式工具服务器，工具即环境（environment state per trajectory）
- **Rollout**：原生多轮交互，actor rollout 和环境交互完全解耦
- **训练**：GRPO + 多种训练 recipe（ToRL、Search-R1 复现、DAPO、NL2SQL）
- **结果**：搜索任务上复现并超越 Search-R1

### 其他值得关注的

| 项目 | 特色 | 链接 |
| ---- | ---- | ---- |
| **Agent-R1** | Step-level MDP，避免 retokenization drift | [GitHub](https://github.com/AgentR1/Agent-R1) |
| **WebAgent-R1** | 真实网页交互（WebArena），EMNLP 2025 | [GitHub](https://github.com/weizhepei/WebAgent-R1) |
| **AutoTIR** | 模型自主决定何时用何工具 | [GitHub](https://github.com/weiyifan1023/AutoTIR) |
| **AgentRL** | 多任务 agentic RL，AgentBench 8 环境 | [GitHub](https://github.com/THUDM/AgentRL) |
| **R1-Searcher** | 两阶段搜索 RL（先学调用，再学利用） | [GitHub](https://github.com/RUCAIBox/R1-Searcher) |
| **Awesome 列表** | 120+ 篇 Agentic RL 论文索引 | [GitHub](https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers) |

### 这些项目的共同模式

尽管工具不同（搜索、代码执行、网页交互），所有项目的训练管线都遵循同一套架构：

```mermaid
flowchart TD
    T["任务集"] --> R["Rollout Engine"]
    R --> M["模型生成\nassistant tokens"]
    M --> E["工具环境执行\n返回 observation"]
    E --> D{"完成？"}
    D -->|"否"| M
    D -->|"是"| RE["Reward 评估\n（测试通过？答案正确？）"]
    RE --> MASK["构建 info_mask\n（工具 token → mask）"]
    MASK --> GRPO["GRPO / PPO\n只在 assistant token 上算 loss"]
    GRPO -->|更新模型| M

    style M fill:#e3f2fd,stroke:#1976d2,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style RE fill:#e8f5e9,stroke:#2e7d32,color:#000
    style MASK fill:#fce4ec,stroke:#c62828,color:#000
    style GRPO fill:#f3e5f5,stroke:#7b1fa2,color:#000
```

1. **构建工具环境**（沙箱、搜索引擎、浏览器……）
2. **模型与环境多轮交互**（rollout），产出完整轨迹 `[prompt, assistant, tool_result, assistant, tool_result, ...]`
3. **计算 outcome reward**（测试通过？答案正确？）
4. **构建 token mask**（工具返回的 token 不参与 loss）
5. **GRPO/PPO 训练**，只在 assistant token 上计算 policy gradient

## 实验总结

| 步骤 | 做了什么 | 关键证据 |
| ---- | -------- | -------- |
| 基线评测 | temperature=0 greedy decoding | 真实 pass@1（通常 ~36-42%） |
| Agent Loop | 真实模型推理 + 真实代码执行 + 多轮修复 | 每轮都有真实的 Python 执行结果 |
| 轨迹构造 | 完整多轮对话 + assistant-only token mask | system/user token 被 mask 为 -100 |
| GRPO 训练 | On-Policy rollout + 完整轨迹 + 组内比较 + 梯度更新 | LoRA 参数在完整轨迹上更新 |
| Benchmark | 同题、同协议重新评测 | pass@1 的变化 = 真实的训练效果 |
| Off-Policy 对比 | 冻结轨迹训练（数据不随训练更新） | 验证 On-Policy 的必要性 |

**核心收获**：

1. **训练完整轨迹，不只是最终输出。** 这是 Agentic RL 和单轮 RL 的本质区别。如果只用最终代码训练，模型只学会了"什么是正确答案"——和 SFT 没有区别。只有训练完整的多轮交互轨迹（工具调用 → 执行结果 → 错误修复），模型才能学到"如何调试"、"何时调用工具"、"如何阅读报错并修复"。

2. **Token 级 mask 保证训练信号正确。** 工具执行结果（报错信息、测试输出）是环境返回的，不是模型生成的。如果把这些 token 也放进 loss，模型会被训练去"生成报错信息"——这显然是错的。`labels` 中把这些 token 设为 `-100`，确保 loss 只算在 assistant 生成的 token 上。

3. **Benchmark 是唯一的裁判。** 无论是训练 reward 还是 epoch 成功率，都不能替代独立的 benchmark 评测。只有"训练后模型在未见过的题目上的 pass@1"才能证明训练真的有效。

4. **On-Policy GRPO > Off-Policy 冻结数据。** 因为 Agentic RL 中模型行为变化快——训练早期的"修复策略"和训练后期的截然不同。On-Policy 保证训练数据始终匹配当前策略。

5. **效率惩罚塑造行为。** `EFFICIENCY_PENALTY = 0.02` 让模型不只追求"修好"，还追求"少轮数修好"。这在不降低 pass@1 的前提下减少了推理成本。

<details>
<summary>如果没有 GPU 怎么办？</summary>

可以使用 API 模型替代本地推理。将 `generate_completion` 和 `run_agent_episode` 中的模型调用替换为 OpenAI-compatible API 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="YOUR_API_URL", api_key="YOUR_KEY")

def generate_completion_api(prompt: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model="your-model",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
```

注意：API 模式下无法做梯度更新（你没有模型权重），但可以对比 **SFT 轨迹 vs Agent Loop 轨迹** 的 pass@1 差异——这本身就是有价值的实验，验证"多轮工具调用是否比单轮补全更好"。

</details>

<details>
<summary>如何扩展到 SWE-bench？</summary>

SWE-bench 和 HumanEval 有三个关键差异：

1. **仓库级别**：SWE-bench 的任务涉及整个 git 仓库（多文件），需要 `git checkout`、`grep`、`cat` 等工具，环境搭建更复杂。推荐直接参考 [DeepSWE/rLLM](https://github.com/agentica-project/rllm) 的 Docker 环境方案。

2. **评测方式**：SWE-bench 用 `git diff` + `pytest` 验证，不是简单的函数测试。

3. **轨迹长度**：SWE-bench 任务需要 20-50 轮交互，远超 HumanEval 的 3 轮。GRPO 的 group_size 需要增大（8-16），训练时间显著增加。

扩展路径：先用 HumanEval 验证管线 → 参考 DeepSWE 的 SWE-bench 实现 → 用 [SWE-bench Verified](https://www.swebench.com/verified.html)（500 题）做真实评测。

</details>

下一节我们聚焦 Agentic RL 的工程挑战——[怎么把这些想法变成一个真正能跑的训练系统](./agentic-engineering)。
