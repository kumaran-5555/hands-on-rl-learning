# B.4 Agentic RL 基础设施

> 前面的附录讲了 LLM RL 的训练架构——Rollout 和 Training 分离、异步训练、分布式并行。但当你训练的是 **Agent**（会调用工具、执行代码、多轮交互），基础设施的需求会再上一个台阶。
>
> 本节只回答一个问题：**Agentic RL 需要什么额外的基础设施？**

## 和 LLM RL 的区别

LLM RL 的环境是"生成文本"——输入 prompt，输出 completion，单次交互。

Agentic RL 的环境是**多轮、有状态、需要工具执行**的复杂交互：

```
LLM RL:     prompt → completion → reward           （单轮）
Agentic RL: prompt → action → tool_result → action → ... → reward  （多轮）
```

这带来了 5 个新的基础设施需求。

## 1. 沙箱隔离

Agent 执行代码、浏览网页、调用 API，每个 episode 需要隔离环境：

```python
sandbox = DockerSandbox(
    image="agent-env:latest",
    timeout=30,           # 单步超时
    memory_limit="512m",  # 内存限制
    network="none"        # 默认断网
)
result = sandbox.execute(agent_action)
```

为什么需要隔离？Agent 可能生成恶意代码（即使不是故意的），或者陷入无限循环。沙箱确保一个 episode 的异常不会影响其他 episode，也不会破坏宿主系统。

## 2. 轨迹存储与回放

多轮交互产生的轨迹很长且不规则，需要专门存储：

```
Trajectory {
    task: "修复这个 Python bug"
    steps: [
        {role: "user", content: "报错信息..."},
        {role: "agent", content: "我看到问题了", tool_call: "edit_file(...)"},
        {role: "tool", content: "文件已修改"},
        {role: "agent", content: "让我测试一下", tool_call: "run_python(...)"},
        {role: "tool", content: "测试通过"},
        {role: "agent", content: "修复完成"}
    ]
    reward: 1.0
    token_count: 1523
}
```

轨迹存储需要支持：

- 按 task 类型检索（分析哪类任务做得好/差）
- 按步骤切片（单独分析某一步的决策质量）
- 去重（相同 task 不重复训练）

## 3. 工具执行管理

Agent 调用的工具需要并发管理和结果缓存：

- **超时控制**：避免 Agent 卡在无限循环
- **并发执行**：多个独立工具调用同时跑
- **结果缓存**：相同输入不重复计算

工具执行是 Agentic RL 的主要延迟来源。一个 episode 可能调用 5-10 次工具，每次都有网络/IO 开销。

## 4. 多轮信用分配

这是 Agentic RL 独有的问题：

```
Agent 做了 5 步才完成任务，奖励 = 1.0
→ 哪一步贡献最大？
→ 第 3 步的 edit_file 是关键动作，应该获得更多信用
→ 需要轨迹级的奖励分解（credit assignment）基础设施
```

简单的做法是把最终奖励平均分配给每一步。更好的做法是用 Process Reward Model 对每一步单独打分。这需要在基础设施层面支持轨迹级的奖励存储和分解。

## 5. 评测集群

Agentic 任务的评测本身就是多轮交互，成本比文本评测高得多：

| 评测方式                         | 成本 | 覆盖度         |
| -------------------------------- | ---- | -------------- |
| 静态匹配（SWE-bench gold patch） | 低   | 只看结果       |
| 自动化执行（跑测试用例）         | 中   | 验证功能正确性 |
| LLM-as-Judge                     | 中高 | 评估过程质量   |
| 人工评测                         | 高   | 最可靠         |

实践中通常组合使用：自动化执行做日常回归，LLM-as-Judge 做过程质量检查，人工评测做最终验收。

## 整体架构

```
┌──────────────────────────────────────────────┐
│               Agentic RL 训练集群              │
│                                              │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Policy   │  │ 沙箱集群  │  │ 轨迹存储     │ │
│  │ (LLM)    │  │ (Docker)  │  │ (Redis/S3)  │ │
│  └────┬────┘  └─────┬────┘  └──────┬──────┘ │
│       │             │              │         │
│       ▼             ▼              ▼         │
│  ┌──────────────────────────────────────┐    │
│  │         Orchestrator（编排器）         │    │
│  │  多轮对话管理 → 工具调用 → 结果收集     │    │
│  │  信用分配 → 轨迹切分 → 奖励计算        │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

编排器（Orchestrator）是 Agentic RL 的核心组件，负责协调策略模型、沙箱、工具执行和轨迹存储之间的多轮交互。Ray 是目前最常用的编排工具，因为它原生支持异构资源调度和容错。

## 参考文献

- [^1] HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026.
- [^2] PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.

## 参考文献

[^1]: HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026.

[^2]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.
