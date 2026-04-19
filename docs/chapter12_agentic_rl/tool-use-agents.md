# 12.3 工具调用 RL：Web Agent 与 Code Agent

上一节我们拆解了多轮 RL 的信用分配问题——7 轮交互失败了，该怪谁。现在我们聚焦另一个关键问题：模型怎么学会"使用工具"？监督微调（SFT）可以教会模型"工具调用的 JSON 格式长什么样"，但教不会它"什么时候该调用工具、调哪个工具、怎么组合多个工具"。后者需要策略性的决策能力——而这正是 RL 擅长的。

## 为什么 RL 对工具调用至关重要？

想象你在训练一个模型帮用户做数据分析。SFT 阶段你给它看了上千个"正确调用工具"的示例，模型学会了：

```json
{ "tool": "sql_query", "query": "SELECT * FROM users WHERE age > 30" }
```

它学会了这个格式。但到了实际使用时，模型面临的是策略性的决策：

- 用户问"我们的高端用户有多少"，模型需要决定：是直接查数据库？还是先搜索内部文档了解"高端用户"的定义？
- 查完数据库后发现有 10 万条记录，模型需要决定：是进一步筛选？还是做聚合统计？
- 聚合后发现数据异常，模型需要决定：是报告异常？还是尝试其他查询方式？

这些决策没有"标准答案"——不同的策略可能导致不同的结果，而 SFT 只能教模型模仿专家的轨迹，无法教它探索更优的策略。RL 的优势在于：你只需要告诉模型"最终结果对不对"，模型自己会通过试错学到"什么时候该用什么工具"。

|          | SFT                        | RL                                     |
| -------- | -------------------------- | -------------------------------------- |
| 学什么   | 工具调用的语法格式         | 何时调用、调哪个、如何组合             |
| 训练数据 | 需要人工标注的工具调用轨迹 | 只需要最终结果（成功/失败）作为 reward |
| 泛化能力 | 只能处理见过的工具组合     | 能探索新的工具使用策略                 |
| 错误恢复 | 不会教模型从错误中恢复     | 模型通过试错学会修复策略               |
| 代表工作 | Toolformer[^toolformer]    | ReTool、VERL-TOOL、ToolRL              |

## 核心方法

### ReTool：推理中调用工具[^retool]

ReTool（Reasoning with Tools）的思路是让模型在推理过程中**自由地**调用工具，而不是预先决定"什么时候调用"。模型在生成回答的过程中，随时可以"暂停"文本生成，调用一个工具（比如计算器或代码解释器），拿到结果后继续生成。

RL 的作用是优化"何时调用工具"的策略。模型可能发现：对于简单的算术题，直接口算比调用计算器更快；但对于复杂的数值计算，调用计算器更准确。这种"因地制宜"的策略，SFT 很难教会，RL 可以通过 reward 信号让模型自己摸索出来。

### VERL-TOOL：跨领域工具调用[^verltool]

VERL-TOOL 是一个跨领域的工具调用 RL 训练框架，覆盖数学推理、SQL 生成、Web 搜索、软件工程等多种场景。它的关键创新是**统一的工具调用接口**——不同领域的工具（计算器、数据库、搜索引擎）被抽象为统一的 RL 动作空间，可以用同一套 RL 算法训练。

### ToolRL：工具作为 RL 动作[^toolrl]

ToolRL 将工具调用视为 RL 中的一个**特殊动作**，扩展策略的动作空间。标准 LLM 的动作空间是词汇表（几万个 token），ToolRL 在此基础上增加了"调用工具 A"、"调用工具 B"等动作。策略网络需要在"生成文本"和"调用工具"之间做出选择。

```python
class ToolAugmentedPolicy(nn.Module):
    """工具增强的策略网络：在文本生成和工具调用之间做选择"""

    def __init__(self, base_model, tools):
        super().__init__()
        self.base_model = base_model  # 基座 LLM
        self.tools = tools             # 可用工具列表

    def forward(self, state):
        """
        给定当前状态（对话历史 + 工具返回结果），
        决定下一步是生成文本还是调用工具
        """
        # 基座模型输出 logits
        logits = self.base_model(state)

        # 检测特殊的"工具调用 token"
        # 如果模型选择了工具调用 token，则解析参数并执行
        if self._is_tool_call(logits):
            tool_name, tool_args = self._parse_tool_call(logits)
            return ToolAction(tool_name, tool_args)
        else:
            return TextAction(logits)  # 正常文本生成
```

## 奖励设计：场景决定 Reward

工具调用的 reward 不像偏好对齐那样主观——它可以根据客观信号来设计。这实际上就是第 8 章提到的 **RLVR（Reinforcement Learning from Verifiable Rewards）** 在 Agentic 场景的直接应用。

| 场景     | Reward 来源            | 类型                | 特殊考量                |
| -------- | ---------------------- | ------------------- | ----------------------- |
| 代码生成 | 单元测试通过率         | 连续（0-1）         | 部分通过也有部分 reward |
| 数学推理 | 最终答案是否正确       | 二元（0/1）         | 中间步骤可用 PRM        |
| Web 搜索 | 是否找到正确答案       | 二元 + 路径效率惩罚 | 鼓励更少的搜索轮次      |
| SQL 生成 | 查询结果是否匹配预期   | 二元 + 执行时间惩罚 | 避免生成低效查询        |
| 数据分析 | 分析结论是否正确且完整 | 多维评分            | 同时评估准确性和可读性  |

一个值得注意的模式：很多 Agentic 场景的 reward 都包含**效率惩罚**。这不只为了让模型更快，更因为每次工具调用都有成本（API 费用、延迟、资源消耗）。一个好的 Agent 不只是"能完成任务"，还要"高效地完成任务"。

形式化地，工具调用 RL 的总 reward 可以表示为：

$$R_{\text{total}} = R_{\text{task}} - \lambda_{\text{efficiency}} \cdot T - \lambda_{\text{format}} \cdot \mathbb{1}(\text{format error})$$

其中 $R_{\text{task}}$ 是任务完成奖励（0 或 1），$T$ 是使用的交互轮数，$\lambda_{\text{efficiency}}$ 是效率惩罚系数，$\lambda_{\text{format}}$ 是格式错误惩罚。这个公式把"成功完成任务"和"高效完成任务"统一到了一个 reward 信号中。

```python
def compute_agent_reward(task_success, num_turns, max_turns=10):
    """计算 Agentic RL 的综合 reward"""
    # 任务完成的基础 reward
    success_reward = 1.0 if task_success else 0.0

    # 效率惩罚：使用轮次越多，惩罚越大
    efficiency_penalty = -0.1 * (num_turns / max_turns)

    # 工具调用格式错误的额外惩罚
    # （如果模型生成了无法解析的工具调用）
    format_penalty = -0.5 if has_format_error else 0.0

    return success_reward + efficiency_penalty + format_penalty
```

## Web Agent RL：教模型上网

Web Agent 是 Agentic RL 最直观的应用之一：训练一个能够浏览网页、填写表单、搜索信息的智能体。这听起来简单，实现起来却充满了挑战。

**动作空间**。Web Agent 的动作不是"生成文本"，而是浏览器级别的操作：点击某个元素、在输入框中输入文字、滚动页面、导航到新 URL。每个动作都需要精确定位目标元素——这通常通过坐标（x, y）或 DOM 元素 ID 来实现。

**状态空间**。Web Agent 接收的状态通常是两部分：页面截图（视觉信息）和 DOM 树（结构信息）。截图提供了视觉布局，DOM 树提供了精确的元素定位。两者缺一不可——仅用截图很难精确点击小按钮，仅用 DOM 树又无法理解视觉布局。

**奖励信号**。Web Agent 的 reward 基于任务完成度。比如"在携程上预订一张明天北京到上海的机票"，reward 取决于：是否找到了正确的航班？是否成功填写了所有信息？是否提交了订单？

```mermaid
flowchart TD
    U["用户任务：在电商网站搜索并购买指定商品"] --> A1["Agent 观察页面\n（截图 + DOM 树）"]
    A1 --> D1["决策：在搜索框输入关键词"]
    D1 --> E1["执行：定位搜索框 → 输入文字 → 点击搜索"]
    E1 --> A2["Agent 观察搜索结果"]
    A2 --> D2["决策：从结果中选择目标商品"]
    D2 --> E2["执行：点击商品链接"]
    E2 --> A3["Agent 观察商品详情页"]
    A3 --> D3["决策：点击'加入购物车'"]
    D3 --> E3["执行：定位按钮 → 点击"]
    E3 --> R["Reward：任务是否完成？"]

    style U fill:#e3f2fd,stroke:#1976d2,color:#000
    style D1 fill:#fff3e0,stroke:#f57c00,color:#000
    style D2 fill:#fff3e0,stroke:#f57c00,color:#000
    style D3 fill:#fff3e0,stroke:#f57c00,color:#000
    style R fill:#e8f5e9,stroke:#388e3c,color:#000
```

Web Agent RL 的主要挑战是状态空间的巨大规模和动态性。一个网页可能有上千个 DOM 元素，页面内容会动态加载，同一个网站在不同时间的布局可能不同。这意味着 Agent 需要强大的泛化能力——不能记住"某个按钮在屏幕左上角"，而是要理解"提交按钮通常长什么样"。

### ReLook：用眼睛给网页打分[^relook]

现有的 Web Agent reward 主要依赖 DOM 结构匹配或任务完成度的二元判断。ReLook 引入了一种全新的 reward 来源——**视觉反馈**。它的工作流程是：Agent 生成网页代码 → 渲染成截图 → 用多模态 LLM 对截图进行视觉评分 → 将视觉评分作为 RL 的 reward 信号。这种"看到效果再打分"的方式，比纯文本 reward 更符合人类对"好网页"的判断——毕竟用户看到的是渲染后的页面，不是源代码。

### Agent Workflow Memory：从经验中学习工作流[^awm]

Agent Workflow Memory（AWM）解决的是 Web Agent 的**记忆**问题。AWM 从 Agent 过去的成功经验中抽取可复用的工作流（workflow），并在未来的任务中主动提供相关的工作流来指导 Agent 的行动。比如，Agent 在多次购物任务中学到了"先搜索 → 加购物车 → 填写地址 → 支付"这个通用流程，AWM 就会把这个工作流存储起来，下次遇到类似的购物任务时自动激活。AWM 在 WebArena 和 Mind2Web 上的实验表明，这种"从经验中学习"的方式显著提升了 Agent 在新网站上的泛化能力。

### Web-Shepherd：专为网页导航打造的 PRM[^webshepherd2]

上一节我们提到 Web-Shepherd 作为 PRM 在真实场景的落地案例。这里我们更详细地讨论它在 Web Agent 中的具体应用。传统的 Web Agent reward 只有"任务最终是否完成"这一个信号。Web-Shepherd 在每一步操作后都给出一个评分——"这次点击是不是对的？""这个表单填得对不对？"。它用结构化的检查清单（checklist）来引导评估，配合 4 万条标注数据训练出的专用 PRM，评估成本仅为 GPT-4o-mini 做判官的 1/10。这让 Web Agent 的训练可以用更密集的步骤级 reward 来替代稀疏的 episode 级 reward，训练效率和最终性能都大幅提升。

## Code Agent RL：写代码、调试、迭代

Code Agent RL 训练的是能够**写代码、执行代码、阅读报错、修复代码**的智能体。这比 Web Agent 更接近"真实程序员"的工作方式——不是一次性写出完美代码，而是通过"写 → 执行 → 报错 → 修复"的循环迭代来完成任务。

Code Agent 的 RL 训练有一个天然的优势：**reward 非常明确**。代码要么通过所有单元测试（reward = 1），要么不通过（reward < 1，按通过率给部分分）。这比 Web Agent 的"任务完成度"更容易量化和自动化。

```python
def code_agent_reward(generated_code, test_cases):
    """Code Agent 的 reward：基于测试通过率"""
    results = []
    for test_input, expected_output in test_cases:
        try:
            # 在沙箱中执行生成的代码
            actual_output = execute_in_sandbox(generated_code, test_input)
            results.append(actual_output == expected_output)
        except Exception:
            results.append(False)  # 执行异常 = 测试不通过

    # 基础 reward = 通过率
    pass_rate = sum(results) / len(results)

    # 额外奖励：代码简洁性（越短越好，但有最低要求）
    # 额外惩罚：执行时间过长
    return pass_rate
```

Code Agent RL 的一个关键发现来自 ICML 2025 的研究：**单步 reward 可以有效引导多轮代码生成**。也就是说，你不需要对每一轮的"写代码 → 执行 → 报错 → 修复"过程都给 reward——只需要给最终代码是否通过测试这一个信号，模型就能学会"写出正确的代码 → 修复错误"的完整策略。这和上一节的 ORM 思路一致——如果 reward 足够明确，稀疏信号也能工作。

### rStar2-Agent：14B 打败 671B 的实战标杆[^rstar2]

如果说前面的讨论还在讲"理论上的可行性"，那么 rStar2-Agent 就是最有力的实战证明。微软训练的这个 14B 参数模型，在 64 张 AMD MI300X GPU 上用**仅 510 步 RL 训练**，就在 AIME24 数学竞赛上达到了 80.6% 的准确率——超越了 671B 参数的 DeepSeek-R1。

rStar2-Agent 的核心创新是 **GRPO-RoC**（Group Relative Policy Optimization with Resampling on Correct）算法。传统 GRPO 在组内采样后只做一次比较，而 GRPO-RoC 会**对正确的轨迹做重采样**——如果模型在某条轨迹上成功了，就在这条成功轨迹的基础上继续探索，看能不能找到更好的路径。这比单纯的组内比较提供了更精细的学习信号。

这个结果有两个重要 insight：(1) **Agentic RL 的训练效率远超预期**——510 步 RL 训练就能超越 40 倍大的模型，说明 RL 的数据效率在大模型上非常高；(2) **小模型 + RL 可以打败大模型 + SFT**，关键在于 RL 让模型学会了"如何有效地使用工具和推理策略"，而不仅仅是模仿专家行为。

### Agnostics：任何语言都能做代码 RL[^agnostics]

现有的 Code Agent RL 几乎都默认用 Python 做代码执行和验证。Agnostics 打破了这个限制：通过一个**语言无关的代码执行验证器**，它可以对任何编程语言做 RL 训练。验证器的工作流很简单：从模型输出中提取代码 → 编译（如果需要）→ 执行 → 对比结果。无论代码是 Python、Rust、Go 还是 SQL，验证器都一视同仁。这意味着你可以用同一套 RL 框架，训练一个"什么语言都会写"的代码模型——而不需要为每种语言单独设计训练管线。代码、数据和配置均已开源。

### 不执行也能评分：Agentic Code Reasoning[^agcodereason]

到目前为止，所有 Code Agent RL 的 reward 都依赖**代码执行**——运行代码，看结果对不对。但 Meta 的研究表明，还有一种更优雅的方式：**让模型在不执行代码的情况下推理出代码的行为**。这种方法叫"半形式化推理"（Semi-Formal Reasoning）：模型需要显式列出前提、追踪每条执行路径、写出形式化的结论——就像数学证明一样，不能跳步，不能含糊。

这种方法在真实世界的补丁验证上达到了 93% 的准确率。它的核心价值在于：**不需要沙箱，不需要执行环境，没有安全风险**。你可以把它理解为 Code Agent RL 的"低成本替代方案"——如果 reward 信号只需要"这段代码大概是对的还是错的"，半形式化推理就够了；如果需要精确的输出匹配，还是得老老实实执行代码。

### 代码自举的 Scaling Law[^zeroscaling]

第 8 章讨论了 RL 的 Scaling Law——更多训练步数，更强推理能力。Agentic RL 领域也有自己的 Scaling Law。ZeroTIR 方法让模型在**没有监督示例**的情况下，自发学会生成和执行代码来辅助推理。研究者发现了一个可预测的关系：训练步数与代码执行频率、最终准确率之间存在**幂律关系**。这意味着你可以在训练早期就预测出最终模型的表现——如果训练了 100 步后代码执行频率还在上升，说明模型还在持续学习，可以继续训练；如果频率已经趋于平稳，说明学习接近饱和，可以提前停止。

这个发现对工程实践非常重要：它给了你一个**免费的训练进度指示器**——不需要跑完整个训练，只需监控代码执行频率就能判断"该不该继续训"。ZeroTIR 被 NeurIPS 2025 收录。

<details>
<summary>思考题：Web Agent 和 Code Agent 的 reward 设计有什么本质区别？这对 RL 训练策略有什么影响？</summary>

Web Agent 的 reward 通常是**二元且不可分**的——要么任务完成了，要么没完成，中间状态很难量化。这导致 reward 信号极度稀疏，训练难度大。

Code Agent 的 reward 是**可分的**——10 个单元测试过了 7 个，reward = 0.7。这种连续的 reward 信号让训练更容易：即使代码不完全正确，模型也能得到"方向对了"的信号。这就是为什么 Code Agent RL 的进展比 Web Agent RL 快得多。

对训练策略的影响是：Web Agent RL 更需要 PRM（每步评估）来提供密集信号，而 Code Agent RL 用 ORM（只看最终测试结果）就能取得不错的效果。

</details>

## 搜索工具 RL：SearchR1 与搜索增强推理

前面讨论的 Web Agent 和 Code Agent 各有侧重，但有一个工具场景特别重要、也特别有挑战性：**搜索引擎**。搜索和计算器、数据库等工具不同——搜索返回的结果是开放式的、非结构化的，而且"好"的搜索策略极度依赖上下文。问"GRPO 和 PPO 的区别"时，模型不需要搜索；但问"2025 年诺贝尔物理学奖得主是谁"时，模型必须搜索——内部知识可能已经过时。

2025 年，SearchR1[^searchr1]（Jin et al., 819 次引用）开创性地将 RL 引入搜索工具训练，让模型**自主学习何时搜索、搜什么、怎么用搜索结果**。随后 ReSearch[^research]、ToRL[^torl] 等工作从不同角度推进了这一方向。

### 为什么 prompting 不够？

在 SearchR1 之前，主流做法是通过 prompting 教模型"你可以在推理过程中调用搜索引擎"。ReAct[^react]、Self-RAG[^selfrag] 等方法都走这条路。但 prompting 有三个根本局限：

**搜索时机判断无法穷举。** 你可以在 prompt 里写"当知识不确定时搜索"，但什么是"不确定"？模型可能对过时信息充满信心（不知道自己不知道），也可能对显而易见的常识过度搜索。

**搜索 query 的策略性无法模仿。** 面对"比较三个量子计算平台的最新性能数据"这种任务，搜索策略需要根据前一次搜索的结果动态调整——第一次搜"quantum computing benchmark 2025"发现太宽泛，第二次改为"IBM quantum advantage vs Google Sycamore 2025"。这种**自适应查询生成**是策略学习问题，不是语言建模问题。

**多轮搜索的长期优化。** 一个复杂任务可能需要 5-10 次搜索。过早停止 = 信息不足，过晚停止 = 资源浪费。这个 trade-off 恰恰是 RL 的用武之地。

### SearchR1 的 MDP 建模

SearchR1 将搜索增强推理建模为一个特殊的 MDP：

- **状态 $s_t$**：当前推理上下文（已生成文本 + 之前的搜索结果）
- **动作 $a_t$**：两类——(1) 继续生成 token，(2) 生成搜索 query 并触发搜索（通过 `<search>...</search>` 标记）
- **转移**：搜索动作触发搜索引擎，结果被追加到上下文中
- **奖励**：基于最终答案正确性（RLVR）+ 搜索效率惩罚

```
推理 + 搜索的交互过程：

用户: "2025 年图灵奖颁给了谁？"

模型推理: "这个问题涉及 2025 年的最新信息，我需要搜索一下。"
模型动作: <search>2025 Turing Award winner</search>
搜索返回: "The 2025 ACM Turing Award was given to..."
模型推理: "现在信息充足了。"
最终答案: [完整答案]

Reward: 答案正确性 - λ × 搜索次数
```

训练使用 GRPO 的组采样 + 组内比较。关键设计：搜索返回的文本在 loss 中被 **mask 掉**——模型不应因为搜索引擎返回了好结果而被强化。搜索行为从 RL 训练中**自发涌现**——即使 SFT 没有教过搜索，模型也会逐渐学会在合适时机触发搜索。

### SearchR1 的关键发现

- **搜索行为从 RL 中涌现**。RL 不仅能优化已知策略，还能发现新策略
- **搜索频率的 Scaling Law**。训练步数越多，模型在需要搜索的问题上搜索频率上升，在不需要搜索的问题上搜索频率下降——模型学会了区分两种场景
- **泛化到未见过的搜索场景**。在数学问题上训练的搜索策略能泛化到历史、科学问题

### 技术谱系：SearchR1 之后

| 工作                      | 核心创新                                                 | 引用         |
| ------------------------- | -------------------------------------------------------- | ------------ |
| **SearchR1**[^searchr1]   | RL 训练模型自主搜索，GRPO + RLVR                         | 819          |
| **ReSearch**[^research]   | 推理与搜索深度融合，每步推理可包含搜索策略反思           | —            |
| **ToRL**[^torl]           | 扩展到计算工具（代码执行器），发现工具使用的 Scaling Law | 131          |
| **ReTool**[^retool]       | 区分推理型 vs 计算型任务，RL 让模型策略性选择工具        | 247          |
| **ZeroTIR**[^zeroscaling] | 无监督示例下模型自发学会代码执行，幂律 Scaling Law       | NeurIPS 2025 |

ToRL[^torl] 训练的 32B 模型在数学推理上超越了不使用工具的 70B 模型，证明了"小模型 + 工具 > 大模型纯推理"。ReTool[^retool] 进一步让模型学会策略性的工具选择——不是所有问题都需要工具，而是根据问题特征动态决定。

```python
def search_reward(answer, ground_truth, num_searches, max_searches=5):
    """搜索 RL 的 reward 函数"""
    correctness = 1.0 if verify_answer(answer, ground_truth) else 0.0
    efficiency_penalty = -0.05 * num_searches  # 搜索成本惩罚
    return correctness + efficiency_penalty
```

## 工具调用策略的训练流程

把上面的概念串起来，一个完整的工具调用 RL 训练流程通常包含三个阶段：

**阶段一：SFT 教格式。** 用人类标注的工具调用轨迹做监督微调，教会模型"工具调用的 JSON 格式长什么样"。这一步不涉及策略优化——模型只是学会了如何正确地格式化一个工具调用请求。

**阶段二：RL 教策略。** 在 SFT 模型的基础上，用 RL 优化工具使用的策略。模型开始探索不同的工具使用方式——有时候该调工具却不调，有时候不该调却调了。Reward 信号（任务成功/失败）告诉模型哪种策略更好。

**阶段三：迭代优化。** 随着 RL 训练的进行，模型会发现自己策略中的弱点——比如"在某些场景下总是忘记先搜索再回答"。这些弱点可以通过增加针对性的训练数据来修复，形成一个持续改进的循环。

```python
# 简化的工具调用 RL 训练循环
def tool_rl_training_loop(
    model, tool_env, tasks, num_epochs=100, group_size=4
):
    """工具调用 RL 的核心训练循环（简化版）"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    for epoch in range(num_epochs):
        for task in tasks:
            # 生成多条轨迹（group sampling，类似 GRPO）
            trajectories = []
            for _ in range(group_size):
                traj = model.interact_with_tools(task, tool_env)
                trajectories.append(traj)

            # 计算每条轨迹的 reward
            rewards = [compute_agent_reward(t.success, t.num_turns) for t in trajectories]

            # 组内比较（GRPO 思路）：用相对排名来计算 advantage
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8
            advantages = [(r - mean_reward) / std_reward for r in rewards]

            # 策略梯度更新
            for traj, advantage in zip(trajectories, advantages):
                loss = traj.total_log_prob * (-advantage)  # 策略梯度
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()
```

注意这个训练循环的核心思路和第 8 章的 GRPO 非常相似——都是"组内采样多条轨迹，用相对比较来计算 advantage"。区别在于：GRPO 比较的是多条文本回答，这里比较的是多条工具调用轨迹。

## 与 RLVR 的联系

你可能已经注意到，工具调用 RL 的 reward 设计和第 8 章的 RLVR 非常相似。这不是巧合——**Agentic RL 就是 RLVR 在多轮交互场景的自然延伸**。RLVR 的核心思想是"用可验证的结果作为 reward，不需要训练 Reward Model"。在工具调用场景中，工具的执行结果天然就是可验证的：代码是否通过测试、SQL 查询结果是否正确、搜索结果是否包含目标信息——这些都可以自动化验证，不需要人工标注。

这也是为什么 Agentic RL 被认为比偏好对齐（RLHF/DPO）更适合 Agent 训练的原因之一：偏好对齐需要训练一个 Reward Model 来模拟人类偏好，而 Agent 的任务通常有客观的评判标准，直接用可验证奖励就可以了。

<details>
<summary>思考题：SFT 教格式 + RL 教策略的两阶段范式，和第 2 章的 DPO 有什么异同？</summary>

相同之处在于两者都是"先 SFT 再 RL"——先用监督学习教模型基本的格式和能力，再用 RL 优化策略。这是大模型训练的通用范式。

不同之处在于目标：DPO 的 RL 阶段优化的是"回答的偏好排序"（人类更喜欢哪个回答），而工具调用 RL 的 RL 阶段优化的是"工具使用策略"（什么时候该用什么工具）。前者需要 Reward Model（或隐式的偏好模型），后者可以用可验证奖励（不需要额外的 Reward Model）。

更深层的区别是：DPO 依然在单轮范式中——输入 prompt，输出完整回答。工具调用 RL 在多轮范式中——模型需要在多步交互中做出连续决策。这使得后者面临更复杂的信用分配问题（上一节讨论的核心难题）。

</details>

下一节我们来聊聊 Agentic RL 的工程挑战和评估体系——[如何把这一切跑起来并衡量效果](./agentic-engineering)。

## 参考资料

[^toolformer]: Schick T, Dwivedi-Yu J, et al. "[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)." NeurIPS 2023. —— SFT 教工具调用格式的代表工作，证明 LLM 可以自学使用工具。

[^retool]: Feng J, et al. "[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)." arXiv:2504.11536, 2025. —— 用 RL 优化推理过程中的工具调用策略。

[^toolrl]: Qian C, Acikgoz EC, et al. "[ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)." NeurIPS 2025. —— 将工具调用视为 RL 中的特殊动作，扩展策略的动作空间。

[^verltool]: verl-tool Team. "[verl-tool](https://github.com/volcengine/verl-tool)." GitHub, 2025. —— VeRL 的工具调用增强版，跨领域工具调用 RL 训练框架。

[^rstar2]: Microsoft rStar2 Team. "[rStar2-Agent: Agentic Reasoning Technical Report](https://arxiv.org/abs/2508.20722)." arXiv:2508.20722, 2025. —— 14B 模型用 GRPO-RoC 算法在 AIME24 上达 80.6%，超越 671B DeepSeek-R1。

[^agnostics]: Boruch-Gruszecki A, et al. "[Agnostics: Learning to Code in Any Programming Language](https://arxiv.org/abs/2508.04865)." arXiv:2508.04865, 2025. —— 语言无关的代码 RL 训练管线，通过通用执行验证器支持任何编程语言。[项目主页](https://abgru.me/project/agnostics/)

[^agcodereason]: Meta Research. "[Agentic Code Reasoning](https://arxiv.org/abs/2603.01896)." arXiv:2603.01896, 2026. —— 通过半形式化推理在不执行代码的情况下进行 93% 准确率的补丁验证。

[^zeroscaling]: Agent RL Scaling Team. "[Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving](https://arxiv.org/abs/2505.07773)." NeurIPS 2025. —— ZeroTIR 方法：无监督示例下自发学会代码执行，发现训练步数与性能的幂律关系。

[^relook]: ReLook Team. "[ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding](https://arxiv.org/abs/2510.11498)." arXiv:2510.11498, 2025. —— 用多模态 LLM 对网页渲染截图进行视觉评分作为 RL reward。

[^awm]: Wang Z, Mao J, et al. "[Agent Workflow Memory](https://arxiv.org/abs/2409.07429)." ICML 2025. —— 从 Agent 过去经验中抽取可复用工作流，增强 Web 导航的泛化能力。[GitHub](https://github.com/zorazrw/agent-workflow-memory)

[^webshepherd2]: Chae H, et al. "[Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](https://arxiv.org/abs/2505.15277)." NeurIPS 2025 Spotlight. —— 首个网页导航专用步骤级 PRM，相比 GPT-4o-mini 做判官，成本降至 1/10。

[^searchr1]: Jin B, et al. "[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)." arXiv:2503.09516, 2025. 819 次引用。首次将搜索工具使用建模为 RL 问题。[GitHub](https://github.com/PeterGriffinJin/Search-R1)

[^torl]: Li X, et al. "[ToRL: Scaling Tool-Integrated RL](https://arxiv.org/abs/2503.23383)." arXiv:2503.23383, 2025. 131 次引用。将工具使用 RL 扩展到计算工具，发现工具使用的 Scaling Law。

[^research]: Chen M, et al. "[ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470)." arXiv:2503.19470, 2025. 推理与搜索深度融合框架。

[^react]: Yao S, et al. "[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)." ICLR 2023. 推理与行动协同的经典 prompting 方法。

[^selfrag]: Asai A, et al. "[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)." ICLR 2024. 通过自我反思进行检索增强生成。
