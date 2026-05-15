---
title: 9.7 Industrial Post-Training Practice
---

# 9.7 A Full View of Industrial Post-Training Practice

When DPO, GRPO, and RLVR are placed inside real companies, post-training no longer looks like a single algorithm. It becomes an entire production system: data synthesis, SFT, preference optimization, verifiable rewards, online rollouts, tool environments, evaluation, refusal behavior, and safety policies are iterated together. The following survey organizes the mainstream practices that could be found in public materials as of 2026-05-06, with emphasis on the learnable methodological pieces in each company's disclosures: task construction, environment wrapping, reward design, the handoff between SFT and RL, training stability, and capability backfilling.

## Chinese Companies and Major Labs

### MiniMax

> **Sources**: [MiniMax M2.1: Post-Training Experience and Insights for Agent Models][^minimax_m2_1], [MiniMax-M1][^minimax_m1], [WebExplorer][^minimax_webexplorer]

MiniMax's public materials should be read along three lines: the agent post-training experience of M2.1, the long-thinking RL scaling of MiniMax-M1, and the long-horizon web-agent data synthesis and RL of WebExplorer. These three sources cover code agents, application-development agents, web-search agents, and long-context reasoning models. They should not be reduced to the phrase "verifiable environment plus reward."

#### 1. M2.1 SWE Scaling: from raw GitHub data to runnable RL environments.

**Motivation**: Software-engineering tasks are naturally verifiable, but raw GitHub data cannot be used to train agents directly. An issue, PR, or commit contains text and code differences, but that is not yet an RL sample. The system must restore the repository state, construct the task description, prepare dependencies, and determine the verification command before it becomes a closed loop of "model action -> environment feedback -> reward."

**Data construction** can be split into five steps:

- **Mine real events**: filter samples with clear repair targets from merged GitHub PRs, commits, issues, test changes, and code diffs.
- **Rewrite task forms**: SWE-Resolve asks the model to fix a bug or implement a requirement; SWE-Test reverses the task and asks the model to write tests that fail before the patch and pass after it; SWE-Review asks the model to review a code change and point out problems.
- **Wrap executable environments**: check out the repository at the pre-patch state, install dependencies, prepare test commands, and build Docker / sandbox environments so a patch submitted by the model can be verified automatically.
- **Complete training fields**: generate fields such as original problem description, test-case reward, and runnable environment, producing samples that can be used for both SFT and RL.
- **Expand languages and scenarios**: the M2.1 report mentions coverage of more than 10 major programming languages, more than 10,000 runnable PRs, and more than 140,000 variable tasks.

**Reward design** also has to be separated by task type:

- **SWE-Resolve**: the main reward is whether the tests pass, while also checking whether existing tests were broken, unrelated files were modified, or the solution merely hard-coded the tests.
- **SWE-Test**: the test must fail before the patch and pass after the patch; only then does it show that the test really covers the target bug.
- **SWE-Review**: this is not a fully executable task, so it can only be verified approximately, for example by using another LLM to check whether the review hits the real issue while controlling the hallucination rate.

The **key method** is to rewrite one real development event into multiple equivalent tasks. In this way, the same GitHub data can produce training signals for fixing, testing, reviewing, and other behaviors.

**Multi-scaffold** is another main line in M2.1. A scaffold is the execution framework around the agent, such as context-management strategy, tool-calling protocol, reflection / planning template, or file-editing interface. If SFT and RL are both performed only on one ReAct loop, the model overfits the format of that loop. MiniMax uses multi-scaffold rejection sampling to generate SFT data, and lets different scaffolds participate in rollout during RL.

A **minimal reproduction** can begin with two or three simple scaffolds: direct ReAct, plan-then-edit, and a test-driven loop. Run the same batch of SWE tasks under these scaffolds, keep successful trajectories for SFT, and then randomize the scaffold during RL so that the model learns the task itself rather than the template.

#### 2. M2.1 AppDev: from fixed tests to Agent-as-a-Verifier.

**Motivation**: It is hard to prewrite complete tests for "building an application from scratch." Whether a frontend, backend, or mobile application is complete is not determined only by function outputs. It also depends on whether the app can start, whether interaction is correct, whether the visual result is reasonable, and whether the business logic closes the loop.

**Expert data**: MiniMax introduces experts-in-the-loop. Frontend, backend, Android, and iOS experts design prompts, meta-queries, rubric-based rewards, and system prompts. Expert prompts contain best practices. During training, the system prompt can be removed from the trajectory so that the model distills the expert heuristics into its default behavior.

The **three-layer reward structure** is:

- **Execution level**: check whether the code can compile, start, and run.
- **Interaction level**: use tools such as Playwright to click pages, fill forms, and inspect state changes, judging whether the business logic is correct.
- **Visual level**: score against relatively consistent aesthetic criteria, such as whether the layout is obviously misaligned, whether key information is visible, and whether interactive controls are usable.

The **difference from ordinary LLM-as-a-judge** is that the judge does not merely inspect a static screenshot or final text. It enters the sandbox as an agent and interacts with the application.

A **minimal reproduction** can use a Todo App / login page / data-table task set: give the model a requirement and ask it to generate a project; automatically run `npm install && npm run dev`; use Playwright to run 5-10 interaction checks; finally use a rubric judge to supplement visual quality and requirement-coverage scores.

#### 3. WebExplorer: solving the shortage of hard data for long-horizon web agents.

**Motivation**: Existing open-source web agents are weak on complex information-retrieval tasks such as BrowseComp, GAIA, WebWalkerQA, and FRAMES. One core reason is the lack of high-quality, long-horizon, trainable data. Manually writing this kind of task is expensive; simple query evolution easily degenerates into unnatural questions; graph-construction methods require complex node expansion and heuristic selection.

The **core idea** of WebExplorer is to use model-based exploration so that a strong model explores the information space by itself, and then to generate harder questions through iterative long-to-short query evolution.

The synthesis process of **WebExplorer-QA** can be understood in two stages, "exploration" and "evolution":

- **Exploration stage**: starting from Wikipedia seed entities, the prompt provides three BrowseComp-en QA exemplars. The model then performs search / browse around the seed entity and constructs a local information space. This stage does not directly produce questions. It lets the model find related entities, webpages, fact chains, and potential answers that can be asked about.
- **Evolution stage**: perform long-to-short query evolution. The initial question usually contains many explicit clues. Later iterations gradually delete or compress salient information, making the question shorter, more implicit, and more dependent on multi-step search. The paper uses 5 rounds of evolution and ultimately synthesizes about 40K WebExplorer-QA examples.
- **Design focus**: first explore the answer and evidence, then make the question harder. Do not invent hard questions out of thin air.

The **training recipe** has two segments:

- **Cold-start SFT**: first teach the model to use search / browse tools correctly and form a basic long-horizon search format. The paper uses Qwen3-8B as the base, about 13K SFT samples, batch size 32, learning rate 1e-5, and trains for 4 epochs.
- **GRPO stage**: no human trajectories are required. Only synthetic QA pairs are needed, and the model explores different search paths by itself. The paper uses about 12K samples for GRPO, 8 rollouts per group, batch size 64, learning rate 1e-6, and gradually expands the maximum response length to 128K and the maximum number of tool-call rounds to 100.

The **reward formula** is a composite: `R = 0.2 * R_format + R_correct`.

- **`R_format`**: checks whether tool calls, thinking structure, and answer format obey the protocol.
- **`R_correct`**: uses DeepSeek-V3 as an automatic judge to decide whether the final answer is correct according to the ground-truth answer.
- **Weight motivation**: in web-agent RL, if format is not rewarded, the model drifts on the search / browse protocol; if only format is rewarded, the model learns to spin tools without substance. Therefore format receives only a small weight, while correctness is the main signal.

A **minimal reproduction** can proceed as follows: choose 500-1000 Wikipedia seed entities; for each seed, let a strong model use search tools to explore 5-10 related webpages and produce candidate QA pairs; use another model to filter out questions with non-unique answers, insufficient evidence, or excessive simplicity; run 2-3 rounds of query compression, gradually hiding explicit clues such as entity names, dates, and locations; keep the answer and evidence URLs; use 1K-5K trajectories for SFT so that a 7B/8B model learns the search/browse format; then run GRPO on 1K QA pairs, with reward from format checks plus an LLM judge / exact match.

At minimum, **evaluation metrics** should include three items: whether the average number of tool-call rounds increases, whether accuracy improves, and whether invalid search loops appear.

#### 4. MiniMax-M1: the efficiency problem in long-thinking RL.

**Motivation**: M1 is not primarily about agent environments. It addresses the fact that test-time compute scaling is too expensive. When long CoT outputs reach 64K, 80K, or 100K tokens, ordinary attention and ordinary RL become extremely costly.

**Model and RL**: M1 uses hybrid MoE plus Lightning Attention to support a 1M-token input context and 40K/80K thinking budgets, then uses CISPO for efficient RL. The point of CISPO is to clip importance-sampling weights rather than directly clipping token updates. This preserves the gradient contribution of all tokens even in long responses while reducing variance.

**Data and curriculum**:

- **Continued pretraining**: continue training from MiniMax-Text-01, strengthening STEM, code, books, reasoning, and long context.
- **SFT cold start**: inject a CoT mode to give RL a starting policy.
- **RL data mixture**: verifiable tasks include mathematics, competitive programming, 41 classes of logic tasks generated by SynLogic, and SWE-bench-derived software-engineering sandboxes; unverifiable tasks such as QA and creative writing use feedback from a generative reward model.
- **Gradual length expansion**: when expanding to 80K output, the system does not simply open the full length at once. It expands from 40K to 48K, 56K, 64K, 72K, and 80K.
- **Promotion metrics**: use metrics such as whether generated-sequence perplexity has converged and whether the 99th percentile output length approaches the window limit to decide whether to enter the next stage.
- **Data filtering**: the report also mentions using the 40K model to filter data, removing overly easy samples, increasing the proportion of difficult math and code, and downsampling synthetic reasoning data that would cause repetitive homogenization.

The reproducible point is not to replicate 512 H800s. It is to verify three facts on a small model: long-output training should expand the window gradually; overly easy samples should be filtered or they provide no RL signal; synthetic reasoning data with patterns that are too uniform can make long-context RL repetitive and unstable.

### Alibaba Qwen / Tongyi

> **Sources**: [Qwen2.5 Technical Report][^qwen2_5], [Qwen2.5-Math][^qwen2_5_math], [QwQ-32B][^qwq_32b], [Qwen3][^qwen3], [Qwen3-Coder][^qwen3_coder], [Qwen3-Coder-Next][^qwen3_coder_next], [Tongyi DeepResearch][^tongyi_dr]

Qwen's materials should be read in four layers: general instruct post-training, mathematical self-improvement, reasoning RL, and agentic coding / deep research. Their value is that the public materials discuss both "verifiable tasks" and "general-experience backfilling," rather than stopping at the name of a particular GRPO method.

#### 1. Qwen2.5: first make general instruct post-training multi-stage.

The post-training in the Qwen2.5 report is not just SFT. It first uses millions of supervised examples to cover general QA, code, mathematics, multilingual ability, structured data analysis, long-text generation, and safety, then performs multi-stage RL. The motivation is that the base model has knowledge, but does not stably follow user intent; one round of SFT can also make the model "able to answer" while still not preference-aligned enough. Therefore Qwen2.5 handles instruction following, long text, structured output, professional ability, and human preferences in stages. For a minimal reproduction, general SFT data should be bucketed by capability domain rather than mixed into one large JSONL. Each bucket should keep an independent evaluation set, such as long-form summarization, JSON output, code, math, refusal behavior, and multilingual tasks.

#### 2. Qwen2.5-Math: the core of math post-training is a CoT / TIR self-improvement loop.

The motivation of this report is that ordinary CoT makes mistakes in exact calculation, symbolic operations, and algorithmic reasoning, so Qwen2.5-Math trains both Chain-of-Thought and Tool-Integrated Reasoning. Data construction does not merely collect human solutions. The model generates candidate CoT / TIR solutions, and then answer parsers, a Python executor, majority voting, and a reward model filter them. Qwen2.5-Math-RM-72B is used as a math reward model for both rejection sampling and later RL.

The details of TIR training are especially important. The tokens generated by the model include natural-language thinking, Python code, tool returns, and the final answer. The executor return is not content written by the model, so the loss for executor-output tokens must be masked during training. Otherwise the model learns the wrong target: treating the environment's return as text it should generate. The RL stage uses GRPO, scores math problems with a rule verifier / reward model, and constrains policy drift with a KL coefficient. The report also emphasizes that Qwen2.5-Math mainly targets Chinese and English mathematics and is not recommended as a general chat model. This shows that specialized post-training creates capability bias; if the model is to become a product, general capability must later be backfilled.

A minimal reproduction path is: choose subsets of GSM8K / MATH / OlympiadBench; ask a strong model to generate 4-16 CoT and Python TIR candidates for each problem; filter with an answer parser, `sympy`, Python execution, and majority vote; use correct trajectories for SFT; then sample 8 responses per problem, score them with a rule verifier using 0/1 or partial credit, and run GRPO. Handle the loss mask for TIR data separately, and record tool error rate, format error rate, and final-answer accuracy.

#### 3. QwQ-32B: two-stage RL with outcome-based reward.

The public QwQ-32B blog separates RL very clearly. Starting from a cold-start checkpoint, the first stage expands only math and coding RL. It does not use a traditional reward model; instead it uses a math accuracy verifier and a code execution server. The math reward checks whether the final answer is correct; the code reward checks whether the generated code passes predefined tests. The goal of this stage is to push up "can the model solve the problem?"

The second stage performs general-capabilities RL. Rewards come from a general reward model and some rule-based verifiers, and a small number of steps improves instruction following, human preference alignment, and agent performance while avoiding obvious regression in math and code. This is a very practical recipe: first amplify capability on high-confidence verifiable tasks, then use general preference and rule data to repair the experience. In a minimal reproduction, the first stage can use only math and code; the second stage can mix in 5%-20% general instruction / preference / safety data and observe whether math, code, chat, and agent evaluations squeeze each other.

#### 4. Qwen3: unified post-training for thinking / non-thinking.

The key to Qwen3 is not simply "it used GRPO." It makes thinking mode a product-controllable capability. The first two post-training stages are long-CoT cold start and reasoning RL: cold start uses a small amount of high-quality long-thinking data to teach format, reasoning organization, and answer boundaries; reasoning RL uses query-verifier pairs for GRPO. The report discloses four criteria for filtering query-verifier pairs: they must not appear in the cold-start data; they must be learnable for the cold-start model; they should be as challenging as possible; and they should cover broad subdomains. The final set contains 3,995 query-verifier pairs. The report also emphasizes large batches, multiple rollouts per problem, off-policy methods to improve sample efficiency, and entropy control to stabilize exploration and exploitation.

The last two stages solve the problem of "a model that only knows how to think at length." Qwen3 synthesizes a unified training set containing data with reasoning paths and data without reasoning paths, so the model supports both thinking and non-thinking modes. Finally, general-domain RL backfills general ability, safety, multilingual behavior, and tool experience. This sequence gives a reproducible template: first train the `<think>` format and long CoT; then perform RL only on high-confidence verifier tasks; then mix in short answers, ordinary chat, and tool instructions so the model learns when not to unfold long thinking; finally evaluate average output length, accuracy, user preference, and non-thinking-mode quality.

#### 5. Qwen3-Coder / Tongyi DeepResearch: from answer RL to process RL.

The training object of Qwen3-Coder is repository-level action: read files, locate bugs, write patches, run tests, handle failures, and submit changes. The main reward signal comes from unit tests, static checks, compilation, issue-requirement coverage, and patch reasonableness. The training object of Tongyi DeepResearch is the search / read / synthesize process. The task is not to answer one fact, but to search for evidence, deduplicate sources, compare conflicting information, and organize a cited report. Together they show that Qwen's agent post-training has changed "prompt -> answer" into "environment episode -> verified outcome." For reproduction, start with a small SWE-bench Lite or web QA setup: fix the tool protocol, keep successful trajectories for SFT, and then use test pass rate or an answer judge for RL.

### Moonshot Kimi

> **Sources**: [Kimi k1.5][^kimi_k1_5], [Kimi K2][^kimi_k2], [Kimi-Researcher][^kimi_researcher]

Kimi's three public lines correspond to reasoning scaling, an open agentic model, and a research agent. The most useful pieces to analyze here are k1.5 and Kimi-Researcher: the former answers "how can long-thinking RL be stabilized?", while the latter answers "how does a research agent emerge from end-to-end RL?"

#### 1. Kimi k1.5: long-thinking RL without MCTS / value function / process RM.

The motivation of k1.5 is to train test-time compute scaling. It does not make the system especially complex. Instead it emphasizes a concise framework: the policy samples multiple responses, the reward only looks at outcome, and policy optimization increases the probability of high-reward trajectories under a KL constraint. The report explicitly distinguishes this from MCTS, value functions, and process reward models. The focus is not to train a step-by-step scorer, but to let the model explore more effective reasoning paths by itself through sufficiently many rollouts.

In data, k1.5 divides tasks into verifiable and preference-style categories. Math, code, and multiple-choice questions are more suitable for rule / execution verifiers; open QA, writing, and complex preferences require a reward model or judge. During training, the same prompt samples multiple candidates; after reward scoring, the candidates form relative strengths and weaknesses, and policy mirror descent updates the model. The reproducible key is to make the number of samples large enough, because a single response has noisy reward. Multiple samples for the same problem are what reveal which reasoning paths are more stable.

#### 2. Length reward: solving overthinking, not simply cutting max tokens.

The k1.5 report discusses overthinking explicitly. After a model learns long CoT, it may write more and more tokens, even continuing to wander after it has already found the answer. Its length reward does not unconditionally reward shortness. It compares multiple candidates for the same problem: among correct answers, shorter responses receive extra reward; wrong answers are not rewarded just because they are short, and long wrong answers may be penalized. This design writes "correct and efficient" into the reward instead of relying only on a generation max length.

Length reward also needs a warm-up. Early in training, the model cannot solve problems reliably. If length is penalized too early, exploration is suppressed and the model may never learn full reasoning. Only after accuracy rises should the length term be added, compressing long thinking into effective thinking. A minimal experiment can work as follows: use only correctness reward for the first 30%-50% of RL steps; then, among correct samples for each problem, rank by length and add 0.1-0.3 reward to shorter correct solutions while penalizing overly long wrong solutions; monitor both accuracy and average response length to confirm that the model is not sacrificing correctness by becoming shorter.

#### 3. Long-to-short: first learn to think, then learn to write less.

k1.5's long-to-short idea pairs with length reward. In the first stage, the model is allowed to use very long reasoning to reach correct answers and learn strategies for complex problems. In the second stage, distillation, preference optimization, or length reward removes redundant steps. This differs from "directly training a short-answer model": the short model should retain the search and self-checking ability learned by the long model while reducing useless expression. For reproduction, keep successful long-CoT trajectories, then ask a strong model or the same model to produce concise solutions, run one round of SFT / DPO, and use a verifier to confirm that the shorter solution remains correct.

#### 4. Kimi K2: the data and tool loop of agentic intelligence.

The K2 public report emphasizes open agentic intelligence. The focus is not one benchmark, but enabling the model to act in tools, code, and complex tasks. Corresponding post-training samples should contain task goals, tool protocols, observations, actions, error recovery, and final results. The lesson from K2 is that agent data cannot rely only on manually written demonstrations. It must combine real tasks, synthetic tasks, tool execution results, verifiers, and judges, continuously filtering successful trajectories and feeding them back into SFT / RL.

#### 5. Kimi-Researcher: research-agent reward must cover the evidence chain.

Kimi-Researcher targets long-horizon research tasks. Its training unit is a research episode: the model proposes a search plan, calls search/browse tools, reads multiple sources, extracts evidence, merges conflicting information, and writes a cited answer. The final reward cannot check only whether "the answer looks similar." It must also check whether citations exist, whether evidence supports conclusions, whether sources cover the key angles, whether counterexamples are missed, and whether the agent repeatedly searches low-value pages. A minimal reproduction path is: construct 200-500 questions that require evidence from multiple webpages; record trajectories with browser tools; ask a judge to score evidence coverage, citation correctness, answer faithfulness, and redundant-search penalty separately; first SFT successful trajectories, then use episode-level reward for GRPO / DPO.

### ByteDance Seed / Doubao

> **Sources**: [Seed1.5-Thinking][^seed1_5_thinking], [VAPO][^vapo], [DAPO][^dapo], [DAPO GitHub][^dapo_github], [UI-TARS][^ui_tars], [UI-TARS GitHub][^ui_tars_github], [UI-TARS-2][^ui_tars_2], [Seed Prover 1.5][^seed_prover], [Seed1.8][^seed1_8]

ByteDance Seed's public materials are useful for learning two things: engineering patches for reasoning RL, and environment-style agent post-training for GUI / prover tasks. DAPO, VAPO, and UI-TARS-2 do not merely provide algorithm names. They answer the question "why do large-scale rollouts become unstable?"

#### 1. Seed1.5-Thinking: the basic recipe for a reasoning model.

The goal of Seed1.5-Thinking is to improve math, code, and complex reasoning through RL. Its task construction still centers on verifiable questions: math is checked by answers, code by execution, and logic problems by rule verifiers. The SFT stage first gives the model a long-CoT cold start; the RL stage then amplifies verifiable ability through outcome reward. This pattern resembles DeepSeek-R1 and Qwen3, but later Seed reports emphasize training systems, sampling, and advantage handling more strongly.

#### 2. DAPO: breaking GRPO's hard training points into four patches.

DAPO's motivation is that when the open-source community tries to reproduce large-scale reasoning RL, failures often happen not because the GRPO formula is unknown, but because details of samples, clipping, length, and gradient normalization are mishandled. DAPO adds four key components on top of GRPO.

Dynamic Sampling handles prompts with "no learning signal." If all sampled answers for the same problem are correct or all are wrong, the within-group reward variance is near zero and the advantage is meaningless. DAPO keeps sampling or filtering until the batch retains groups with non-zero advantage, spending compute on boundary problems. Clip-Higher addresses the problem that exploration is suppressed by PPO clipping: in long CoT, certain low-probability tokens may open a new solution path; if the upper bound is too tight, a correct but rare reasoning path cannot be sufficiently reinforced. Therefore `eps_clip_high` is set higher than the lower bound, for example a common configuration is low 0.2 and high 0.28.

Token-Level Policy Gradient addresses the dilution of long responses by sample-level averaging. Ordinary sequence-level loss averages key tokens in long CoT together with irrelevant tokens, weakening the signal. DAPO aggregates by token so that each generated token in the long reasoning chain participates more directly in optimization. Overlong Reward Shaping handles noisy samples that exceed the length limit: if the model fills the context and still has not finished, the truncated text should not simply be treated as a normal failed sample, because the reward noise is large. Overlong responses require segmented penalties, masking, or separate shaping.

A minimal reproduction path is: use a Qwen2.5-7B/32B base, verifiable AIME/MATH-style problems, and 8-16 rollouts per problem; first run ordinary GRPO as a baseline; then add dynamic sampling, clip-higher, token-level loss, and overlong shaping one by one; record the effective prompt ratio, entropy, average length, AIME pass@1, and number of training collapses. DAPO's value appears precisely in this kind of ablation.

#### 3. VAPO: value models are not impossible; long-CoT advantage must be redesigned.

VAPO studies value-model-based RL. Under long CoT, GAE easily decays the final sparse reward into earlier tokens, and the advantage scale also differs between short and long responses. The report's ablations are informative: removing decoupled GAE causes reward signals to decay exponentially and performance to drop significantly; Length-Adaptive GAE adjusts GAE parameters according to sequence length so both short and long responses receive suitable credit; token-level policy gradient gives long responses more reasonable weight; positive-example LM loss uses the language-model loss of positive samples to stabilize the policy; group sampling uses fewer prompts with more repetitions to improve within-group comparison quality. The report also gives reproducible experimental parameters such as `epsilon_low=0.2`, `epsilon_high=0.28`, positive LM loss weight 0.1, and 512 prompts with 16 samples each.

A small VAPO reproduction does not need to train a large value model first. Start with a simplified experiment: train the same batch of math problems with GRPO and with a PPO/VAPO-style method using a value baseline, then compare advantage variance, delayed reward decay, and final accuracy on long-answer tasks. The point is not to chase SOTA, but to observe how credit assignment affects training stability in long-sequence RL.

#### 4. UI-TARS: GUI agents first learn basic operations from trajectories and preferences.

UI-TARS takes screenshots / interface states, action history, and task goals as input, and outputs GUI actions such as clicking, typing, and scrolling. Its data problem is that high-quality action traces are scarce. The learnable method in the public materials is to use many virtual machines to explore real software tasks, generate trajectories from constructed instructions, and then apply rule filtering, VLM scoring, and human review. Reflection tuning brings error recovery into training: annotators point out which step in a trajectory is wrong and provide a corrective action or recovery step, then preference optimization such as DPO biases the model toward strategies that can correct mistakes.

#### 5. UI-TARS-2: multi-turn RL, mixed environments, and a data flywheel.

The motivation of UI-TARS-2 is that GUI-only agents are not enough for real tasks. Many workflows also require the filesystem, terminal, downloaded files, and local data. It introduces a hybrid GUI environment that places GUI, filesystem, and terminal in one sandbox, and uses a large-scale rollout platform to support multi-turn RL. The data flywheel works as follows: the model generates new trajectories; high-quality trajectories enter SFT; low-quality but learnable data enters continual pretraining or later exploration; after each round, the stronger model produces harder and longer trajectories.

Rewards in this kind of system need multiple layers: whether the final task is completed, whether the interface state reaches the target, whether files are generated, whether terminal commands succeed, whether actions are invalid or out of bounds, whether the number of turns is excessive, and whether safety boundaries are violated. A minimal reproduction can use a subset of MiniWoB / BrowserGym / OSWorld: define a unified action schema; each task provides reset, observe, step, and success check; use 200 human or strong-model trajectories for SFT; then run multi-turn rollout plus success reward for RL; collect failed trajectories additionally for reflection training.

#### 6. Seed Prover 1.5: agentic RL in a formal-proof environment.

In formal mathematics, the environment is a theorem prover rather than a browser. Actions are selecting tactics, generating lemmas, and calling searchers; rewards are whether the proof passes, proof length, search steps, and whether intermediate lemmas are reused. Its lesson for agent RL is that as long as an environment can verify, a complex task can become a trainable episode. Seed1.8 then puts reasoning, multimodality, tools, and generalized agent ability into one model card, showing that post-training goals are expanding from "accuracy on problem sets" to "task execution across multiple environments."

### DeepSeek

> **Sources**: [DeepSeekMath][^deepseek_math], [DeepSeek-R1][^deepseek_r1], [DeepSeek-V3.2][^deepseek_v3_2]

DeepSeek's public materials are one of the main threads for understanding GRPO / RLVR. DeepSeekMath first provides critic-free within-group relative advantage; R1 then shows that pure rule reward can induce long thinking; V3.2 pushes verifiable tasks toward agentic task synthesis.

#### 1. DeepSeekMath: the minimal reproducible version of GRPO.

DeepSeekMath-RL starts from DeepSeekMath-Instruct 7B and uses about 144K CoT problems related to GSM8K and MATH for RL. For each problem, it samples a group of outputs, scores them with a reward model / rule correctness, and normalizes by the within-group mean and standard deviation to form advantages. This removes the critic/value model used by PPO and lowers memory usage and training complexity. Typical settings in the report include a policy learning rate of 1e-6, KL coefficient 0.04, 64 outputs per problem, max length 1024, batch size 1024, and one policy update after each exploration round.

The intuition of GRPO is that the model does not need to know "the absolute value of this answer." It only needs to know "which response in this group for the same problem is better." Math problems are naturally suitable, because multiple samples for the same problem produce correct, wrong, format-wrong, and partially correct candidates. A minimal reproduction can use a 7B math SFT model, a MATH subset, 8-16 rollouts per problem, and an answer parser for scoring. Normalize reward within the group, add a KL term to the reference model, and observe GSM8K/MATH improvement and general-ability loss. DeepSeekMath's experience is that even a model that is already strong after SFT can still gain out-of-domain reasoning improvements from RL.

#### 2. DeepSeek-R1-Zero: rule RL directly from a base model.

The motivation of R1-Zero is to test whether long CoT must come from human SFT. It starts from a base model and performs RL directly. The rewards are mainly accuracy reward and format reward: math/code problems check the final answer or execution result, while format reward ensures that the model outputs in the agreed format. After training, reflection, backtracking, self-verification, and longer thinking emerge, showing that some reasoning patterns can be induced by outcome-reward pressure.

The limitations of R1-Zero are also important: poor readability, mixed languages, and unstable output format. This shows that "pure RL can explore the capability ceiling" does not mean it is a product recipe. A reproduction experiment should treat R1-Zero as a research experiment: start from a base model, use only verifiable math/code problems, and avoid open QA; evaluate not only accuracy, but also format failure, repetition, language mixing, and average length.

#### 3. DeepSeek-R1: cold start + reasoning RL + rejection sampling + final RL.

The official R1 returns to a more engineered four-stage pipeline. First, a small amount of high-quality cold-start data corrects format, readability, and the basic long-thinking structure. Second, reasoning-oriented RL continues to reinforce verifiable tasks such as math, code, and logic. Third, the trained model is used for rejection sampling to generate more SFT data, while general data such as writing, factual QA, and role play are mixed in so the model does not only solve problems. Fourth, final RL jointly optimizes helpfulness, harmlessness, and reasoning.

The core lesson of this pipeline is to "shape capability and experience separately, then merge them." Verifiable RL raises math/code ability, but it can bring verbosity, style drift, and degraded general chat. Rejection sampling and final RL are capability backfilling. A minimal reproduction can use 1K cold-start long-CoT examples, 20K verifiable problems for RL, sample and filter 10K general SFT examples from the RL model, and finally run one round of DPO or GRPO on mixed safety/preference data. Evaluation should simultaneously inspect math/code, ordinary instruction following, refusal behavior, average length, and format stability.

#### 4. DeepSeek-V3.2: from answer verifiers to agentic verifiers.

The direction of V3.2 is to let the model synthesize and complete agentic tasks in tool environments. The training sample here is not a single answer, but an episode with tool calls, environment observations, failure recovery, and final delivery. The reward checks not only final text, but also whether tool calls succeed, whether evidence is found, whether code passes tests, and whether the task is actually completed in the environment. It belongs to the same trend as MiniMax M2.1, UI-TARS-2, and LongCat: RLVR is expanding from math verifiers to software / browser / GUI / tool verifiers.

### Zhipu Z.ai / GLM

> **Sources**: [GLM-4.5][^glm_4_5], [GLM-5][^glm_5]

GLM's main line is ARC: Agentic, Reasoning, and Coding. GLM-4.5 first shows that these three kinds of ability can be jointly optimized in one MoE model; GLM-5 then explains the post-training recipe more clearly: after multi-task SFT, training proceeds through Reasoning RL, Agentic RL, and General RL, while asynchronous RL infrastructure improves the efficiency of long-horizon interaction training.

#### 1. GLM-4.5: hybrid reasoning and expert model iteration.

GLM-4.5 supports two modes: thinking and direct response. The motivation is similar to Qwen3: complex problems need long thinking, while ordinary assistant scenarios cannot unfold verbose CoT every time. During post-training, expert model iteration and RL jointly improve agentic, reasoning, and coding ability. Expert iteration can be understood as "first let specialized strong models produce or filter high-quality data, then feed that data back to train a unified model." RL then further amplifies verifiable ability on math, code, tools, and agent benchmarks.

#### 2. GLM-5: Reasoning RL -> Agentic RL -> General RL.

The public GLM-5 report explicitly describes progressive alignment: first perform multi-task SFT and introduce interleaved thinking modes; then perform reasoning RL; then agentic RL; finally general RL aligns the model with a human style. Reasoning RL mainly handles high-confidence outcome-verifier tasks such as math, logic, and code, first raising long-chain reasoning and self-checking ability. Agentic RL connects the model to multi-turn tool environments, filesystems, codebases, and software-engineering tasks, teaching "observe -> act -> environment feedback -> correct." General RL finally backfills ordinary chat, conciseness, safety, instruction following, and style, reducing the verbosity and capability bias introduced by the first two stages.

#### 3. Asynchronous agent RL: decoupling generation and training.

GLM-5 introduces new asynchronous RL infrastructure and uses slime's customizable rollout interface. Long-horizon agent rollouts vary greatly in duration: some tasks need only one turn, while others run tests, call tools, and wait for environments. Synchronous PPO/GRPO forces the trainer to wait for the slowest episode. GLM-5 decouples rollout generation, environment interaction, verifier branches, and training, so experiences from different tasks can continuously enter the training queue. Slime's server-based rollout execution allows different tasks to define multi-turn loops, tool invocation, environment-feedback handling, and verifier-guided branching without changing the underlying training stack.

#### 4. On-policy cross-stage distillation: preventing forgetting in staged training.

The risk of staged RL is that the abilities learned in Reasoning RL may be washed out during Agentic RL or General RL. GLM-5 uses on-policy cross-stage distillation to preserve previous-stage strengths in later stages: the current policy generates data online, while the previous-stage ability participates in training as a distillation target or filtering signal. A minimal reproduction can run a three-stage experiment: obtain a reasoning model with MATH/code GRPO; obtain an agentic model with SWE-bench Lite or tool-task RL; finally mix in general instructions for DPO/GRPO while using the first-stage model's outputs on math problems for distillation, observing whether math regresses.

### Tencent Hunyuan

> **Sources**: [Hunyuan-T1][^hunyuan_t1], [Hunyuan-A13B][^hunyuan_a13b], [Hunyuan-A13B-Instruct Model Card][^hunyuan_a13b_instruct]

Hunyuan's public materials can be divided into T1's reasoning RL and A13B's fast/slow-thinking instruct model. The A13B technical report discloses fewer details than MiniMax or MiMo, but the T1 page provides several clues about training stability, enough to serve as a reference for reasoning-RL system design.

#### 1. Hunyuan-T1: allocating most post-training compute to RL.

T1 explicitly states that 96.7% of post-training compute is spent on reinforcement learning, with the goal of improving pure reasoning and human preference alignment. Task sources cover world science and reasoning problems such as math, logical reasoning, science, and code, combined with ground-truth feedback. The main reward is still verifiable signal: math answers, logic-problem rules, code execution, standard answers or judges for science problems. Public materials do not give a complete reward formula, but they show that this is not single chat-preference RL; it is reasoning-heavy RL.

#### 2. Curriculum, context expansion, and token efficiency.

T1's training plan uses curriculum learning: gradually increase data difficulty and context length so the model improves reasoning ability while learning to use tokens more effectively. This design is consistent with long-thinking training in MiniMax-M1 / Qwen3: one should not open maximum length, hardest problems, and complex rewards all at the beginning, or early training will be dragged down by noise and overlong outputs. In reproduction, math/code problems can be divided into three difficulty stages: first short CoT plus medium problems, then long CoT plus difficult problems, and finally length/efficiency evaluation.

#### 3. Data replay, periodic policy reset, and unified reward.

T1's public materials mention reference data replay and periodic policy resetting, improving long-term training stability by more than 50%. This shows that Hunyuan is handling policy drift during long RL runs: data replay prevents the model from forgetting early abilities; policy reset pulls the model back to a more stable checkpoint / reference when the policy drifts too far or degenerates. The preference-alignment stage uses self-reward plus reward model: the earlier T1-preview acts as a self-reward evaluator that comprehensively scores outputs, and a reward model is added to guide self-improvement. A reproduction can maintain a replay buffer and mix in high-quality samples from earlier stages; every N steps, evaluate entropy, format errors, average length, and general eval, then roll back or reset the reference if the model degenerates.

#### 4. Hunyuan-A13B: the product form of fast / slow thinking.

The A13B-Instruct model card shows that slow thinking is enabled by default and can also be disabled with `enable_thinking=False`. This means post-training data must contain two response types: slow-thinking trajectories with `<think>`, and direct-answer fast-thinking trajectories. Otherwise the model either thinks slowly every time or lacks reasoning for complex problems. A minimal reproduction can construct two labels for the same batch of prompts: keep thinking processes for complex problems, and provide only short answers for simple QA; after SFT, add preference about "whether thinking is needed" in RL / DPO, and evaluate complex-problem accuracy and average length on simple problems.

### Baidu ERNIE

> **Sources**: [ERNIE 4.5 Technical Report][^ernie_4_5], [ERNIE 5.0][^ernie_5_0]

ERNIE 4.5 discloses post-training in a structured way: LLM post-training is SFT + RL, with Progressive RL and Unified Preference Optimization in the RL stage; VLM post-training is three-stage SFT plus one reasoning-RL stage. ERNIE's value is that it explains the compatibility problem of multi-task, multi-reward, multimodal post-training relatively clearly.

#### 1. SFT: first cover task domains, then enter RL.

ERNIE 4.5 SFT covers general instructions, logic, math, code, professional tasks, safety, and multimodal understanding. The key is not the amount of data, but that every capability domain has an evaluable target. For LLMs, SFT teaches basic response format and task ability; for VLMs, three-stage SFT separately strengthens visual perception, complex visual reasoning, and mixed thinking / non-thinking data. If a multimodal model enters RL directly, visual-recognition errors and reasoning errors are easily entangled, so perception must first be stabilized with SFT.

#### 2. Unified Rewarding System: placing rule, sandbox, RDRM, GRM, and other rewards in one framework.

The ERNIE 4.5 diagram lists components such as rule-based reward, RLLM, sandbox, RDRM, checklist-aware verifier, GRM, and DRM. The problem it solves is heterogeneity of reward sources: math problems may use rule answers; code problems use sandbox execution; open QA uses a generative reward model; safety/checklist tasks use checklist verifiers; preference tasks use a discriminative reward model. Without domain normalization, rewards at different scales suppress each other. ERNIE's unified reward system can be abstracted into three steps: first choose a verifier / RM by task domain; then normalize reward to comparable scales; finally control task weights by training stage.

#### 3. Progressive RL: Logic RL -> Reasoning RL -> General RL.

ERNIE 4.5 divides LLM RL into Stage 1 Logic RL, Stage 2 Reasoning RL, and Stage 3 General RL. Logic RL uses cleaner, more rule-like tasks to stabilize the reasoning format; Reasoning RL expands to math, code, and complex reasoning; General RL backfills ordinary instruction following, human preference, and safety. This sequence is consistent with the "capability first, then generalization" pattern in GLM-5 / Qwen3. A minimal reproduction can organize data in this order: first 2K logic/symbolic problems, then 10K math/code problems, and finally 10K general preference problems; each stage separately evaluates whether earlier-stage abilities have been overwritten.

#### 4. UPO: scale and stability in multi-task RL.

Unified Preference Optimization is motivated by the fact that when reasoning tasks and non-reasoning tasks are mixed, reward format, domain normalization, and informative prompt filtering all affect training. Math/code 0-1 reward, preference scores, and safety scores cannot simply be added. A reproduction idea for UPO is: maintain reward normalization for each task domain; filter prompts with no information; assign domain weights to different reward sources; during training, record each domain's reward mean and variance to avoid one task type dominating updates.

#### 5. ERNIE 5.0: extending post-training to unified multimodality.

ERNIE 5.0 continues toward a unified model for text, image, video, and speech. The biggest difficulty is reward comparability and modality balance: image-understanding reward, video temporal reward, text preference reward, and speech-task reward have completely different error sources. For reproduction, do not simply concatenate multimodal questions into text JSON. Instead, prepare perception eval, reasoning eval, and preference eval for each modality, then perform staged SFT/RL in a unified way.

### StepFun

> **Sources**: [Step3][^step3], [STEP3-VL-10B][^step3_vl_10b], [Step-DeepResearch][^step_deepresearch]

StepFun's public materials cover multimodal reasoning and deep research agents. STEP3-VL-10B shows how a compact 10B VLM approaches larger models through scaled post-training; Step-DeepResearch belongs to research-agent training.

#### 1. STEP3-VL-10B: fully unfrozen pretraining followed by 1K+ RL iterations for visual reasoning.

The motivation of this report is that small models can approach large models on complex multimodal reasoning, but visual-language coordination and post-training must be designed together. The model first undergoes unified, fully unfrozen pretraining on 1.2T multimodal tokens, aligning the perception encoder with a Qwen3-8B decoder; the post-training stage then performs more than 1K iterations of reinforcement learning. The key is that VLM RL does not train only text answers. Visual evidence, text reasoning, and answer generation are jointly constrained by reward.

#### 2. RLVR + RLHF: separate verifiable visual tasks from open preference tasks.

Visual math, OCR followed by calculation, chart reading, multiple-choice questions, and geometry/spatial problems can use RLVR: answers can be checked by rules or by programs / standard answers. Open-ended image description, complex aesthetics, visual safety, and explanation quality are more suitable for RLHF / judge reward. In reproduction, split data into two buckets: the first uses MathVista, ChartQA, OCR-VQA, and geometry problems with exact / numeric verifiers; the second uses a multimodal judge to score helpfulness, faithfulness, detail, and safety. Do not add the two reward types directly; normalize by domain first.

#### 3. PaCoRe: generate visual hypotheses in parallel, then coordinate the answer.

Parallel Coordinated Reasoning aims to scale test-time compute. In multimodal tasks, errors often come from "seeing the image incorrectly" rather than "not knowing how to reason." PaCoRe lets the model explore multiple visual hypotheses or reasoning paths, then synthesize a more reliable answer. In training, this corresponds to two signals: candidate paths should be diverse and evidence-based, and the final integration should be correct and non-hallucinatory. A small reproduction can build a multimodal version of self-consistency: sample multiple evidence chains for the same image, select the correct chain with a verifier / judge, and SFT the model to learn "list candidate visual evidence -> cross-check -> answer."

#### 4. Step-DeepResearch: train the research process, not a report template.

Deep research agents perform search, browsing, evidence extraction, conflict comparison, citation, and long-form organization. The SFT stage should use high-quality research trajectories, teaching the model how to plan queries, read sources, and record evidence. The RL-stage reward should be decomposed into answer correctness, citation existence, evidence support, source coverage, redundant-search penalty, and final-report structure. A reproduction can use 300 multi-source questions, a search API, a browser extractor, and a citation checker: first train trajectory format, then apply episode-level reward to final answers and cited evidence.

### Meituan LongCat

> **Sources**: [LongCat-Flash-Thinking-2601][^longcat_flash]

LongCat-Flash-Thinking-2601 reads like an engineering-system design document for agent RL. Its core is not one reward formula, but environment scaling, reinforcement-learning scaling, noise-robust training, and heavy thinking.

#### 1. Environment scaling: automatically generating solvable tool environments from domain definitions.

LongCat's motivation is that there are too many real agent scenarios, and manually adapting prompts, toolchains, and environment interfaces is extremely expensive. It builds an environment-generation system covering more than 20 domains and tens of thousands of scenarios: given a domain definition, it automatically synthesizes more than 60 tools, database schemas, tool-call interfaces, and verification logic. Covered scenarios include file management, data analysis, e-commerce retail, and telecom service. This design turns "training data" into an "interactive environment graph."

The hardest part of environment generation is consistency. A complex environment may contain dozens of databases and tool-parameter dependencies. If tasks are generated randomly, it is easy to produce tasks that appear solvable but are actually impossible. LongCat uses a solvable-path-first strategy: first randomly sample a long tool-call chain as the golden toolchain; construct the task and database state around this chain; then use controlled BFS to expand the environment subgraph, ensuring that predecessor dependencies for new tools already exist; dynamically add new golden chains according to environment complexity and remaining tools; if there are fewer than 20 tools, supplement a usable medium-sized chain from the global tool library. The reproducible point is to first guarantee at least one successful path, then expand the environment, rather than building an environment first and hoping the task is solvable.

#### 2. Cold-start data: real trajectories and dual-route synthesis.

Before RL, LongCat redefines the pretraining / fine-tuning objective as "providing a cold-start policy for RL." In domains with real data, such as math and coding, high-quality trajectories are selected through quality control and executable verification. In domains lacking real data, such as search and tool use, it uses text-driven synthesis and environment-anchored synthesis. Text-driven synthesis generates trajectories from task descriptions; environment-anchored synthesis generates tasks from existing toolchains and database states, ensuring that tasks can be verified by the environment. A reproduction can first build a small environment with five tools: order lookup, refund, inventory, user information, and logs; sample golden chains first, then ask the model to generate tasks and trajectories.

#### 3. DORA: fully asynchronous streaming RL.

Agent rollouts vary enormously in duration, so synchronous training wastes a large amount of GPU time. DORA supports parallel exploration by multiple model versions, with experiences from different versions collected into the sample queue as soon as they are produced; the trainer does not need to wait for all tasks to finish. Scheduling is split into a lightweight Rollout Manager and multiple Rollout Controllers, where each controller manages virtual rollout groups and handles environment interaction through data parallelism. Environment deployment extends PyTorch RPC to instantiate environments on idle CPU machines.

To adapt to a 560B-parameter MoE, DORA also performs Prefill-Decode decoupling and KV-cache exchange. PD decoupling places long-context prefill and decode on different device groups, preventing prefill from blocking decode in multi-turn interaction. KV-cache is dynamically exchanged through chunk-level aggregation, asynchronous transfer, compute overlap, and CPU residency, reducing repeated computation. Resource allocation uses two-level balancing: overall rollout quotas are adjusted by environment difficulty, while task-domain diversity is maintained inside each batch. The report says this system reaches 2-4 times the efficiency of traditional synchronous training and supports stable training beyond a thousand steps.

#### 4. Noise-robust training: inject real-world disturbances early.

LongCat actively injects tool timeouts, tool errors, missing return fields, inconsistent databases, ambiguous instructions, and requirement changes so the model learns to recover. Reward should not only check final success; it should also reward error detection, replanning, tool switching, and asking the user for clarification. A minimal reproduction can randomly make 10%-30% of tool calls fail or return partial fields, then train the model to retry, change parameters, or use backup chains according to error codes; evaluate the gap between clean success rate and noisy success rate.

#### 5. Heavy Thinking: scaling both width and depth.

LongCat's heavy-thinking mode does not merely stretch one CoT. It first generates multiple reasoning / action paths, then uses a summary model to analyze, select, and integrate them. This is suitable for complex agent tasks because if a single path chooses the wrong tool early, the later trajectory drifts farther and farther away. A small reproduction can sample 3-5 plans for the same tool task, use a verifier / judge to select the best plan or merge plans, and then execute. During training, feed the "candidate paths -> comparison -> final plan" trajectory back into SFT / RL.

### Ant Ling / Ring

> **Sources**: [Ling-1T][^ling_1t], [Ring-1T][^ring_1t]

#### 1. Disclosure boundary: many model releases, few complete recipes.

Ling / Ring public materials focus more on model releases and inference efficiency. They do not unfold a complete post-training pipeline the way DeepSeek-R1, Qwen3, or MiniMax M2.1 do. This section is therefore better read as an "industry signal" than as a directly reproducible training recipe.

#### 2. Learnable point: deep thinking and inference efficiency must be designed together.

Two points can be learned clearly. First, trillion-scale MoE models also treat deep-thinking / insight-style ability as a post-training target, rather than only doing chat alignment. Second, long-sequence reasoning and efficient inference deployment must be considered together.

#### 3. Minimal reproduction: bucket fast / slow thinking data.

For reproduction, mainly borrow the data design for fast/slow thinking: keep long-thinking trajectories for complex math, code, and analysis problems, and keep short answers for ordinary QA. Use preference data to penalize meaningless long thinking, and evaluate both accuracy and token cost.

### Huawei Pangu

> **Sources**: [Pangu Ultra][^pangu_ultra], [Pangu Pro MoE][^pangu_pro_moe], [Pangu open-source news][^pangu_news]

#### 1. Disclosure boundary: more detail on hardware and the open-source ecosystem.

Pangu public information emphasizes Ascend-native training, MoE sparse efficiency, and the open-source model ecosystem. Post-training details are not unfolded the way they are in R1/Qwen/MiniMax.

#### 2. Learnable point: the post-training recipe is constrained by deployment hardware.

The learnable point is the coupling between hardware and training recipe. If the model is to be deployed on Ascend NPUs, post-training cannot consider only the algorithm. It must also consider MoE routing, long-context memory, inference throughput, and the cost of fast/slow thinking.

#### 3. Minimal reproduction: put cost metrics into evaluation.

At the reproduction level, Pangu can be treated as a case of "post-training under engineering constraints": evaluate the same reasoning model on accuracy, number of activated experts, average output length, throughput, and deployment cost.

### 01.AI Yi

> **Sources**: [Yi-Lightning][^yi_lightning]

#### 1. Traditional product-level RLHF route.

Yi-Lightning discloses a traditional product-level LLM post-training line: after pretraining, perform SFT and RLHF, emphasizing multi-stage training, synthetic data construction, reward modeling, and the RAISE safety framework throughout pretraining, post-training, and serving. It does not provide a tool-environment recipe like agent reports, but it is useful for learning how chat models are aligned to human preferences.

#### 2. Minimal reproduction: SFT -> RM -> PPO / DPO.

A reproducible setup can have three stages: use high-quality Chinese/English instructions for SFT; for the same prompt, sample multiple responses and use humans or a judge to rank them, then train a reward model; finally run PPO/DPO and evaluate Chinese, Math, Coding, Hard Prompts, and safety separately.

#### 3. Evaluation reminder: do not look only at static benchmarks.

Yi-Lightning also reminds us that static benchmarks and real human preferences can diverge. Post-training metrics cannot look only at problem sets.

### InternLM / Shanghai AI Lab

> **Sources**: [InternLM2][^internlm2]

#### 1. Engineering traditional RLHF: data governance matters more than the algorithm name.

InternLM2 is an important reference for the open-source community to understand traditional RLHF engineering. Its focus is not long-CoT RLVR, but data governance, SFT, reward modeling, and online RLHF.

#### 2. COOL: conditional preference, avoiding an average-person style.

The motivation of COOL, Conditional Online RLHF, is that preference optimization makes the model drift across task domains: some users prefer concision, some tasks need detail, and safety scenarios require conservatism. Conditional training lets the model adjust the optimization target according to task conditions, preference conditions, or data domains, rather than compressing all preferences into one average person.

#### 3. Minimal reproduction: add domain / style / safety conditions to preference data.

A minimal reproduction can work as follows: annotate each preference example with domain / style / safety conditions; use the condition as an input when training the reward model; during online RLHF, sample prompts and rewards by condition; evaluate helpfulness, harmlessness, verbosity, and Chinese ability by domain.

**Learning point**: even without an executable verifier, preference RL should use data bucketing and condition control. Otherwise the model easily collapses toward a single style.

### Baichuan and 360 Zhinao

> **Sources**: [Baichuan 2][^baichuan2], [360Zhinao][^zhinao]

#### 1. Baichuan2: SFT -> RM -> PPO in the Chinese open-source context.

Baichuan2 is one of the earlier Chinese reports to disclose the classic SFT -> RM -> PPO alignment process. In the SFT stage, the base model first learns dialogue and instruction following; in the RM stage, preference comparisons are collected to train a reward model; in the PPO stage, the policy is optimized with RM scores and a KL constraint. It is suitable for this course as a Chinese/open-source counterpart to the InstructGPT route: when large-scale verifiable RLVR is absent, SFT/RM/PPO remains a complete post-training loop.

#### 2. 360Zhinao: RM is also a data-governance tool.

360Zhinao public materials emphasize data quality and data governance. RM is not only a rewarder for PPO. It can also serve as a judge, filter, and relabeling tool: score candidate answers, filter out low-quality samples, discover repetitive patterns, and feed the result back into SFT.

#### 3. Minimal reproduction: rejection-sampling SFT + DPO.

A reproducible experiment can sample 4 answers for the same batch of Chinese instructions, score them with a judge/RM, keep the top-1 for rejection-sampling SFT, and use bottom/top pairs for DPO. This workflow is less flashy than agent RL, but it is very close to the daily post-training practice of many real product models.

### Skywork and Xiaomi MiMo

> **Sources**: [Skywork-OR1][^skywork_or1], [MiMo][^mimo], [MiMo-VL-Miloco][^mimo_vl]

Skywork-OR1 and MiMo are both useful for learning the problem of "continuing RL on small / distilled models." Instead of merely piling on frontier-lab scale, they focus on entropy collapse, data difficulty, sparse reward, and training stability.

#### 1. Skywork-OR1: continuing RL on R1-Distill, where the core risk is entropy collapse.

Skywork-OR1 is built on the DeepSeek-R1-Distill series. The distilled model already knows long CoT, but during continued RL it can easily converge too early to a few expressions and solution patterns; once entropy drops, exploration disappears. The report's main line is to identify factors affecting entropy dynamics through training pipelines and ablations, and to show that mitigating premature entropy collapse is critical for test performance. Public results show average accuracy improving from 57.8% to 72.8% for 32B, and from 43.6% to 57.5% for 7B, with weights, code, and data open-sourced.

The reproduction focus is monitoring entropy, not only reward. Use R1-Distill-7B for math/code RL; record token entropy, response length, pass@1, repeated n-grams, and format error rate at every step; try adjusting sampling temperature, KL, clip, data difficulty, and dynamic sampling. If reward rises while entropy collapses quickly, late-stage generalization is often poor.

#### 2. MiMo: the key to 7B reasoning post-training is 130K verifiable problems.

MiMo-7B constructs 130K verifiable mathematics and programming problems for RL during post-training. Math problems use an answer verifier; programming problems use test execution. It also proposes test-difficulty-driven code reward to reduce reward sparsity in code: not all passed/failed tests are equivalent, and passing harder tests or more hidden tests should provide finer-grained reward. Strategic data resampling stabilizes training by concentrating compute on samples that are both challenging and learnable.

MiMo's minimal reproduction is clear: prepare 80K math problems and 50K programming problems, or a smaller 5K/2K version; use Math-Verify / parsers to judge math answers; prepare easy/medium/hard tests for each code problem and weight reward by test difficulty; after each RL round, count which problems are all-correct, all-wrong, or partially correct, downsample all-correct/all-wrong problems, and raise sampling for partially correct problems. This method is especially important for 7B models, because small models have limited training budgets and cannot waste rollouts on samples with no learning signal.

#### 3. MiMo-VL-Miloco: extending small-model reasoning to multimodality.

MiMo-VL continues the route of "small model + high-quality verifiable data + stable RL," but the object becomes vision-language. The learnable point is similar to STEP3-VL: visual tasks must distinguish perception errors from reasoning errors; reward needs to cover answer correctness, visual-evidence citation, and output format at the same time. In reproduction, math charts / OCR / geometry problems can be used as RLVR data, then open image-description preference data can be mixed in for backfilling.

### Kuaishou, SenseTime, iFlytek

> **Sources**: [Kwai Keye-VL][^keye_vl], [SenseNova U1][^sensenova_u1], [Spark X1][^spark_x1]

#### Disclosure boundary: useful as industry dynamics.

These three companies have disclosed developments in multimodal post-training (Kuaishou), native understanding and generation (SenseTime), and deep reasoning (iFlytek), but they lack complete training-recipe reports. In this course, they can be kept as "directional coverage" to avoid mistakenly writing them as detailed reproducible training recipes.

**Reading focus**: look at which capability surfaces they emphasize, such as VLM, multimodal generation, deep reasoning, Chinese scenarios, and end-to-end product experience. Do not infer undisclosed SFT/RL details from release materials.

---

## International Companies and Major Labs

### OpenAI

> **Sources**: [InstructGPT][^instructgpt], [GPT-4][^gpt4], [o1][^o1], [o3/o4-mini][^o3_o4_mini], [o3 Operator][^o3_operator], [GPT-4.5][^gpt4_5], [GPT-5][^gpt5], [GPT-5.1][^gpt5_1], [GPT-5.4 Thinking][^gpt5_4], [GPT-5.5][^gpt5_5], [GPT-5.5 Instant][^gpt5_5_instant], [GPT-5-Codex][^gpt5_codex], [GPT-5.1-Codex-Max][^gpt5_1_codex_max], [GPT-5.2-Codex][^gpt5_2_codex]

OpenAI's public materials span three generations of post-training: classic RLHF in InstructGPT, reasoning / deliberation in the o-series, deliberative alignment in safety system cards, and agent models such as Codex / Operator. Closed-source system cards do not disclose the complete recipe, but the methodological boundary is clear.

#### 1. InstructGPT: the minimal RLHF loop.

The InstructGPT process can be directly reproduced as a teaching experiment. The first step is demonstration SFT: labelers write high-quality answers so the base model first learns to follow instructions. The second step is reward modeling: for the same prompt, sample multiple answers, ask labelers to rank them, and train a reward model to predict human preference. The third step is PPO: use the reward model to score policy outputs and use a KL penalty to keep the policy from drifting too far from the SFT model. The key point is that the three data types are different: SFT data is "good answers," RM data is "preference comparisons," and PPO data is "prompt + on-policy samples."

A minimal reproduction can use 5K instructions for SFT; for 1K prompts, sample 4 answers each and train an RM with pairwise preferences; finally choose PPO / DPO / IPO for preference optimization. Evaluation cannot look only at reward-model score. Human or LLM judges must inspect helpfulness, truthfulness, toxicity, verbosity, and instruction following, because RM is easily exploited by the policy.

#### 2. GPT-4 to o-series: reasoning post-training expands the action space.

The GPT-4 technical report describes post-training and safety only at a high level; o1/o3/o4-mini system cards are clearer: the model learns through reinforcement learning to deliberate longer before answering and to use tools when needed. The change is that the action is no longer only "next token." It also includes when to write code, when to browse, when to call image/file tools, when to stop, and when to refuse. Reward expands from human preference to final answer, tool result, policy compliance, safety boundaries, and user experience.

The reproducible abstraction of this ability is: choose a task family with tools, such as math problems with code execution; the model outputs thinking and tool calls; the environment returns execution results; reward checks final answer, tool format, number of calls, and safety policy at the same time. First SFT a small number of successful tool trajectories, then run RL. This reproduces the shape of the o-series method without pretending to replicate a closed recipe.

#### 3. Deliberative alignment: safety also becomes a reasoning task.

One recurring direction in OpenAI system cards is to make the model reason about policy before deciding how to answer difficult safety questions. Early safety alignment can degenerate into refusal templates. Deliberative alignment is closer to turning policy specs, boundary cases, and safety evaluations into training tasks: the model must identify the request type, judge whether it can be completed safely, and if necessary transform it into a safe alternative. A reproduction can construct safety prompts, policy clauses, and examples of correct handling; use SFT to teach the model to reference policy; then use preference/RL to reward "safe completion" rather than blind refusal.

#### 4. Operator / Codex: agent post-training needs real environments.

Operator and Codex-style models extend post-training to browser / software-engineering episodes. A coding-agent environment must contain repository state, test commands, patch verifier, lint, user-instruction hierarchy, and failure recovery. A browser-agent environment must contain page state, clickable elements, task-success checks, and a safety sandbox. The GPT-5-Codex system card explicitly says it is trained with RL on real software-engineering tasks, learning to match human code style and PR preferences, strictly follow instructions, and repeatedly run tests until they pass. GPT-5.1-Codex-Max further extends training to long-horizon agentic coding across multiple context windows, using compaction to maintain coherence over million-token tasks. GPT-5.2-Codex emphasizes SWE-Bench Pro, Terminal-Bench 2.0, native Windows environments, long-context understanding, and reliable tool use. A minimal reproduction can use SWE-bench Lite: check out the repository, provide the issue, let the model edit files and run tests, and set reward to test pass plus patch reasonableness. It can also use MiniWoB / BrowserGym: the model observes DOM/screenshots, clicks and types, and receives reward for task completion and action legality.

### Anthropic

> **Sources**: [Constitutional AI][^constitutional_ai], [Anthropic CAI overview][^anthropic_cai], [Claude 4 System Card][^claude4], [Claude Sonnet 4.5][^claude_sonnet_4_5], [Claude Opus 4.5][^claude_opus_4_5], [Claude Opus 4.6][^claude_opus_4_6]

The most valuable things to learn from Anthropic are Constitutional AI and systematic safety evaluation. Its public materials do not provide the full Claude 4 training recipe, but Constitutional AI is a reproducible method.

#### 1. Constitutional AI: write safety preferences as principles, then scale data with AI feedback.

Traditional RLHF requires many human comparisons. Constitutional AI first defines a set of principles, the constitution. In the supervised phase, the model first generates an answer, then critiques and rewrites itself according to the constitution, forming safer SFT samples. In the preference phase, AI compares two answers according to the constitution, generating preference data; a preference model is trained, and the policy is finally optimized with RL. This is RLAIF: humans move partly from per-example preference judgments to principle design and quality auditing.

A minimal reproduction path is: write 20-50 safety / honesty / privacy / harmlessness principles; sample answers for risky prompts; ask a strong model to point out problems and rewrite according to the principles; use rewritten data for SFT; then ask the strong model to rank two answers by the principles and train DPO or a reward model. Evaluation must separately inspect over-refusal, because overly strong safety principles can make the model refuse normal requests.

#### 2. Claude system cards: post-training and evaluation are one system.

Claude 4 series system cards focus on reward hacking, sabotage, sycophancy, alignment faking, hidden objectives, jailbreaks, and policy following under extended thinking. The learning point is not one RL formula, but that safety post-training must include adversarial evaluation. Good behavior under the training reward does not prove that the model will not deviate in long contexts, tool calls, role play, or high-pressure prompts.

#### 3. Safety risks of extended thinking.

When a model has longer thinking and tool ability, safety training is no longer just "output a refusal." The model may formulate circumvention strategies in its reasoning, or complete forbidden steps in a tool environment. Therefore safety reward must cover policy compliance, tool restrictions, information leakage, privacy, deception, and refusal quality. A reproduction can combine tool tasks with safety rules: for example, ask the model to process files while forbidding it from reading unrelated sensitive files; reward checks both task success and unauthorized behavior.

### Google DeepMind

> **Sources**: [Gemini 1.5][^gemini_1_5], [Gemini 2.5][^gemini_2_5], [Gemini 2.5 Deep Think][^gemini_2_5_deep_think], [Gemini 2.5 Computer Use][^gemini_2_5_computer_use], [Gemini 3.1 Pro][^gemini_3_1_pro], [Gemma 3][^gemma_3]

Google DeepMind's public disclosures are less granular than open papers, but the direction is very clear: multimodality, long context, tools, reasoning, and safety evaluation are trained together. The Gemini / Gemma line is useful for learning how to design post-training tasks for unified multimodal models.

#### 1. Gemini 1.5 / 2.5: the core of long-context post-training is evidence grounding.

A long-context model is not just a model with a larger context window. Post-training must teach it to locate evidence across hundreds of thousands of tokens, images, videos, and documents, and to avoid mixing irrelevant passages into the answer. Task construction should include needle-in-a-haystack, long-document QA, multi-document conflict, video event localization, and cross-modal citation. Reward cannot check only the final answer. It must also check whether evidence locations are correct, whether citations support conclusions, and whether irrelevant distractors were ignored.

#### 2. Deep Think: lateral exploration plus aggregation, not infinitely extending one CoT.

Gemini 2.5 Deep Think shows another form of test-time compute scaling: generate multiple candidate lines of thought, compare them, and integrate. It belongs to the same family as LongCat heavy thinking and self-consistency. Training requires reward to distinguish "useful diversity" from "meaningless divergence": candidate paths should cover different hypotheses, and the integrated answer should be more correct than a single path. A small reproduction can sample 5 reasoning traces for math/visual problems, use a verifier to select correct paths, and then train the model to output "candidate analysis + final merged answer."

#### 3. Computer Use: safety-aware action learning in GUI environments.

Gemini Computer Use targets screen states and action sequences: observe a webpage/desktop, output actions such as click, type, and scroll, then continue according to environment feedback. Reward should at least include task completion, action validity, number of turns, whether sensitive controls were clicked by mistake, whether information was leaked, and whether user authorization was violated. For reproduction, use BrowserGym / OSWorld: each task provides reset, observe, step, success check, and safety check; first SFT successful trajectories, then use RL to learn long-horizon strategies and error recovery.

#### 4. Gemma: distillation plus targeted post-training for open small models.

The Gemma series provides a route closer to the open-source community: use strong-teacher distillation and high-quality data filtering to improve small models, then conduct targeted post-training for math, instructions, multilingual ability, and safety. The point is that frontier-scale RL systems are not the only path. Small models can also gain practical ability from data quality, teacher choice, capability bucketing, and targeted preference optimization.

### Meta Llama

> **Sources**: [The Llama 3 Herd of Models][^llama3_herd]

Llama 3 Herd is one of the best open-model references for a product-level chat-model post-training pipeline. Its value is not one standalone algorithm, but a complete loop of data governance, SFT, reward model, rejection sampling, preference optimization, safety alignment, and evaluation.

#### 1. SFT data is mixed by capability domain.

Llama SFT should not be understood as "piling up instruction JSON." Data must cover general QA, code, math, multilingual ability, tools, safety, and long context, with independent evals for each domain. Engineering steps usually include deduplication, quality filtering, format unification, refusal-boundary cleaning, and control of overly long/short samples. For reproduction, first use small-scale capability buckets rather than one mixed data pool.

#### 2. Rejection sampling: using RM / rules to turn sampling into new SFT data.

For the same prompt, sample multiple answers and use a reward model, rule verifier, or judge to select the best, then add it to the next SFT round. This sits between SFT and RL: it does not directly apply policy gradients, but it distills the model's own high-quality outputs back into the model. Math/code can use verifiers; chat/safety can use RM / judges. In a minimal reproduction, sample 4-8 answers per prompt, keep the top-1, and also keep top/bottom pairs for DPO.

#### 3. Preference optimization and safety run through the whole process.

Llama safety is not a final refusal-data add-on. It appears continuously in data filtering, SFT, safety RM, red teaming, and release thresholds. Preference optimization further separates the probability of good and bad answers, but it may also sacrifice diversity and honesty, so truthfulness, safety, refusal, and helpfulness must be evaluated together. This is a foundation line suitable for open-source teams: even without an agent environment, SFT, RS, DPO/RLHF, and safety evaluation can be made complete.

### Microsoft Phi

> **Sources**: [Phi-4][^phi_4], [Phi-4-reasoning][^phi_4_reasoning]

Phi-4-reasoning focuses on small-model reasoning. It does not rely on huge parameter counts alone, but uses high-quality synthetic data, teachable prompts, and a short segment of outcome-based RL to push a 14B-class model to strong reasoning ability.

#### 1. Data first: teachable prompts matter more than large messy data.

Small models have limited capacity, so post-training data must be learnable, clean, and clearly supervised. Math, science, code, and logic problems should be organized by difficulty. Overly hard problems are all wrong and provide no RL signal; overly easy problems are all correct and waste rollouts. In reproduction, first construct a 5K-20K teachable problem set, ensuring that the model reaches a certain success rate after SFT before entering RL.

#### 2. SFT teaches format and reasoning; short RL corrects accuracy and length.

The Phi-4-reasoning idea can be abstracted as: first use high-quality synthetic reasoning traces for SFT so the model learns to unfold reasoning; then use outcome reward on verifiable problems for RL, reinforcing correct paths while controlling useless long thinking. Small models especially need monitoring of average response length, because a little RL can make outputs longer without improving accuracy. A minimal experiment uses Phi/Qwen 7B-14B, subsets of MATH/GPQA, and strong-teacher-generated CoT; after SFT, sample 8 responses per problem, run GRPO with an answer verifier, and add length statistics.

### NVIDIA Nemotron

> **Sources**: [Nemotron-4 340B][^nemotron_4], [Llama-Nemotron][^llama_nemotron], [Llama Nemotron Ultra][^nemotron_ultra], [Nemotron Agent Blog][^nemotron_agents], [Nemotron-H][^nemotron_h], [Nemotron 3][^nemotron_3]

NVIDIA Nemotron's feature is that post-training becomes reusable assets: models, data, rewards, and deployment stack are released or productized together. Nemotron-4 340B comes with synthetic data, preference data, and a reward model; Llama Nemotron places reasoning, tool use, RAG, instruction following, and enterprise deployment together.

#### 1. Nemotron-4: assetizing alignment.

It does not only release instruct weights. It also treats synthetic data, preference data, reward models, and evaluation components as training assets. The method is to use strong models and rules to generate candidate data, train RM through quality filtering and preference annotation, and then perform RLHF / preference optimization. In reproduction, maintain the RM as an independent artifact: it is used not only for PPO/DPO, but also for rejection sampling, data filtering, and automatic evaluation.

#### 2. Llama Nemotron: prune/distill, then post-train reasoning and agent ability.

NVIDIA public blogs describe a three-stage process: start from a Llama base, first prune for efficiency, then distill to improve ability, and finally use post-training data and RL to strengthen reasoning, instruction following, function calling, and chat. The Llama-Nemotron-Post-Training Dataset covers math, coding, general reasoning, and instruction following; OpenCodeReasoning and related data strengthen code reasoning. Ultra also supports reasoning on/off, showing that it must handle the cost of long thinking and ordinary interaction experience.

#### 3. RLVR and enterprise agents.

NVIDIA emphasizes that distillation can move teacher ability, but further improvement requires curriculum-driven RLVR. Reward in enterprise-agent scenarios comes from tool-call correctness, RAG faithfulness, function-calling schema, code execution, and user-intent alignment. Public materials also mention using REINFORCE and heuristic-based verifiers to strengthen instruction following / function calling, followed by RLHF with preference data such as HelpSteer2. A reproduction can build two RL buckets: one math/code verifier bucket, and one function-calling verifier bucket; finally mix in chat/RAG preferences for backfilling.

#### 4. Deployment constraints enter the post-training objective.

Nemotron outputs not only weights, but also NIM, NeMo Gym, and an enterprise inference stack. Post-training evaluation should include latency, throughput, function-calling success rate, RAG citation faithfulness, and reasoning overhead. If an enterprise model looks only at AIME, it misses the most common failure points in production.

### Mistral

> **Sources**: [Magistral][^magistral]

Magistral's public summary is worth including in a reasoning-RL chapter: Mistral explicitly says it uses its own scalable RL pipeline, does not rely on existing implementations or RL traces distilled from other models, and instead performs pure RL from the ground up.

#### 1. Pure RL: avoid treating teacher traces as the capability ceiling.

Distillation can quickly provide a long-CoT format, but it also inherits the teacher's style and mistakes. Magistral's direction is to start from its own checkpoint and let RL explore reasoning ability itself. Public materials also mention that Magistral Medium is based on Mistral Medium 3 and trains reasoning only with RL, while Magistral Small includes cold-start data from Medium. This corresponds to two reproduction routes: a large model explores directly with RL; a small model first distills cold-start data from the large model, then runs RL.

#### 2. Forcing the reasoning language.

Multilingual reasoning RL can produce mixed reasoning languages. Magistral mentions a simple method to force reasoning language, which shows that post-training must manage not only answer correctness but also reasoning language and output style. In reproduction, explicitly specify reasoning language in the prompt / template, and use format reward to check the language and structure of the reasoning segment and answer segment.

#### 3. The effect of text RL on other abilities.

One interesting conclusion from Magistral is that RL on text data alone can preserve or improve multimodal understanding, instruction following, and function calling. This shows that RL does not necessarily destroy general ability, but continuous evaluation is required. In a minimal reproduction, after math/text RL, also run function calling, ordinary instruction, multilingual, and vision-textualized tasks to confirm the model has not been biased too far by reasoning data.

### Apple

> **Sources**: [Apple Foundation Models 2024][^apple_fm], [Apple Foundation Models 2025][^apple_fm_2025]

Apple's foundation-model reports tie post-training tightly to deployment constraints: an on-device model of about 3B parameters must run on Apple silicon, the server model must serve through Private Cloud Compute, and the model must support multilingual, multimodal, and tool-call use.

#### 1. SFT + RL are performed on an asynchronous platform, but the objective is constrained by on-device deployment.

The 2025 report explicitly says that after training on large-scale multilingual, multimodal, synthetic, and licensed data, models are further optimized by supervised fine-tuning and reinforcement learning on a new asynchronous platform. The post-training target is not simply "maximize benchmark score." It also involves guided generation, constrained tool calling, LoRA adapter fine-tuning, privacy, and low latency. On-device models in particular cannot improve the experience by relying on infinitely long CoT. Reward should include correctness, concision, latency, memory, and energy.

#### 2. Multi-source rewards: preference, rules, and tool constraints coexist.

Apple reports emphasize Responsible AI and locale-specific evaluation. Consumer models must handle different regional languages, safety norms, and product experiences. A reproduction can construct three reward types: text/image-text preference RM for answer quality; rule verifiers for math/STEM reasoning; and tool-calling schema checkers for constrained tool calls. Different objectives can then be set by device type: the on-device model emphasizes concision and privacy, while the cloud model can carry more complex reasoning.

#### 3. Post-training and system interfaces are designed together.

Apple's Foundation Models framework exposes guided generation, constrained tool calling, and LoRA. It shows that product-level post-training cannot be separated from the API: if the interface supports constrained decoding, training should include JSON/schema/tool data; if LoRA personalization is allowed, the base model's post-training should preserve adaptability. A minimal reproduction can train tool-calling JSON schema on a small model and evaluate schema success rate with a constrained decoder.

### xAI Grok

> **Sources**: [Grok-1][^grok_1], [Grok 4][^grok_4], [Grok 4.1][^grok_4_1], [Grok 4.1 Model Card][^grok_4_1_card]

#### 1. Disclosure boundary: many model cards, few training recipes.

xAI public materials focus more on model cards and release notes, without providing a complete post-training recipe. They repeatedly emphasize RL scaling, truthfulness, personality, style, and emotional intelligence. These goals are not traditional "problem-set accuracy," but they are crucial for consumer products.

#### 2. Learnable point: decompose product personality into rewards.

The learnable point is to decompose product personality into evaluable rewards instead of only writing a system prompt. A personality reward can evaluate humor, directness, and avoiding excessive flattery; a truthfulness reward checks facts and expressions of uncertainty; an emotional-intelligence reward checks whether the model recognizes user emotion and responds in an appropriate tone; a safety reward checks risk boundaries.

#### 3. Minimal reproduction: multi-objective preference optimization.

A reproduction can train multiple reward heads or multiple judge rubrics using preference data, then run multi-objective DPO/RL. The risk is that personality reward may promote sycophancy, so a separate evaluation must check whether the model agrees with false claims just to please the user.

### IBM Granite

> **Sources**: [Granite 3.3][^granite_3_3], [Granite 4.0][^granite_4_0], [Granite 4.1][^granite_4_1]

#### 1. Enterprise small models: RAG, tools, safety, and low-cost inference.

IBM Granite's post-training focus is enterprise small models: RAG, tool calling, safety, low-cost inference, and switchable thinking. Public materials for Granite 3.3/4.x show that reasoning post-training such as GRPO/TPO has entered enterprise small models, not only frontier large models.

#### 2. Minimal reproduction: post-training a small model for enterprise tasks.

Granite's route can be abstracted as "small-model post-training for enterprise tasks": first use enterprise QA, RAG citation, tool schemas, and math/logic problems for SFT; then use GRPO to reinforce verifiable reasoning; use preference data to optimize RAG faithfulness and refusal; finally use model merging or adapter merging to combine domain experts.

**Evaluation metrics** should include RAG citation faithfulness, function-calling success, refusal correctness, latency, cost, and the difference between thinking on/off.

### Salesforce xLAM / SFR-RL

> **Sources**: [Salesforce xLAM][^xlam], [Salesforce SFR-RL][^sfr_rl]

Salesforce's xLAM / SFR-RL represents tool calling and agentic RL infrastructure. xLAM focuses on the action model: given user intent and API documentation, the model must choose the correct tool, fill the correct arguments, and call tools in the right order. SFR-RL answers how large-scale agent rollouts can be trained efficiently.

#### 1. xLAM: tool-calling reward is more structured than text preference.

API-agent errors are usually not "the answer sounds bad," but wrong tool selection, missing arguments, incorrect call order, invalid schema, or failure to use the result. Training data should include API schema, user request, tool-call sequence, environment return, and final answer. Reward can be decomposed into schema validity, tool-selection accuracy, argument exact match, execution success, and final-answer groundedness. A minimal reproduction can use 50 mock APIs, automatically generate user requests and correct call chains, and use function execution results as verifier.

#### 2. SFR-RL: pipelined synchronous RL.

Agentic rollouts are long and unstable. Pure synchrony waits for slow tasks; pure asynchrony sacrifices on-policy quality. SFR-RL uses pipelined synchronous training: rollout phase and training phase alternate, and each phase uses the whole GPU cluster. During rollout, the training model is unloaded, the policy is loaded into an elastic inference engine, and generation runs concurrently. During training, the inference engine is released and the training model is reloaded for on-policy updates. Cross-batch pipelining keeps GPUs from idling while preserving data composition and on-policy guarantees.

#### 3. Failure recovery and local-first tool execution.

In long-horizon agent rollouts, one inference-engine crash or stuck tool can block a batch. SFR-RL's inference gateway automatically detects failures, rebuilds engine actors, restores weights, and reschedules in-flight work. It also emphasizes scalable local-first tool execution and Expert Parallelism support. Even on a small cluster, reproduction should implement timeout, retry, task rescheduling, and failure marking. Otherwise agent-RL data is polluted by system errors.

### Amazon Nova

> **Sources**: [Amazon Nova][^nova], [Nova Family Technical Report][^nova_report], [Nova Premier][^nova_premier], [Nova Forge][^nova_forge]

#### 1. Nova Forge: platformizing post-training.

Amazon Nova's technical report is closer to a model card, and the internal post-training recipe is not expanded to paper-level detail. Nova Forge more directly shows the direction of "post-training as a platform." Traditional fine-tuning means an enterprise provides a batch of data and the model performs SFT. Nova Forge allows enterprises to enter from pretraining, mid-training, or post-training checkpoints, mix private data with Nova-curated data, and then align enterprise tasks through an RL stage.

#### 2. Remote reward functions: connecting enterprise verifiers to RL.

The most important method is remote reward functions. Enterprise rewards are often not in the training set, but in internal systems: whether code passes private CI, whether robot actions pass simulation, whether a customer-service answer follows the business process, whether a tool call succeeds in a real API. Nova Forge connects these systems to RL through APIs and scores model rollout results. This pattern can be called Reward as a Service.

#### 3. Minimal reproduction: simulate enterprise reward with a local verifier.

A minimal reproduction can simulate enterprise reward with a private API verifier: the model generates SQL, a code patch, or a customer-service action; a local service executes it and returns pass/fail plus rubric scores; the trainer calls reward only through HTTP. This reproduces Nova Forge's key abstraction: the model provider supplies checkpoints, training infrastructure, and a reward interface, while the enterprise supplies private environments and verifiers.

### Cohere Command A

> **Sources**: [Cohere Research][^cohere_research], [Command A][^command_a]

#### 1. Decentralized pipeline: avoid serially overwriting all abilities.

The Command A report shows how enterprise models can avoid "all abilities being trained serially and overwriting each other." Its post-training is not one line from start to finish, but a decentralized pipeline. First train a core model for basic instruction following, then train expert tracks for code, safety, RAG, math, multilingual, long-context, and other abilities. Each expert track can use its own data recipe, preference objective, and evaluation standard.

#### 2. Expert soup: merge specialized abilities after training.

Next, parameter merging aggregates expert abilities. The report contains SFT Expert Models, SFT Soup Model, RL Expert Models, RL Soup Model, and Polished Model. Six expert tracks can serve long-context, safety, instruction, RAG & agents, multilingual, code/reasoning, and other abilities. RL experts use pairwise comparisons or verifiable rewards. After merging, the model is polished: the RL Soup model first receives best-of-N supervised training, then alternates between offline preference and online RL until human-preference performance plateaus.

#### 3. Minimal reproduction: 3-6 expert tracks + soup + polish.

Command A's reproducible template is: train a base instruct model; copy it into 3-6 experts, each optimized with different data and DPO/RLVR; merge with model soup / task arithmetic; finally polish with a small amount of general preference data. This approach is especially suitable for enterprise models, because safety, RAG, code, and sales copy often have conflicting optimization targets. Training experts separately reduces fights between losses.

### Databricks, AI21, Cursor, LG, NAVER, AI2 Tulu 3

> **Sources**: [DBRX Instruct][^dbrx], [Jamba 1.5a][^jamba_1_5a], [Jamba 1.5a Whitepaper][^jamba_whitepaper], [Cursor Composer 2][^cursor_composer_2], [EXAONE 4.0][^exaone_4_0], [K-EXAONE][^k_exaone], [HyperCLOVA X][^hyperclova_x], [HyperCLOVA X THINK][^hyperclova_x_think], [Tulu 3][^tulu_3], [Tulu 3 Blog][^tulu_3_blog], [RL Post-Training Survey][^rl_survey]

This group of sources does not need a large separate section for every system, but together they fill several important practices.

#### 1. Databricks DBRX: the baseline for enterprise open-source instruct.

DBRX Instruct represents traditional enterprise instruct models: it emphasizes data quality, instruction following, code, RAG, and deployment efficiency. The lesson is to bind post-training to enterprise-scenario evals rather than only chat leaderboards.

#### 2. AI21 Jamba 1.5a: post-post-training safety alignment.

Jamba 1.5a's topic is writing an enterprise code of conduct into the model. Methodologically it resembles a second alignment stage: after an existing instruct model, synthetic safety preference data and enterprise principles are used to correct behavior. A reproduction can give the model a set of company policies, generate pairs of policy-violating and policy-compliant answers, and use DPO or RLAIF to adjust it.

#### 3. Cursor Composer 2: coding-agent training should use real repository tasks.

The target of Cursor's coding agent is not writing a single-file function. It is understanding context inside a codebase, editing, making multi-file changes, running tests, and recovering from failures. The training environment should contain repository state, issue, editor actions, terminal, tests, and patch verifier. It belongs to the same class as GPT-5-Codex, Qwen3-Coder, and MiniMax SWE Scaling.

#### 4. LG EXAONE / NAVER HyperCLOVA X THINK: localization and thinking-mode fusion.

Materials from major Korean companies remind us of an easily overlooked problem: post-training is not only English benchmarks. It must handle local languages, culture, safety norms, and business style. Thinking / non-thinking modes must also be evaluated separately in local languages; one cannot assume that an English CoT recipe transfers directly.

#### 5. AI2 Tulu 3: the open post-training textbook.

Tulu 3 fully open-sources data, code, and training recipes, with the theme of multi-stage post-training: SFT, preference learning, and RLVR. Its value is transparency. One can see how prompt data, preference data, verifiable rewards, training parameters, and evaluation are organized. When reproducing modern post-training, Tulu 3 should serve as the open baseline, and specific techniques from MiniMax/Qwen/DeepSeek/Seed can then be added.

---

## Methodological Main Lines

1. **Reward has moved from "which answer do humans prefer?" to "was the task process truly completed?"** Early RLHF looked at preference pairs; R1, Qwen, Seed, and Mistral look at answer verifiability; MiniMax, Kimi, LongCat, and Tongyi look at tool trajectories, environment state, and final delivery.
2. **Data has moved from static samples to generatable, verifiable, replayable environments.** GitHub PRs, Docker, Playwright, browsers, databases, tool graphs, and searched webpages all become part of post-training data.
3. **Post-training order is increasingly staged.** The common order is cold-start SFT, reasoning RL, agentic RL, and general preference / safety backfilling. If the order is wrong, the model easily develops excessive long CoT, degraded chat, tool abuse, or safety drift.
4. **Training systems are becoming a source of competitiveness.** Asynchronous rollout, PD decoupling, KV-cache exchange, environment scheduling, failure recovery, reward services, LLM-as-judge, and executable verifiers are part of "post-training practice," not peripheral engineering.

If the company practices above are abstracted into a small reproducible project, it can follow this sequence:

1. **First choose a verifiable task family.** Math is simplest, code is next, and web/GUI/research agents are hardest. Reward is difficult to define for open chat tasks, so they are not suitable for a first RL experiment.
2. **Wrap the task as an environment.** A math environment needs an answer parser and verifier; a code environment needs repository checkout, dependency installation, test commands, and patch checks; a web environment needs a browser, state recording, and evidence extraction; a GUI environment needs screenshots, an action space, and a resettable sandbox.
3. **Do SFT cold start first.** Collect or generate successful trajectories so the model learns output format, tool protocol, thinking structure, and stopping conditions. Without a cold start, direct RL easily fails first on format and tool calls.
4. **Then sample and filter.** For each prompt, sample multiple outputs and use a verifier / judge / reward model to select trajectories that are correct, concise, and process-reasonable. This stage is the rejection sampling / self-improvement that Qwen, DeepSeek, Kimi, and MiniMax all perform repeatedly.
5. **Finally run RL.** Simple tasks can use GRPO / DAPO-style within-group relative advantages; tasks that need a value model can refer to PPO / VAPO; long-horizon agents additionally require asynchronous rollout, failure recovery, tool noise handling, and token-level credit assignment.
6. **Backfill capability after training.** Run another alignment pass with general instructions, safety, short answers, style, and non-thinking-mode data, so reasoning RL does not make the model long and slow.

A very small but complete exercise is: use 5K math problems or 1K code-fix problems for SFT, sample 8 candidates, filter with a rule verifier, run one round of GRPO, and finally evaluate accuracy, average output length, format error rate, and general-chat regression. Such a closed loop reveals the real difficulties of post-training much better than stopping at algorithm names.

## References

### Chinese Companies and Labs

#### MiniMax

[^minimax_m2_1]: [MiniMax M2.1: Post-Training Experience and Insights for Agent Models](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)

[^minimax_m1]: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)

[^minimax_webexplorer]: [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501)

#### Alibaba Qwen / Tongyi

[^qwen2_5]: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

[^qwen2_5_math]: [Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement](https://arxiv.org/abs/2409.12122)

[^qwq_32b]: [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)

[^qwen3]: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

[^qwen3_coder]: [Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)

[^qwen3_coder_next]: [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)

[^tongyi_dr]: [Tongyi DeepResearch Technical Report](https://arxiv.org/abs/2510.24701)

#### Moonshot Kimi

[^kimi_k1_5]: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)

[^kimi_k2]: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)

[^kimi_researcher]: [Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)

#### ByteDance Seed / Doubao

[^seed1_5_thinking]: [Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.13914)

[^vapo]: [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)

[^dapo]: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)

[^dapo_github]: [DAPO GitHub Repository](https://github.com/BytedTsinghua-SIA/DAPO)

[^seed1_5_vl]: [Seed1.5-VL Technical Report](https://arxiv.org/abs/2505.07062)

[^ui_tars]: [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326)

[^ui_tars_github]: [UI-TARS GitHub Repository](https://github.com/bytedance/ui-tars)

[^ui_tars_2]: [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://huggingface.co/papers/2509.02544)

[^seed_prover]: [Seed Prover 1.5: Advanced Mathematical Reasoning through a Novel Agentic Architecture](https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture)

[^seed1_8]: [Official Release of Seed1.8: A Generalized Agentic Model](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model)

#### DeepSeek

[^deepseek_math]: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

[^deepseek_r1]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

[^deepseek_v3_2]: [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)

#### Zhipu Z.ai / GLM

[^glm_4_5]: [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)

[^glm_5]: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/html/2602.15763v1)

#### Tencent Hunyuan

[^hunyuan_t1]: [Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)

[^hunyuan_a13b_instruct]: [Hunyuan-A13B-Instruct Model Card](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)

[^hunyuan_a13b]: [Hunyuan-A13B Technical Report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)

#### Baidu ERNIE

[^ernie_4_5_family]: [ERNIE 4.5 Model Family](https://ernie.baidu.com/blog/posts/ernie4.5/)

[^ernie_4_5]: [ERNIE 4.5 Technical Report](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)

[^ernie_5_0]: [ERNIE 5.0 Technical Report](https://arxiv.org/abs/2602.04705)

#### StepFun

[^step3]: [Step3: Cost-Effective Multimodal Intelligence](https://stepfun.ai/research/en/step3)

[^step3_vl_10b]: [STEP3-VL-10B Technical Report](https://huggingface.co/papers/2601.09668)

[^step_deepresearch]: [Step-DeepResearch Technical Report](https://arxiv.org/abs/2512.20491)

#### Meituan LongCat

[^longcat_flash]: [LongCat-Flash-Thinking-2601 Technical Report](https://tech.meituan.com/2026/02/02/longcat-flash-thinking-2601-techreport.html)

#### Ant Ling / Ring

[^ling_1t]: [Ling-1T Model](https://ant-ling.medium.com/deep-insight-efficient-inference-introducing-the-trillion-parameter-ling-1t-model-77d6170e5e8e)

[^ring_1t]: [Ring-1T](https://ant-ling.medium.com/ring-1t-release-the-flow-state-of-insight-born-of-epiphany-c20e8e32817c)

#### Huawei Pangu

[^pangu_ultra]: [Pangu Ultra](https://github.com/pangu-tech/pangu-ultra)

[^pangu_pro_moe]: [Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity](https://arxiv.org/abs/2505.21411)

[^pangu_news]: [Huawei announces open-source Pangu 7B dense and 72B mixture-of-experts models](https://www.huawei.com/cn/news/2025/7/pangu-opensource)

#### 01.AI Yi

[^yi_lightning]: [Yi-Lightning Technical Report](https://arxiv.org/abs/2412.01253)

#### InternLM / Shanghai AI Lab

[^internlm2]: [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)

#### Baichuan and 360 Zhinao

[^baichuan2]: [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)

[^zhinao]: [360Zhinao Technical Report](https://arxiv.org/abs/2405.13386)

#### Skywork and Xiaomi MiMo

[^skywork_or1]: [Skywork Open Reasoner 1 Technical Report](https://huggingface.co/papers/2505.22312)

[^skywork_or1_github]: [Skywork-OR1 GitHub Repository](https://github.com/SkyworkAI/Skywork-OR1)

[^mimo]: [MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)

[^mimo_github]: [Xiaomi MiMo GitHub Repository](https://github.com/XiaomiMiMo/MiMo)

[^mimo_vl]: [Xiaomi MiMo-VL-Miloco Technical Report](https://arxiv.org/abs/2512.17436)

#### Kuaishou, SenseTime, iFlytek

[^keye_vl]: [Kwai Keye-VL Technical Report](https://arxiv.org/abs/2507.01949)

[^sensenova_u1]: [SenseNova U1](https://www.sensetime.com/en/news-detail/51170629?categoryId=1072)

[^spark_x1]: [Spark X1 deep reasoning model](https://news.cgtn.com/news/2025-01-15/China-releases-Spark-X1-deep-reasoning-model-that-packs-a-punch-1AbIq8PzzEI/index.html)

### International Companies and Labs

#### OpenAI

[^instructgpt]: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

[^gpt4]: [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

[^o1]: [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)

[^o3_o4_mini]: [OpenAI o3 and o4-mini System Card](https://openai.com/index/o3-o4-mini-system-card/)

[^o3_operator]: [Addendum to o3 and o4-mini system card: OpenAI o3 Operator](https://openai.com/index/o3-o4-mini-system-card-addendum-operator-o3/)

[^gpt4_5]: [OpenAI GPT-4.5 System Card](https://openai.com/index/gpt-4-5-system-card/)

[^gpt5]: [OpenAI GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)

[^gpt5_1]: [Addendum to GPT-5 system card: GPT-5.1](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/)

[^gpt5_4]: [OpenAI GPT-5.4 Thinking System Card](https://openai.com/index/gpt-5-4-thinking-system-card/)

[^gpt5_5]: [OpenAI GPT-5.5 System Card](https://openai.com/index/gpt-5-5-system-card/)

[^gpt5_5_instant]: [OpenAI GPT-5.5 Instant System Card](https://openai.com/index/gpt-5-5-instant-system-card/)

[^gpt5_codex]: [Addendum to GPT-5 system card: GPT-5-Codex](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/)

[^gpt5_1_codex_max]: [GPT-5.1-Codex-Max System Card](https://openai.com/index/gpt-5-1-codex-max-system-card/)

[^gpt5_2_codex]: [Introducing GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/)

#### Anthropic

[^constitutional_ai]: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

[^anthropic_cai]: [Anthropic Constitutional AI overview](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

[^claude4]: [System Card: Claude Opus 4 & Claude Sonnet 4](https://www.anthropic.com/claude-4-system-card)

[^claude_sonnet_4_5]: [Claude Sonnet 4.5 System Card](https://www.anthropic.com/claude-sonnet-4-5-system-card)

[^claude_opus_4_5]: [Claude Opus 4.5 System Card](https://www.anthropic.com/claude-opus-4-5-system-card)

[^claude_opus_4_6]: [Claude Opus 4.6 System Card](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf)

#### Google DeepMind

[^gemini_1_5]: [Gemini 1.5 Technical Report](https://arxiv.org/abs/2403.05530)

[^gemini_2_5]: [Gemini 2.5 Technical Report](https://arxiv.org/abs/2507.06261)

[^gemini_2_5_deep_think]: [Gemini 2.5 Deep Think](https://blog.google/products/gemini/gemini-2-5-deep-think)

[^gemini_2_5_computer_use]: [Gemini 2.5 Computer Use Model](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/)

[^gemini_3_1_pro]: [Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)

[^gemma_3]: [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)

#### Meta Llama

[^llama3_herd]: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

#### Microsoft Phi

[^phi_4]: [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)

[^phi_4_reasoning]: [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318)

#### NVIDIA Nemotron

[^nemotron_4]: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)

[^llama_nemotron]: [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)

[^nemotron_ultra]: [NVIDIA Llama Nemotron Ultra Open Model](https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/)

[^nemotron_agents]: [Build Enterprise AI Agents with NVIDIA Llama Nemotron Reasoning Models](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)

[^nemotron_h]: [Nemotron-H Reasoning Model Family](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/)

[^nemotron_3]: [Inside NVIDIA Nemotron 3](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)

#### Mistral

[^magistral]: [Magistral](https://arxiv.org/abs/2506.10910)

#### Apple

[^apple_fm]: [Apple Intelligence Foundation Language Models](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models)

[^apple_fm_2025]: [Apple Intelligence Foundation Language Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)

#### xAI Grok

[^grok_1]: [xAI Grok-1 Model Card](https://x.ai/news/grok/model-card)

[^grok_4]: [xAI Grok 4](https://x.ai/news/grok-4)

[^grok_4_1]: [xAI Grok 4.1](https://x.ai/news/grok-4-1/)

[^grok_4_1_card]: [xAI Grok 4.1 Model Card](https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf)

#### IBM Granite

[^granite_3_3]: [IBM Granite 3.3](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras)

[^granite_4_0]: [IBM Granite 4.0](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)

[^granite_4_1]: [IBM Granite 4.1 Build Notes](https://huggingface.co/blog/ibm-granite/granite-4-1)

#### Salesforce xLAM / SFR-RL

[^xlam]: [Salesforce xLAM](https://www.salesforce.com/blog/large-action-model-ai-agent/)

[^sfr_rl]: [Salesforce SFR-RL](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)

#### Amazon Nova

[^nova]: [Amazon Nova](https://aws.amazon.com/nova/)

[^nova_report]: [The Amazon Nova Family of Models: Technical Report and Model Card](https://www.isi.edu/results/publications/31887/the-amazon-nova-family-of-models-technical-report-and-model-card/)

[^nova_premier]: [Amazon Nova Premier: Technical report and model card](https://www.amazon.science/publications/amazon-nova-premier-technical-report-and-model-card)

[^nova_forge]: [Amazon Nova Forge](https://aws.amazon.com/nova/forge/)

#### Cohere Command A

[^cohere_research]: [Cohere Research](https://cohere.com/research)

[^command_a]: [Command A: An Enterprise-Ready Large Language Model](https://cohere.com/research/papers/command-a-technical-report.pdf)

#### Databricks

[^dbrx]: [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)

#### AI21

[^jamba_1_5a]: [Jamba 1.5a: Enhancing AI Safety Through Post-Post-Training Alignment](https://www.ai21.com/research/jamba-1-5a/)

[^jamba_whitepaper]: [Jamba 1.5a Whitepaper](https://lp.ai21.com/hubfs/resources/Jamba-1-5a-Whitepaper.pdf)

#### Cursor

[^cursor_composer_2]: [Cursor Composer 2 Technical Report](https://cursor.com/blog/composer-2-technical-report)

#### LG EXAONE

[^exaone_4_0]: [EXAONE 4.0 Technical Report](https://www.lgresearch.ai/data/cdn/upload/EXAONE_4_0.pdf)

[^k_exaone]: [K-EXAONE Technical Report](https://www.lgresearch.ai/data/cdn/upload/K-EXAONE_Technical_Report.pdf)

#### NAVER HyperCLOVA X

[^hyperclova_x]: [HyperCLOVA X Technical Report](https://arxiv.org/abs/2404.01954)

[^hyperclova_x_think]: [HyperCLOVA X THINK Technical Report](https://huggingface.co/papers/2506.22403)

### Open Baselines and Surveys

#### AI2 Tulu / Survey

[^tulu_3]: [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://openreview.net/forum?id=i1uGbfHHpH)

[^tulu_3_blog]: [Tulu 3 Technical Blog](https://allenai.org/blog/tulu-3-technical)

[^rl_survey]: [Reinforcement Learning for LLM Post-Training: A Survey](https://openreview.net/forum?id=UdsXTNzzvg)
