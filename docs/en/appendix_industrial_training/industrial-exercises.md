---
title: B.5 Industrial Practice Exercises
---

# B.5 Industrial Practice Exercises

This page does not present exercises as abstract algorithm puzzles. Instead, it works backwards from real job requirements: what post-training and RL roles actually do day to day, and what kind of projects convincingly demonstrate competence.

The examples here were distilled from public job postings around May 2026. Job descriptions change, so do not memorize company names. Focus on the stable capabilities behind the roles: data, training, reward, evaluation, systems, and the product feedback loop.

## One-Sentence Summary

Industrial post-training/RL roles usually do not end at "knowing PPO formulas." They expect you to deliver outcomes like:

- rewrite a business problem into trainable and measurable model-behavior targets,
- build SFT, preference, reward, RLVR, or agent-trajectory datasets,
- choose among SFT, DPO, RM, PPO, GRPO, RLOO, RLVR, and explain why,
- run stable training while monitoring KL, entropy, reward, pass rate, length, throughput, and regressions,
- diagnose reward hacking, capability regressions, data contamination, evaluation distortion, and slowdowns,
- translate model improvements into product metrics, UX, safety/compliance, or real-environment task success.

## From Role Descriptions to a Capability Map

| Region | Typical Role Titles                                                                                | What They Emphasize                                                                                                      | What You Should Be Able to Deliver                                                                                                    |
| ------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| China  | LLM post-training engineer, domain post-training engineer, search/assistant post-training engineer | full SFT/RM/RL pipeline, data synthesis, online feedback flywheels, domain effectiveness, domestic/hybrid compute stacks | data cleaning and mixture plans, SFT/DPO/RLHF experiments, evaluation reports, badcase attribution, iteration plan for real scenarios |
| China  | RL engineer (robotics/autonomy/recsys/marketing)                                                   | problem modeling, simulation environments, offline RL, PPO/SAC/GRPO tuning, sim-to-real or online effects                | MDP modeling, reward design, sim training, offline evaluation, deployment risk control, postmortems                                   |
| US     | post-training research engineer/scientist                                                          | final post-training for product models, capability + safety + eval + productionization                                   | end-to-end training experiments, robust eval, behavior diagnosis, training code debugging, product feedback loops                     |
| US     | agentic post-training / synthetic RL / RL engineering                                              | agents, tool use, code ability, synthetic environments, graders, RL system throughput and reliability                    | agent environments, reproducible graders, trajectory storage, RLVR/GRPO training, pipeline performance optimization                   |
| US     | reward models / alignment research                                                                 | preferences, reward models, LLM-as-judge, rubrics, reward hacking                                                        | preference data plans, RM training and calibration, judge consistency eval, anti-cheating datasets                                    |
| EU     | AI scientist / research engineer / research platform                                               | frontier models, enterprise customization, platforms, HPC, eval, privacy compliance                                      | training/eval toolchains, enterprise data adaptation, reliable deployment, experiment management and reproducible reports             |
| EU     | robotics / autonomous systems / RL research engineer                                               | robotics, autonomy, defense, simulation and real-world alignment                                                         | simulators, policy training, scenario libraries, safety eval, sim-to-real gap analysis                                                |

The key point: post-training roles center around "improving language-model behaviors," while classic RL roles center around "policies acting in environments." In the LLM era, the two overlap heavily in agents, tool use, code, math, robotics, and autonomy.

## How to Interpret Regional Differences

**China roles: more emphasis on business deployment and data flywheels.**

Common keywords in public postings include "application scenarios," "vertical domains," "data synthesis," "online feedback," "evaluation systems," "training efficiency," and "real user experience." This means you cannot just show an open-source benchmark score; you must also explain: where user problems come from, how badcases enter the data pool, how SFT/preference/RL data are mixed, and how you prove business experience improved after training.

**US roles: more emphasis on frontier post-training and large-scale systems.**

Roles at OpenAI, Anthropic, and similar companies place post-training as the critical step before releasing final product models. Common tasks include eval, graders, reward models, RL systems, agentic models, tool use, code capabilities, and training pipeline stability. Candidates must be able to switch between research, engineering, and product boundaries.

**EU roles: more emphasis on research platforms, enterprise scenarios, and embodied/autonomous systems.**

European LLM companies handle enterprise customization, research platforms, HPC, evaluation, and compliance constraints; at the same time, RL roles in robotics, autonomous driving, and defense autonomous systems emphasize simulation, real-world systems, safety boundaries, and engineering verification. When practicing, do not only work on chat models — keep at least one "environment-interactive RL" project.

## Real Work Breakdown

### 1. Post-Training Algorithm Engineer

What you actually do:

- Translate product problems into training objectives. For example, "more helpful answers" must be decomposed into factuality, coverage, structure, refusal boundaries, and citation quality.
- Build SFT data, preference data, reward data, RL prompt pools, and regression evaluation sets.
- Choose among SFT, DPO, RM+PPO, GRPO, RLVR, or mixed losses, and explain compute, data, and risk tradeoffs.
- Run training and monitor loss, reward, KL, entropy, length, repetition rate, refusal rate, pass rate.
- Analyze badcases, determining whether the problem comes from data, rewards, sampling, training hyperparameters, prompt templates, or evaluation.

Practice:

- Pick a vertical domain — legal, education, financial research, code assistant, or enterprise customer service — and build 500 SFT examples, 200 preference pairs, and 100 regression tests.
- Run LoRA SFT on a 1B/3B/7B open-source model, then do one round of DPO or ORPO preference optimization.
- Write a training report: data sources, filtering rules, training configs, metric changes, badcases, next-round data plan.

### 2. Reward Model / Judge / Grader Engineer

What you actually do:

- Define what a "good answer" means, decomposing vague preferences into rubrics.
- Design hybrid scoring schemes: human annotation, synthetic preferences, LLM-as-judge, rule verifiers.
- Train or calibrate a reward model, checking whether it favors verbosity, boilerplate, excessive refusals, or fixed formats.
- Provide stable rewards for RL training and continuously monitor reward hacking.

Practice:

- Build a math or code verifier: correct answer gives main reward; format, length, execution safety give auxiliary rewards.
- Build an LLM-as-judge rubric: factuality, instruction following, completeness, safety each scored 1-5, with spot-checks for judge-human annotation consistency.
- Construct 50 counterexamples of "looks polite but contains no information" and test whether the reward model is fooled.

### 3. RL / GRPO / RLVR Training Engineer

What you actually do:

- Design verifiable rewards for math, code, tool calling, or agent tasks.
- Control on-policy data, sampling group size, KL penalty, advantage normalization, training batch, and rollout batch.
- Diagnose reward increasing while real metrics declining, KL spiking, entropy collapse, outputs getting longer or shorter.
- Achieve stable training with limited compute, not just one-time high scores.

Practice:

- Set up an RLVR training on GSM8K, MATH subset, or HumanEval subset: sample 4-8 responses per prompt, use rules or execution results for reward.
- Compare three configurations: low KL, medium KL, high KL; plot reward, pass rate, KL, entropy, average output length.
- Write a one-page incident postmortem: if reward rises but pass@1 drops, how do you diagnose and roll back.

### 4. Agentic Post-Training Engineer

What you actually do:

- Build tool-calling environments: browser, code executor, file system, database, API.
- Record complete trajectories: observation, action, tool call, tool result, reward, failure reason.
- Design graders that judge whether tasks are truly completed, not just whether final text looks polished.
- Teach the model multi-turn planning, function calling, error recovery, and termination.

Practice:

- Build a small coding agent training environment: input issue, model modifies code, runs tests, reward for passing tests.
- Create 30 web/file operation tasks, record trajectories and write graders — e.g., "find price and fill in form," "fix a failing test."
- For each failed trajectory, annotate failure type: didn't understand task, wrong tool-call parameters, couldn't recover, stopped early, hallucinated file path.

### 5. RL Systems / Training Infrastructure Engineer

What you actually do:

- Optimize throughput of rollout, reward, training, buffer, and weight sync.
- Handle policy version, staleness, long-tail completions, GPU utilization, failure retries.
- Build training health checks so errors surface on small clusters first, rather than discovering a large training run is broken days later.
- Support new algorithm integration while maintaining stability, speed, and reproducibility.

Practice:

- Write a simplified rollout buffer: record prompt, response, reward, old logprob, policy_version.
- Simulate asynchronous training: have rollout workers generate data using stale policies, observe staleness impact on metrics.
- Add 10 monitoring metrics to a training script: tokens/s, samples/s, reward latency, KL, entropy, OOM count, retry count, queue length, policy lag, eval regression.

### 6. Robotics / Autonomous Driving / Autonomous Systems RL Engineer

What you actually do:

- Model tasks as MDPs: states, actions, rewards, termination conditions, constraints.
- Train policies in simulation, then analyze the sim-to-real gap.
- Do safety evaluation: collision, boundary violations, energy consumption, comfort, robustness, extreme scenarios.
- Debug collaboratively with perception, control, simulation, and hardware teams.

Practice:

- Build a continuous control task with Gymnasium, MuJoCo, Isaac Gym, or PyBullet, comparing PPO and SAC.
- Design three versions of reward: task success only, adding energy constraints, adding safety constraints; observe policy differences.
- Write a sim-to-real risk checklist: sensor noise, latency, dynamics errors, actuator limits, environment distribution shift.

## Eight Industrial Exercises

### Exercise 1: Turn Job Posts Into a Skills Matrix

**Goal**: Train yourself to read real work from job descriptions.

**Tasks**:

1. Find 3 post-training or RL roles each from China, US, and Europe.
2. Decompose each role into 6 columns: algorithms, data, evaluation, systems, product/business, safety/compliance.
3. Mark which evidence projects you have already done, and which you still need.

**Deliverable**: A capability-matrix table plus a 300-word summary: which roles you are targeting and the most important project gap to close next.

### Exercise 2: Minimal Domain Post-Training Loop

Tasks:

1. Collect 20 job posts for the role type you are targeting.
2. Decompose each job into 6 columns: algorithms, data, evaluation, systems, product/business, safety/compliance.
3. Mark what evidence projects you already have, and what evidence you still need.

Deliverable: a capability-matrix table plus a ~300-word summary stating which roles you target and the single most important project gap to close next.

### Exercise 2: Minimal Domain Post-Training Loop

Scenario: build a finance research assistant or legal QA assistant, aiming for more professional and reliable answers.

Tasks:

1. Build 500 SFT examples and define filtering rules.
2. Build 200 preference pairs and define chosen/rejected criteria.
3. Run SFT, then run one round of DPO/ORPO.
4. Build 100 regression tests covering factuality, instruction following, safe refusals, and format stability.
5. Attribute 20 badcases.

Deliverable: a data card, training configs, an eval table, a badcase table, and a next-iteration data mixture plan.

### Exercise 3: RLVR for Math or Code

Scenario: improve verifiable correctness for math or code tasks.

Tasks:

1. Pick a subset of GSM8K/MATH or HumanEval/MBPP.
2. Write a verifier: check final answers for math; run unit tests for code.
3. Sample 4-8 responses per prompt and run a small GRPO/RLOO/RLVR training.
4. Track reward, pass@1, pass@k, KL, entropy, and mean length.
5. Construct 10 reward-hacking examples and explain how to fix the verifier.

Deliverable: training curves, config comparisons, verifier code, and a reward-hacking postmortem.

### Exercise 4: Calibrate a Reward Model and a Judge

Scenario: RM scores keep increasing, but human eval says answers are becoming hollow.

Tasks:

1. Design a 4D rubric: factuality, completeness, helpfulness, safety.
2. Label or synthesize 300 preference pairs.
3. Train a small RM or use LLM-as-judge scoring.
4. Check length bias, boilerplate bias, and over-refusal bias.
5. Add 50 counterexamples and re-evaluate.

Deliverable: the rubric, sample preference data, judge/RM consistency report, bias analysis, and a fix plan.

### Exercise 5: Build an Agent Task Environment

Scenario: train an agent that fixes bugs in small repositories.

Tasks:

1. Collect 20 small issues, each with runnable tests.
2. Define an action space: read file, edit file, run tests, submit answer.
3. Record tool calls and test results for every trajectory.
4. Design rewards: tests passing, change scope, recovery from failures, safety constraints.
5. Analyze 10 failed trajectories.

Deliverable: environment spec, grader, example trajectories, failure taxonomy, and next-iteration data.

### Exercise 6: Training-System Health Check

Scenario: an RL training run slows down on day 3, and model metrics start to fluctuate.

Tasks:

1. Draw the dataflow: rollout, reward, buffer, trainer, weight sync.
2. Add `policy_version` and queue-length monitoring.
3. Define 5 alerts: KL anomalies, entropy collapse, high reward latency, eval regression, excessive policy lag.
4. Simulate a long-tail completion that slows down a batch.
5. Write rollback and restart strategies.

Deliverable: system diagram, metrics dashboard, alert rules, incident postmortem, and recovery plan.

### Exercise 7: China-Style Product Data Flywheel

Scenario: an AI search or education product wants more accurate and more user-aligned answers.

Tasks:

1. Sample user logs and define filters for "trainable" data.
2. Split logs into SFT, preference, evaluation, and red-team sets.
3. Propose a synthesis plan, and explain how to prevent synthetic data from contaminating evaluation.
4. Define a weekly iteration rhythm: sampling, labeling, training, evaluation, release, feedback collection.
5. Define privacy, copyright, and safety boundaries.

Deliverable: a flywheel flowchart, a data-mixture table, release metrics for gray rollout, and a risk checklist.

### Exercise 8: US/EU-Style Large-Scale Agent/RL Project Plan

Scenario: build a frontier agent with tool use, aiming to improve code, browsing, and multi-tool coordination.

Tasks:

1. Define 3 task environment classes: code fixing, web information retrieval, API tool calling.
2. Write graders for each environment, and define which signals can be automated vs which need human spot-checking.
3. Design training phases: SFT imitation, preference optimization, RLVR/GRPO, multi-turn badcase replay.
4. Design evaluation gates: which metrics must not regress before release.
5. Write a 4-week schedule and compute budget.

Deliverable: project plan, environments, graders, evaluation gates, and a risk-and-mitigation list.

## How to Present Projects in Interviews

A convincing project is not "I ran a framework." It should clearly answer five questions:

1. **Goal**: what behavior you improve, and why it matters.
2. **Data**: where data comes from, how it is cleaned and mixed, and how leakage is prevented.
3. **Training**: why this algorithm, what key hyperparameters are, and how stability is ensured.
4. **Evaluation**: how offline metrics, human eval, regression tests, and product metrics cross-validate.
5. **Postmortem**: where it failed, how badcases were attributed, and what you changed next.

If you are targeting China post-training roles, prioritize showing "domain data + post-training + evaluation + product feedback loop." If you are targeting US frontier post-training/RL roles, prioritize showing "training systems + eval/graders + agentic RL + large-scale experiment habits." If you are targeting EU roles, prioritize showing "research engineering capability + enterprise/physical/robotics scenarios + HPC/compliance/safety verification."

## Reference Job Samples

- OpenAI's Post-Training role emphasizes improving pretrained models into ChatGPT, API, and other real products, requiring eval construction, research stack debugging, and product-driven research.[^openai-post-training]
- OpenAI's Agentic Post-Training role emphasizes factuality, instruction following, function calling, tool use, grading stack, user-data flywheel, and large-scale RL/post-training infrastructure.[^openai-agentic]
- OpenAI's Synthetic RL role emphasizes synthetic data, environments, feedback, self-play, simulators, and training dynamics analysis.[^openai-synthetic-rl]
- Anthropic's Production Model Post-Training role emphasizes the full post-training stack, Constitutional AI, RLHF, evaluation pipelines, training debugging, and reproducibility.[^anthropic-post-training]
- Anthropic's RL Engineering role emphasizes RLHF training system speed, reliability, usability, pipeline profiling, health checks, and new algorithm implementation.[^anthropic-rl-engineering]
- Anthropic's Reward Models role emphasizes preference learning, LLM-based grading, rubrics, reward hacking, and reward model generalization.[^anthropic-reward-models]
- Shanghai AI Lab's LLM training role emphasizes CPT, SFT, RLHF/DPO, data cleaning, Megatron-LM, veRL, LLaMA-Factory, training monitoring, badcase analysis, and domain effectiveness.[^shlab-training]
- Tencent's public postings show that domestic post-training roles often emphasize PostTraining, SFT/RM/RL, reward systems, data synthesis, online feedback data flywheels, personalization, long-term memory, and comprehensive evaluation.[^tencent-yuanbao][^tencent-hunyuan]
- Mistral AI's Forge product role turns fine-tuning, reinforcement learning, and post-training workflows into enterprise-ready products; its EMEA Applied Scientist/Research Engineer role emphasizes simulation data, training evaluation, agent/RAG, and engineering scenario integration.[^mistral-forge][^mistral-ai4engineering]
- Helsing's European RL roles and Project Centaur public materials show that embodied/autonomous systems RL emphasizes simulation, real-world systems, safety, and policy capabilities in task environments.[^helsing-rl][^helsing-centaur]
- Google DeepMind's Careers page describes Research Engineers as bridging theory and implementation, building scalable systems, and testing and evaluating new ideas — capabilities that also correspond to European/UK research engineering roles.[^deepmind-careers]

[^openai-post-training]: OpenAI, "Research Engineer / Research Scientist, Post-Training", <https://openai.com/careers/research-engineer-research-scientist-post-training-san-francisco/>

[^openai-agentic]: OpenAI, "Researcher, Agentic Post-Training", <https://openai.com/careers/researcher-agentic-post-training-san-francisco/>

[^openai-synthetic-rl]: OpenAI, "Researcher, Synthetic RL", <https://openai.com/careers/researcher-synthetic-rl-san-francisco/>

[^anthropic-post-training]: Anthropic, "Research Engineer, Production Model Post-Training", <https://www.anthropic.com/careers/jobs/4613592008>

[^anthropic-rl-engineering]: Anthropic, "Machine Learning Systems Engineer, RL Engineering", <https://www.anthropic.com/careers/jobs/4952051008>

[^anthropic-reward-models]: Anthropic, "Senior Research Scientist, Reward Models", <https://www.anthropic.com/careers/jobs/5024835008>

[^shlab-training]: Shanghai AI Laboratory, "LLM Training Algorithm Engineer", <https://www.shlab.org.cn/joinus/detail/7615234376275773734?mode=social>

[^tencent-yuanbao]: Third-party repost of Tencent role, "Yuanbao - LLM Post-Training Algorithm Engineer", <https://jobs.niuqizp.com/job-vyU55n5n5.html>

[^tencent-hunyuan]: Third-party repost of Tencent role, "Hunyuan LLM Post-Training Algorithm Engineer", <https://jobs.niuqizp.com/job-vmU55NnaZ.html>

[^mistral-forge]: Mistral AI, "Product Manager, Forge", <https://jobs.lever.co/mistral/11087966-f183-44b1-adc9-3a400c1f52ad>

[^mistral-ai4engineering]: Mistral AI, "Applied Scientist / Research Engineer, AI4Engineering - EMEA", <https://jobs.lever.co/mistral/249d0ec9-1824-41cb-8c4f-cb17a1d5d111>

[^helsing-rl]: Helsing, "AI Research Engineer - Reinforcement Learning", <https://helsing.ai/jobs/4676357101>

[^helsing-centaur]: Helsing, "Helsing Announces Project Centaur: Autonomy for Air Combat", <https://helsing.ai/newsroom/helsing-announces-project-centaur-autonomy-for-air-combat>

[^deepmind-careers]: Google DeepMind, "Careers", <https://deepmind.google/careers/>
