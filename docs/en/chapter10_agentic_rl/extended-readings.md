---
title: 10.14 Further Reading
---

# 10.6 Agentic RL Extended Reading Index

The first six sections of this chapter covered the core theory, engineering practice, and industrial case studies of Agentic RL. But the landscape of Agentic RL extends far beyond that. In 2025--2026, RL is being applied to an increasing range of agent scenarios: from role-playing to creative writing, from scientific discovery to empathetic dialogue. This page organizes over 120 representative works by theme for further exploration.

::: tip How to use this index
Each theme is ordered as: survey -> methods -> systems. We recommend starting with survey works to build a global view, then going deeper into specific directions as needed. Works marked **[open-source]** include GitHub links and can be used for hands-on experimentation.
:::

## Surveys and Theoretical Foundations

The theoretical foundations of Agentic RL are rapidly taking shape. The surveys collected here map the landscape of this emerging field from different angles: some focus on training recipes and engineering practice, others reconceptualize LLMs as autonomous decision-makers and survey 500+ works around six core capabilities, and still others are written specifically for deep research systems or agentic search tasks. If you want to quickly build a mental model of the Agentic RL landscape, start here.

| Work                                                         | Key highlight                                                                                         | Link                                                                                  |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Adaptation of Agentic AI: A Survey                           | Survey of post-training, memory, and skill adaptation techniques for AI agents                        | [arXiv](https://arxiv.org/abs/2512.16301)                                             |
| Training Recipes for Agentic RL in LLMs                      | Systematic compilation of Agentic RL training recipes, including environments and sampling strategies | [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.173816128.89654321/v1) |
| The Landscape of Agentic RL for LLMs: A Survey               | Treats LLMs as autonomous decision-makers and surveys 500+ works around six core capabilities         | [arXiv](https://arxiv.org/abs/2509.02547)                                             |
| A Comprehensive Survey on RL-based Agentic Search            | Survey of reinforcement learning applied to agentic search tasks                                      | [arXiv](https://arxiv.org/abs/2510.16724)                                             |
| Meta-Thinking in LLMs via Multi-Agent RL                     | Explores how multi-agent RL can enable meta-thinking capabilities in LLMs                             | [arXiv](https://arxiv.org/abs/2504.14520)                                             |
| Reinforcement Learning Foundations for Deep Research Systems | First survey written specifically for RL foundations of deep research systems                         | [arXiv](https://arxiv.org/abs/2509.06733)                                             |

## Deep Research and Information Integration

Deep research agents are one of the hottest application directions in Agentic RL. Unlike simple search-and-summarize, they require models to perform multi-turn, long-horizon information search, cross-validation, and synthesis in real web environments. This section includes everything from end-to-end RL frameworks to citation-aware rewards, covering different scales from 7B small models to 30B large models.

| Work                                  | Key highlight                                                                                                         | Link                                                            |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| DeepResearcher **[open-source]**      | End-to-end RL framework for search interaction in real web environments                                               | [GitHub](https://github.com/GAIR-NLP/DeepResearcher)            |
| Tongyi DeepResearch **[open-source]** | Alibaba Tongyi Lab's 30.5B MoE model (3.3B active), using a two-stage "Agentic Mid-training + Post-training" pipeline | [arXiv](https://arxiv.org/abs/2510.24701)                       |
| IntentRL                              | Trains agents to actively clarify ambiguous user intent before starting long-horizon research                         | [arXiv](https://arxiv.org/abs/2602.03468)                       |
| DR Tulu / RLER                        | RL training scheme using evolved scoring criteria (RLER) to improve long-form research capabilities                   | [AllenAI Blog](https://allenai.org/blog/dr-tulu)                |
| EigentSearch-Q+                       | Introduces structured reasoning tools (Q+) to enhance deep research agent capabilities                                | [arXiv](https://arxiv.org/abs/2604.07927)                       |
| Fathom-DeepResearch                   | Multi-agent system composed of Search and Reason 4B models, generating the DUETQA dataset                             | [arXiv](https://arxiv.org/abs/2509.24107)                       |
| PokeeResearch-7B **[open-source]**    | 7B-parameter open-source deep research agent                                                                          | [HuggingFace](https://huggingface.co/PokeeAI/pokee_research_7b) |
| SFR-DeepResearch                      | Salesforce; focuses on continuous RL training for autonomous single agents                                            | [arXiv](https://arxiv.org/abs/2509.06283)                       |
| CaRR / C-GRPO **[open-source]**       | Introduces citation-aware scoring rewards to curb model hallucination                                                 | [GitHub](https://github.com/THUDM/CaRR)                         |

## Reinforcement Reasoning and Code Generation

RLVR (Reinforcement Learning from Verifiable Rewards) naturally fits code generation tasks -- whether code passes tests and executes correctly are objectively verifiable signals. The works in this section build on this core advantage: some integrate code execution feedback directly into multi-turn training, some explore RLVR without ground-truth supervision, and others discover that models spontaneously learn to generate and execute code, revealing scaling laws.

| Work                                               | Key highlight                                                                                      | Link                                                     |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| rStar2-Agent **[open-source]**                     | GRPO-based 14B Agent RL algorithm showing strong competitiveness on math reasoning                 | [arXiv](https://arxiv.org/abs/2508.20722)                |
| Murphy                                             | Multi-turn RLVR framework integrating code execution feedback directly into training               | [arXiv](https://arxiv.org/abs/2511.07833)                |
| ZeroCoder                                          | Explores improving code generation through RLVR without ground-truth supervision                   | [arXiv](https://arxiv.org/abs/2604.07864)                |
| SARL                                               | Achieves label-free reasoning improvement by rewarding reasoning topology structure                | [arXiv](https://arxiv.org/abs/2603.27977)                |
| Agentic RL Scaling Law / ZeroTIR **[open-source]** | Discovers models spontaneously learn to generate and execute code, revealing training scaling laws | [GitHub](https://github.com/yyht/openrlhf_async_pipline) |
| Agnostics                                          | Language-agnostic code RL training framework                                                       | [Project](https://agnostics.abgru.me)                    |
| ReLook                                             | RL based on visual feedback (rendered screenshots) to optimize web frontend code generation        | [arXiv](https://arxiv.org/abs/2510.11498)                |
| Agentic Code Reasoning                             | Provides low-cost, risk-free reward signals for RL through semi-formal reasoning                   | [arXiv](https://arxiv.org/abs/2603.01896)                |
| Code-Space Response Oracles                        | Uses LLMs as code generation oracles, replacing traditional RL oracles                             | [arXiv](https://arxiv.org/abs/2603.10098)                |

## GUI and Web Agents

GUI agents enable AI to operate graphical interfaces like humans -- clicking buttons, filling forms, navigating web pages. The value of RL here is that SFT can only teach models to "mimic clicks," while RL enables models to "choose the optimal action path based on goals." This section covers approaches from web to mobile, from 3B small models to continual learning frameworks.

| Work                                          | Key highlight                                                                                    | Link                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| WebAgent-R1 **[open-source]**                 | End-to-end multi-turn RL framework improving 3B model success rate from 6.1% to 33.9%            | [GitHub](https://github.com/WebAgent-R1/WebAgent-R1)           |
| Web-Shepherd **[open-source]**                | First step-level reward model specifically for web navigation, evaluating each interaction step  | [GitHub](https://github.com/kyle8581/Web-Shepherd)             |
| CRAFT-GUI                                     | Combines curriculum learning with GRPO to improve GUI agent performance                          | [arXiv](https://arxiv.org/abs/2508.11360)                      |
| MobileRL **[open-source]**                    | Mobile online RL framework using ADAGRPO algorithm                                               | [GitHub](https://github.com/MobileRL/MobileRL)                 |
| Co-EPG                                        | Co-evolution framework simultaneously optimizing GUI agent planning and grounding capabilities   | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40981) |
| Continual GUI Agents                          | Defines and addresses learning problems for GUI agents in continually changing environments      | [arXiv](https://arxiv.org/abs/2601.20732)                      |
| WebFactory                                    | Fully automated closed-loop RL flow that "compresses" LLM intelligence into efficient GUI agents | [OpenReview](https://openreview.net/forum?id=HaIEP2PD4S)       |
| ZeroGUI                                       | Zero human-cost online GUI agent learning framework                                              | [arXiv](https://arxiv.org/abs/2505.23762)                      |
| UI-S1                                         | Semi-online RL training method combining offline and online data advantages                      | [arXiv](https://arxiv.org/abs/2509.11543)                      |
| Generalization in Online RL for Mobile Agents | Studies generalization in online RL for mobile agents, proving RL can surpass SFT baselines      | [OpenReview](https://openreview.net/forum?id=INoDyme6wS)       |

## Embodied Intelligence and Robotics

When RL moves from the digital world to the physical world, agents face not text or images, but continuous control signals and uncertain physical environments. The works in this section explore how LLMs can directly participate in robot reasoning and control: some use RL to optimize spatial reasoning so 7B models surpass GPT-4o, some train self-correction capabilities in pixel-level world models, and others study cross-embodiment transfer and maintaining "cognitive identity" during continual learning.

| Work                             | Key highlight                                                                                                | Link                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| Robot-R1                         | Uses RL to directly optimize robot reasoning; 7B model spatial reasoning surpasses GPT-4o                    | [arXiv](https://arxiv.org/abs/2506.00070)       |
| WMPO **[open-source]**           | RL training in pixel-level visual world models, emerging self-correction capabilities                        | [GitHub](https://github.com/HKUST-PEI-Lab/WMPO) |
| ViVa                             | Uses pre-trained video generation models as value function estimators for state value assessment             | [arXiv](https://arxiv.org/abs/2604.08168)       |
| RoboAgent                        | Achieves embodied task planning through composing foundational capabilities                                  | [arXiv](https://arxiv.org/abs/2604.07774)       |
| Cross-Embodiment Offline RL      | Achieves offline RL across different robot morphologies through morphological grouping strategies            | [arXiv](https://arxiv.org/abs/2602.18025)       |
| Sensory-Motor Control with LLMs  | Enables LLMs to directly generate continuous control policies through iterative policy refinement            | [arXiv](https://arxiv.org/abs/2506.04867)       |
| RM-RL                            | Proposes "role model" RL for precise robot manipulation                                                      | [arXiv](https://arxiv.org/abs/2510.15189)       |
| Learning Without Losing Identity | Studies how embodied agents maintain stable "cognitive identity" while continually learning new capabilities | [arXiv](https://arxiv.org/abs/2604.07799)       |

## Multi-Agent Systems and Collaboration

Multi-agent collaboration is far more difficult than single-agent -- when you learn new strategies your teammates are also changing, making the environment non-stationary; when the team succeeds, who gets credit, and when it fails, who is responsible? The works in this section address these challenges from multiple angles: extending GRPO to multi-agent settings, achieving decentralized coordination through knowledge distillation, solving context drift with digital twins, and large-scale MARL frameworks that jointly optimize sampling and training end-to-end.

| Work                       | Key highlight                                                                                                                     | Link                                         |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| MAPoRL                     | New paradigm for multi-agent collaborative training                                                                               | [arXiv](https://arxiv.org/abs/2502.18439)    |
| M-GRPO                     | Extends GRPO algorithm to multi-agent scenarios                                                                                   | [arXiv](https://arxiv.org/abs/2511.13288)    |
| SAGE                       | Closed-loop self-evolution multi-agent RL framework                                                                               | [arXiv](https://arxiv.org/abs/2603.15255)    |
| MARTI **[open-source]**    | Multi-agent debate framework                                                                                                      | [GitHub](https://github.com/MARTI-LLM/MARTI) |
| KD-MARL                    | Transfers centralized expert coordination to lightweight decentralized agents through knowledge distillation                      | [arXiv](https://arxiv.org/abs/2604.06691)    |
| Value-Guidance MeanFlow    | Value-guided flow model for offline multi-agent RL                                                                                | [arXiv](https://arxiv.org/abs/2604.08174)    |
| FlexMARL                   | First end-to-end training framework jointly optimizing sampling, training, and their orchestration for large-scale LLM-based MARL | [arXiv](https://arxiv.org/abs/2602.09578)    |
| TwinLoop                   | Proposes simulation-in-the-loop digital twin framework to address multi-agent performance degradation from context changes        | [arXiv](https://arxiv.org/abs/2604.06610)    |
| Equivariant Multi-agent RL | Equivariant multi-agent RL for multi-modal vehicle-infrastructure cooperative systems                                             | [arXiv](https://arxiv.org/abs/2604.06914)    |

## World Models and Model-Based RL

The core bottleneck of model-free RL is sample efficiency -- agents must learn through extensive trial and error. World models provide a path around this bottleneck: first learn to "simulate the environment in your head," then generate training data in imagination. This section collects approaches from diffusion world models to object-centric representations, all with the core idea of having policy models interact with world models to complete multi-step planning and training "in imagination."

| Work                    | Key highlight                                                                                 | Link                                        |
| ----------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------- |
| GIRL                    | Generative imagination RL through information-theoretic hallucination control                 | [arXiv](https://arxiv.org/abs/2604.07426)   |
| World4RL                | Diffusion world model for policy refinement in robot manipulation                             | [arXiv](https://arxiv.org/abs/2509.19080)   |
| Dreamer-CDP             | Dreamer variant that does not require reconstructing raw pixel observations                   | [Project](https://zenkelab.org/dreamer-cdp) |
| RLVR-World              | Uses RLVR to directly optimize world models                                                   | [arXiv](https://arxiv.org/abs/2505.13934)   |
| OC-STORM                | Enhances world models with object-centric representations for sample-efficient RL             | [arXiv](https://arxiv.org/abs/2501.16443)   |
| Imagine-then-Plan (ITP) | Policy models interact with world models to generate multi-step trajectories "in imagination" | [arXiv](https://arxiv.org/abs/2601.08955)   |

## Role-Playing and Persona Simulation

Role-playing is not just "pretending to be someone" -- it requires models to maintain consistent personality traits, thinking styles, and behavioral patterns across long conversations. The value of RL here is that through verifiable role-awareness rewards, it reinforces the model's continuous perception of "who I am." The works in this section range from dual-layer thinking frameworks (distinguishing character perspective from model perspective) to multi-character self-play, exploring how to make AI truly "get into character" and maintain role consistency.

| Work                                   | Key highlight                                                                                                                                                        | Link                                                        |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| HER (Human-like Reasoning)             | Proposes dual-layer thinking framework distinguishing character first-person thoughts from LLM third-person thoughts (note: not classic Hindsight Experience Replay) | [arXiv](https://arxiv.org/abs/2601.21459)                   |
| OMAR                                   | Cultivates AI social intelligence through multi-turn self-play RL                                                                                                    | [arXiv](https://arxiv.org/abs/2602.03109)                   |
| R4                                     | Equips reward models and role-playing agents with reasoning and retrieval capabilities                                                                               | [ICLR Poster](https://iclr.cc/virtual/2026/poster/10007049) |
| VeriRole                               | Improves role awareness through verifiable prompt-guided RL                                                                                                          | [OpenReview](https://openreview.net/forum?id=lW7kMpMj9K)    |
| SPELL                                  | Multi-character self-play RL framework for long-context reasoning                                                                                                    | [arXiv](https://arxiv.org/abs/2509.23863)                   |
| Consistently Simulating Human Personas | Proposes a unified framework for evaluating and improving LLM role consistency                                                                                       | [OpenReview](https://openreview.net/forum?id=A0T3piHiis)    |
| CPO                                    | Comparative policy optimization for reward ambiguity in role-playing dialogue                                                                                        | [arXiv](https://arxiv.org/abs/2508.09074)                   |
| RAIDEN-R1                              | Proposes verifiable role-awareness reward (VRAR) to reinforce model perception of its own role                                                                       | [arXiv](https://arxiv.org/abs/2505.10218)                   |

## Creative and Long-Form Writing

Creative writing poses unique challenges for RL: rewards are not as objectively verifiable as code execution, and "good" writing is subjective and multi-dimensional. The works in this section explore how to design reward signals that capture creative quality -- from generative reward models performing multi-dimensional reasoning about story preferences, to optimizing rubric-based reward models through alternating RL, to comparing different reward strategies via RLAIF to stimulate creative capabilities in small models.

| Work                                            | Key highlight                                                                                                   | Link                                                           |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Writer-R1                                       | Memory-augmented Replay Policy Optimization                                                                     | [arXiv](https://arxiv.org/abs/2603.15061)                      |
| R2-Write                                        | Systematic study of open-domain writing, proposing a reflection and revision framework                          | [arXiv](https://arxiv.org/abs/2604.03004)                      |
| DPWriter                                        | Addresses output diversity reduction during RL training through diverse planning branches                       | [arXiv](https://arxiv.org/abs/2601.09609)                      |
| RLMR                                            | First to combine subjective preferences with objective verification in online RL training                       | [arXiv](https://arxiv.org/abs/2508.18642)                      |
| Rewarding Creativity                            | Develops generative reward models for multi-dimensional analysis and explicit reasoning about story preferences | [arXiv](https://arxiv.org/abs/2601.07149)                      |
| Alternating RL for Rubric-Based Reward Modeling | Optimizes rubric-based reward models through alternating RL, achieving SOTA on multiple writing benchmarks      | [arXiv](https://arxiv.org/abs/2602.01511)                      |
| Igniting Creative Writing in SLMs               | Compares two reward strategies under RLAIF framework to stimulate creative writing in 7B small models           | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.868/) |

## Emotional Intelligence and Empathetic Dialogue

Empathy is not just "understanding emotions" -- it requires expressing appropriate responses at the right time while maintaining logical coherence in conversation. The value of RL here is enabling models to learn to balance "emotional support" with "cognitive reasoning." The works in this section range from verifiable emotion rewards to psychology-based empathetic reward modeling, exploring how to provide more grounded reward signals for RL.

| Work                              | Key highlight                                                                                                            | Link                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| RLVER                             | Trains LLM higher-order empathy using verifiable emotion rewards                                                         | [arXiv](https://arxiv.org/abs/2507.03112) |
| CARE                              | Cognitive reasoning-enhanced RL improving logical coherence and support quality in emotional support dialogue            | [arXiv](https://arxiv.org/abs/2510.05122) |
| COMPEER                           | Unified process-outcome RL for structured empathetic reasoning                                                           | [arXiv](https://arxiv.org/abs/2508.09521) |
| DialogXpert                       | Online value RL-based dialogue planning with over 94% success rate on negotiation, emotional support, and other tasks    | [arXiv](https://arxiv.org/abs/2505.17795) |
| EILS                              | Bio-emotion-inspired homeostatic learning signal framework for building adaptive autonomous agents                       | [arXiv](https://arxiv.org/abs/2512.22200) |
| SAGE (Steering Dialog Generation) | Uses latent variables to control long-term behavior of dialogue generation for building emotionally intelligent chatbots | [arXiv](https://arxiv.org/abs/2503.03040) |
| PERM                              | Psychology-based empathetic reward modeling providing more grounded reward signals for RL                                | [arXiv](https://arxiv.org/abs/2601.10532) |

## Art and Visual Creation

RL entering the art world is an interesting crossover -- it models "aesthetic judgment" as an optimizable reward signal. The works in this section cover applications from image generation optimization to hierarchical painting, from personalized hand-drawn illustrations to artistic style learning. Core approaches include: coordinating multiple expert models for iterative image generation optimization, learning artist styles from stroke data through inverse RL, and using hierarchical RL to separate high-level planning from low-level rendering.

| Work                    | Key highlight                                                                                                                         | Link                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Image-POSER             | Reflective RL framework coordinating multiple expert models for iterative image generation optimization based on complex text prompts | [arXiv](https://arxiv.org/abs/2511.11780)                                            |
| HRL-Painter             | Hierarchical RL-based painting method with high-level region planning and low-level stroke execution                                  | [Neurocomputing](https://doi.org/10.1016/j.neucom.2025.129972)                       |
| PersonaSketch-RL        | RL-based strategy for optimizing personalized hand-drawn illustration generation                                                      | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1875952125001338) |
| RMLer                   | Models cross-category concept fusion as an RL problem for synthesizing novel objects                                                  | [arXiv](https://arxiv.org/abs/2512.19300)                                            |
| Sequential Art Creation | Deep RL framework for creating sequential artworks that are visually distinct from inputs                                             | [UTA Thesis](https://mavmatrix.uta.edu/cse_theses/539/)                              |
| MVAEx-RL                | RL-based multi-modal art element extraction and dynamic adaptation strategy for environment design                                    | [Springer](https://link.springer.com/article/10.1007/s44163-025-00712-z)             |
| DailyArt                | Models joint estimation as synthesis-mediated inference, inferring dynamics from single static images                                 | [arXiv](https://arxiv.org/abs/2604.07758)                                            |

## RL Training Infrastructure and Algorithm Innovation

The engineering complexity of Agentic RL far exceeds standard LLM RL -- you need to simultaneously manage model training on GPUs, tool execution on CPUs, and environment interaction over networks. This section focuses on the infrastructure and algorithm innovations supporting these complex training pipelines: from fully asynchronous training systems to scalable synthetic learning environments, from retrieval-augmented policy optimization to new paradigms that convert inference compute into training signals.

| Work                           | Key highlight                                                                                                              | Link                                                       |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| AReaL v1.0 **[open-source]**   | Jointly open-sourced by Ant Group and Tsinghua, enabling "one-click agent integration into RL training"                    | [GitHub](https://github.com/inclusionAI/AReaL)             |
| RollArt / RollARC              | Maximizes multi-task Agentic RL training throughput through decoupled infrastructure (RollARC)                             | [arXiv](https://arxiv.org/abs/2512.22560)                  |
| SparrowRL                      | High-performance RL training system achieving lossless sparse incremental synchronization on commodity networks            | [arXiv](https://arxiv.org/abs/2602.11456)                  |
| Laminar                        | Scalable, robust asynchronous RL post-training system based on fully decoupled architecture                                | [arXiv](https://arxiv.org/abs/2510.12633)                  |
| SCALER                         | Synthesizes scalable adaptive learning environments providing infinitely verifiable reasoning environments for RL training | [arXiv](https://arxiv.org/abs/2601.04809)                  |
| L-Zero (L0)                    | Low-cost, scalable end-to-end universal agent training pipeline                                                            | [arXiv](https://arxiv.org/abs/2506.23667)                  |
| Compute as Teacher (CaT)       | Converts inference-time parallel sampling compute into RL training supervision signals                                     | [arXiv](https://arxiv.org/abs/2509.14234)                  |
| RAPO                           | Retrieval-augmented policy optimization, explicitly expanding agent exploration space during training                      | [arXiv](https://arxiv.org/abs/2603.03078)                  |
| LLM-Explorer **[open-source]** | Tsinghua; a plugin that can enhance exploration capabilities of various RL algorithms                                      | [GitHub](https://github.com/tsinghua-fib-lab/LLM-Explorer) |

## Scientific Discovery and Industrial Applications

RL is moving out of the laboratory and into real application scenarios including chemistry, materials science, medicine, and industrial manufacturing. The works in this section model scientific problems as MDPs: lead compound optimization becomes a search problem under synthetic constraints, materials design becomes an optimization problem using formation energy feedback, and industrial anomaly detection becomes a policy learning problem for data synthesis. These applications demonstrate RL's potential as a "universal decision optimizer."

| Work                                 | Key highlight                                                                                                              | Link                                                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| MolReAct                             | Models lead compound optimization as MDP, using RL for efficient search under synthetic constraints                        | [arXiv](https://arxiv.org/abs/2604.07669)                                                                 |
| PolyRL                               | Multi-objective polymer generation and discovery guided by RL                                                              | [RSC](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00272a)                                  |
| Helix                                | Hierarchical evolutionary RL framework for open-ended scientific problem solving                                           | [arXiv](https://arxiv.org/abs/2603.07642)                                                                 |
| RLFEF                                | RL using formation energy feedback to fine-tune material diffusion models, improving crystal stability                     | [dblp](https://dblp.org/rec/journals/nn/HuangXJY26.html)                                                  |
| AnomalyAgent                         | Industrial anomaly data synthesis agent that optimizes generation of highly realistic anomaly samples through RL           | [arXiv](https://arxiv.org/abs/2604.07900)                                                                 |
| Autonomous Adaptive Solver Selection | Uses constrained RL framework for autonomous solver selection during chemical integration                                  | [arXiv](https://arxiv.org/abs/2604.00264)                                                                 |
| PPO-based Surface Reconstruction     | Deep RL framework based on PPO for surface reconstruction of AgPd alloy catalysts                                          | [AIP PDF](https://pubs.aip.org/aip/jap/article-pdf/doi/10.1063/5.0295785/20878476/045001_1_5.0295785.pdf) |
| MedVR                                | For medical VQA, proposes two RL mechanisms: entropy-guided visual relocation (EVR) and consensus-driven credit assignment | [arXiv](https://arxiv.org/abs/2604.08203)                                                                 |

---

> **Note:** The above works are papers or projects published or preprinted in 2025--2026. Some arXiv papers may have updated versions; we recommend searching by paper title on [arxiv.org](https://arxiv.org) or [Semantic Scholar](https://www.semanticscholar.org) for the latest versions.
