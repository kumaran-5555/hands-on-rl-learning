---
title: A Brief History of RL
---

# A Brief History of Reinforcement Learning

If you asked an AI researcher in the early 2010s, "What is reinforcement learning?", they would probably draw a feedback loop of an agent interacting with an environment, and tell you it is mostly used for robotics control and board games.

But if you rewind the clock by a century, or fast-forward to today's era of large models, you will find that reinforcement learning (RL) has gone through a dramatic evolution. It began with behavioral experiments in psychology, and gradually grew into a core engine behind some of the most advanced AI systems we have today.

Before we jump into code, it is worth spending a few minutes on this timeline. These milestones explain why modern RL algorithms look the way they do.

## 1. Origins and Foundations: From Psychology to Mathematical Frameworks (1890s-1950s)

The earliest ideas behind RL did not come from computer science. They came from **psychology and neuroscience**.

In 1898, psychologist Edward Thorndike proposed the **Law of Effect** based on his famous "puzzle box" experiments with cats: behaviors that lead to satisfying outcomes tend to be reinforced; behaviors that lead to unpleasant outcomes tend to be weakened. This is the root of trial-and-error learning.

![Thorndike's Puzzle Box](../../preface/brief-history/images/puzzle_box.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: Thorndike's original puzzle box apparatus. Source: <a href="https://commons.wikimedia.org/wiki/File:Original_%22Puzzle_Box%22_Apparatus_Design.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

Half a century later, the rise of cybernetics and control theory pushed these instincts into rigorous mathematics. In 1957, Richard Bellman introduced the **Markov Decision Process (MDP)** and the **Bellman equation**. He used a 5-tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ to turn sequential decision problems into a precise mathematical object:

- $\mathcal{S}$: the state space
- $\mathcal{A}$: the action space
- $P(s'|s,a)$: transition probabilities
- $R(s,a)$: the reward function
- $\gamma$: the discount factor

Under this framework, an agent seeks a policy $\pi(a|s)$ that maximizes the expected discounted return:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.
$$

To measure how good a policy is, Bellman introduced value functions. $V^\pi(s)$ denotes the expected return starting from state $s$ and following policy $\pi$. Among all policies, the best one corresponds to the **optimal value function** $V^*(s)$, which satisfies the Bellman optimality equation:

$$
V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \, V^*(s') \right].
$$

This equation is profound: the optimal value at the current state equals immediate reward plus the discounted expectation of future optimal values. It transforms an apparently infinite-horizon decision problem into a solvable recursion. This is the conceptual foundation of dynamic programming, and it gave RL a solid theoretical base.

## 2. Theory Takes Shape: Temporal-Difference and Model-Free Learning (1980s-1990s)

Bellman's dynamic programming is mathematically clean, but it has two fatal practical limitations.

1. It assumes you know the environment dynamics: you must have $P(s'|s,a)$ and $R(s,a)$ in advance. In the real world, a robot does not know what lies behind a door, and a game-playing agent does not know the opponent's next move.
2. It suffers from the curse of dimensionality: solving Bellman equations requires enumerating states, but the number of states often grows exponentially with problem complexity. For Go, the number of possible board positions is about $3^{361} \approx 10^{170}$, far beyond any explicit table.

To learn in unknown environments without full state tables, researchers developed new ideas.

In 1988, Richard Sutton proposed **temporal-difference learning (TD)**. TD combines Monte Carlo sampling with the bootstrapping nature of dynamic programming. Its core update rule is simple:

$$
V(s_t) \leftarrow V(s_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma V(s_{t+1}) - V(s_t)}_{\text{TD error } \delta_t} \right].
$$

The TD error $\delta_t$ measures how the new estimate differs from the old one. If the next step turns out better than expected ($\delta_t > 0$), increase the current value; otherwise decrease it. This "learn while you act" mechanism is one of the core ideas of modern RL.

In 1989, Chris Watkins introduced **Q-learning**, one of the most widely taught model-free, off-policy RL algorithms. Its update rule is:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right].
$$

The key is the $\max_{a'}$ term: Q-learning learns the optimal action-value function directly, without needing a model of the environment.

In 1995, Gerald Tesauro demonstrated the practical power of these ideas with **TD-Gammon**, a backgammon program trained with TD learning that reached (and arguably exceeded) expert human level. This was one of the first widely recognized successes of RL with function approximation.

## 3. The Deep Learning Era: From DQN to AlphaGo (2013-2018)

Classic RL algorithms like Q-learning assume you can store values in a table. Deep learning changed the game by letting us approximate value functions and policies with neural networks.

In 2013, DeepMind introduced **DQN (Deep Q-Network)**, combining Q-learning with deep convolutional networks to learn directly from pixels. DQN achieved human-level performance on many Atari 2600 games with the same algorithm and architecture, marking a turning point: RL could scale to high-dimensional inputs with minimal hand-engineering.

![DQN Atari Results](../../preface/brief-history/images/dqn_atari.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 3: DQN performance across dozens of Atari games, surpassing professional human players on many of them. Source: <a href="https://research.google/blog/from-pixels-to-actions-human-level-control-through-deep-reinforcement-learning/" target="_blank" rel="noopener noreferrer">Google Research Blog</a></em>
</div>

In 2016, DeepMind's **AlphaGo** combined deep RL with Monte Carlo Tree Search and defeated world champion Lee Sedol 4:1. This event brought RL into the public spotlight with an unmistakable impact.

![AlphaGo](../../preface/brief-history/images/alphago.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 4: A screenshot from AlphaGo vs. Fan Hui. Source: <a href="https://commons.wikimedia.org/wiki/File:AlphaGo_Fan_Huiren_aurka.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

In 2017, OpenAI introduced **PPO (Proximal Policy Optimization)**. Compared to early policy-gradient methods, PPO found a practical balance between training stability and sample efficiency. Its central idea is to limit the size of each policy update via **clipping**, preventing the infamous "step too large, training collapses" failure mode:

$$
\mathcal{L}^{\\text{CLIP}}(\\theta) =
\\mathbb{E}_t \\left[
\\min\\left(
\\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)} \\hat{A}_t,\\;
\\text{clip}\\left(\\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}, 1-\\epsilon, 1+\\epsilon\\right) \\hat{A}_t
\\right)
\\right].
$$

Here, the ratio $\\pi_\\theta / \\pi_{\\theta_{\\text{old}}}$ compares the new policy against the old one, $\\hat{A}_t$ is an estimate of the advantage function, and $\\epsilon$ is typically 0.1 to 0.2. The clipping behaves like a guardrail: each update is allowed to move, but not too far. PPO quickly became a default workhorse in industry, and OpenAI later used large-scale PPO-based systems (for example, OpenAI Five) to reach world-champion level in Dota 2.

## 4. The LLM Era: New Paradigms for Alignment and Reasoning (2020s-Present)

Just when it seemed RL might stay mostly within games and robotics, large language models (LLMs) gave RL a new mission: **alignment** and **reasoning**.

In 2022, OpenAI released ChatGPT. A key ingredient behind its instruction-following behavior was **RLHF (Reinforcement Learning from Human Feedback)**. The standard RLHF recipe trains a reward model to approximate human preference, then uses PPO to optimize a language-model policy:

$$
\\max_\\theta\\;
\\mathbb{E}_{x \\sim \\mathcal{D},\\; y \\sim \\pi_\\theta(\\cdot|x)}\\left[
r_\\phi(x,y) - \\beta\\, \\text{KL}\\left(\\pi_\\theta(\\cdot|x) \\| \\pi_{\\text{ref}}(\\cdot|x)\\right)
\\right].
$$

The KL penalty term keeps the policy from drifting too far away from a reference model, which is essential for preventing reward hacking.

![Early ChatGPT UI](../../preface/brief-history/images/chatgpt.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 5: An early ChatGPT interface. The release of ChatGPT in 2022 pushed RLHF from papers into real products, marking a new phase of RL for alignment and reasoning. Source: OpenAI <a href="https://openai.com/index/chatgpt/" target="_blank" rel="noopener noreferrer">Introducing ChatGPT</a></em>
</div>

In 2023, researchers introduced **DPO (Direct Preference Optimization)**. The key insight is that you can bypass the explicit reward-model training step and directly fine-tune the policy on preference pairs using a simple classification-like loss. DPO can be derived from the RLHF objective:

$$
\\mathcal{L}_{\\text{DPO}}(\\theta) =
-\\mathbb{E}_{(x, y_w, y_l)}\\left[
\\log \\sigma\\left(
\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)}
- \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}
\\right)
\\right].
$$

Here, $y_w$ (winner) and $y_l$ (loser) are the preferred and dispreferred completions, and $\\sigma$ is the sigmoid function. DPO significantly lowers the engineering barrier of RLHF and became widely adopted in open-source post-training workflows.

From 2024 to 2025, reasoning-focused models (e.g., OpenAI o1 and DeepSeek-R1) brought another shift. In tasks with objective rules (math correctness, code compilation), it became increasingly clear that you can sometimes skip a supervised warm start and train with **pure RL** directly from a base model.

In particular, DeepSeek-R1-Zero demonstrated that with verifiable reward signals, pure RL can lead to the emergence of long chains of thought and even distinct "a-ha" moments. Their **GRPO (Group Relative Policy Optimization)** approach removes the critic network used in PPO, and instead uses group-normalized relative rewards to build an advantage-like signal. For a prompt $q$, sample a group of responses $\\{o_1, o_2, \\ldots, o_G\\}$ and normalize rewards:

$$
\\tilde{r}_i = \\frac{r_i - \\text{mean}(r_1, \\ldots, r_G)}{\\text{std}(r_1, \\ldots, r_G)}.
$$

Then optimize a clipped objective similar in spirit to PPO:

$$
\\mathcal{L}_{\\text{GRPO}}(\\theta) =
\\mathbb{E}_q \\left[
\\frac{1}{G} \\sum_{i=1}^{G}
\\min\\left(
\\frac{\\pi_\\theta(o_i|q)}{\\pi_{\\theta_{\\text{old}}}(o_i|q)} \\tilde{r}_i,\\;
\\text{clip}\\left(\\frac{\\pi_\\theta(o_i|q)}{\\pi_{\\theta_{\\text{old}}}(o_i|q)}, 1-\\epsilon, 1+\\epsilon\\right) \\tilde{r}_i
\\right)
\\right].
$$

This lightweight design avoids training a separate critic network and uses the relative ranking within a group to drive learning, making large-scale reasoning RL more practical on clusters.

## Takeaway

From Thorndike's puzzle box, to Bellman's equations; from DQN on Atari, to today's fast-iterating post-training pipelines with DPO and GRPO: the history of RL is the story of agents that **learn from environments, evolve from feedback, and scale from small systems to giant models**.

RL is no longer a niche theoretical toy. It is one of the most direct roads toward generally capable AI systems. In the chapters ahead, we will follow this history from the first line of code and implement these algorithms ourselves.

## References

[^1]: Bellman, R. (1957). A Markovian Decision Process. _Journal of Mathematics and Mechanics_, 6(5), 679-684. [DOI](https://doi.org/10.1512/iumj.1957.6.56038)

[^2]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44. [PDF](http://incompleteideas.net/papers/sutton-88.pdf)

[^3]: Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. _PhD Thesis, King's College, Cambridge_. [PDF](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)

[^4]: Tesauro, G. (1995). Temporal difference learning and TD-Gammon. _Communications of the ACM_, 38(3), 58-68. [DOI](https://doi.org/10.1145/203330.203343)

[^5]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. [Online](http://incompleteideas.net/book/the-book.html)

[^6]: Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. _arXiv preprint_. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

[^7]: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. _Nature_, 529(7587), 484-489. [DOI](https://doi.org/10.1038/nature16961)

[^8]: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. _arXiv preprint_. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^9]: Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. _arXiv preprint_. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

[^10]: Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv preprint_. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

[^11]: DeepSeek-AI, et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. _arXiv preprint_. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
