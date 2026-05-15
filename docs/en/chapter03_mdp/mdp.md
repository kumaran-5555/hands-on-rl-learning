---
title: 3.2 Markov Decision Processes
---

# 3.2 MDP: Modeling Sequential Decision-Making

## What This Section Solves

**Core content**

- Understand how the MDP 5-tuple describes states, actions, transitions, rewards, and discounting.
- Understand why discounted return $G_t$ is the overall objective in RL.
- Distinguish deterministic policies $a = \pi(s)$ from stochastic policies $\pi(a\mid s)$.

In the previous section, we used two slot machines to establish three basic RL ideas: a **policy** is the rule for choosing actions; **expected return** is an objective way to evaluate a policy; and **exploration vs. exploitation** is a distinctive challenge of RL. Bandits are a great starting point because they remove state changes: no matter what you did last round, the next round looks the same. This lets us isolate trial-and-error and exploration-exploitation without additional complications.

Real tasks are not like that. In CartPole, a push changes the pole angle; in a game, a move changes the map; in language generation, choosing one token changes the future context. You cannot only ask "which action is better on average". You must ask: **given the current situation**, which action is better? And after you act, the situation changes again. The problem upgrades from "which button is better" to "continuous decision-making under evolving situations."

To describe such problems, we need a more complete mathematical framework: the **Markov Decision Process (MDP)**. It is not abstract decoration; it is a precise language for the fact that actions change the next situation. $\mathcal{S}$ specifies what you can observe now, $\mathcal{A}$ specifies what you can do, $P$ specifies how the world changes after acting, $R$ specifies how you are scored, and $\gamma$ specifies how much future rewards matter today.

The purpose of this section is to give you the **modeling language**: the MDP tuple, $G_t$, and $\pi$ are tools for translating a real task into a mathematical object that RL can operate on. Once the language is in place, we can discuss how to solve the problem. In the **next section**, value functions $V(s)$ and Bellman equations become the computational tools for solving MDPs.

Concretely, this section answers three questions: **what are the rules of the game** (the MDP tuple), **what is the overall objective** (discounted return $G_t$), and **how do we choose actions step by step** (the policy $\pi$).

::: info Core concept
The MDP is the **standard modeling language** of reinforcement learning. Almost all RL algorithms, from Q-learning to PPO, and even RLHF for large language models, are built on top of the MDP framework. A bandit is a degenerate MDP (single state, no state transitions). CartPole, Atari, and token-by-token generation are full MDPs.
:::

**Core formulas**

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma) \quad \text{(MDP 5-tuple: modeling sequential decision problems)}
$$

> **Markov Decision Process (MDP):**
>
> - $\mathcal{S}$: the state space, the set of all possible situations.
> - $\mathcal{A}$: the action space, the set of all actions available to the agent.
> - $P$: transition dynamics, describing how the environment evolves after actions.
> - $R$: the reward function, how the environment scores actions.
> - $\gamma$: the discount factor, controlling how short-sighted or long-horizon the agent is.

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma G_{t+1} \quad \text{(discounted return recursion)}
$$

> **Discounted return:**
>
> - $G_t$: the discounted sum of future rewards starting at time $t$.
> - $r_{t+k}$: the reward received at time step $t+k$.

$$
a = \pi(s), \qquad \pi(a\mid s) = P(A_t=a\mid S_t=s) \quad \text{(deterministic vs. stochastic policy)}
$$

> **Policy:**
>
> - $\pi$: the decision rule of the agent.
> - $\pi(s)$: a deterministic policy that outputs a fixed action in state $s$.
> - $\pi(a\mid s)$: a stochastic policy that defines a distribution over actions in state $s$.

## The MDP Tuple: State, Action, and Environment

An MDP is defined by five components, written $(S, A, P, R, \gamma)$. Together they form a complete decision loop: **in a situation (S) you make a choice (A), the world changes (P), you receive feedback (R), and you enter the next situation and repeat. The discount factor $\gamma$ decides how much you care about rewards far in the future.**

Given any RL problem, it is often enough to ask five questions:

- **Where am I? (S)** What does the agent observe? What information defines the current situation?
- **What can I do? (A)** What actions are available? Discrete buttons, or continuous controls (torque, angle, etc.)?
- **What happens next? (P)** After acting, how does the world change? Deterministic or stochastic?
- **How do I get scored? (R)** What feedback is provided after a step? Every step, or only at the end?
- **How much do future rewards count? ($\gamma$)** How do we trade off immediate vs. long-term gains? What is 1 point in the future worth today?

These five components are not arbitrary: they cover every link in sequential decision-making. Without S you cannot define the situation; without A you cannot act; without P you cannot predict consequences; without R you cannot define what is good; without $\gamma$ you cannot trade off short-term vs. long-term. **The 5-tuple is a minimal, complete description of a decision world.**

| Symbol | English                | Meaning         | Intuition                | Bandit                       | CartPole                     |
| ------ | ---------------------- | --------------- | ------------------------ | ---------------------------- | ---------------------------- |
| S      | State                  | state space     | "where am I"             | {initial} (single state)     | R^4 (4D continuous)          |
| A      | Action                 | action space    | "what can I do"          | {choose A, choose B}         | {push left, push right}      |
| P      | Transition probability | dynamics        | "what happens next"      | always returns to same state | determined by physics engine |
| R      | Reward                 | reward function | "how am I scored"        | win +1, lose -1              | survive +1                   |
| γ      | Discount factor        | discounting     | "how much future counts" | not used (single-step)       | typically 0.99               |

### State $S$

The state is a complete description of the environment at a moment. A bandit has a single state: every round looks the same. In CartPole, the state is a 4D vector:

$$s = [x,\ \dot{x},\ \theta,\ \dot{\theta}]$$

representing cart position, cart velocity, pole angle, and pole angular velocity.

The core assumption of an MDP is the **Markov property**: given the current state, the future is independent of the past. You do not need to know how you arrived here to act optimally. Like chess: to choose the best move, you only need the current board, not the last ten moves.

This implies the state must be "complete". If CartPole only gave you the pole angle but not the angular velocity, you would not know whether the pole is accelerating downward or slowing down. It is like seeing only the current speed but not whether the car is accelerating or braking. **The state must contain all information needed for decision-making.**

### Action $A$

A bandit has two discrete actions {choose A, choose B}. CartPole also has two discrete actions {push left, push right}. For robot locomotion, the action may be a continuous joint-torque vector $a \in \mathbb{R}^n$.

The type of action space strongly influences algorithm choice:

| Action space                    | Property                                | Typical algorithms          |
| ------------------------------- | --------------------------------------- | --------------------------- |
| Discrete (e.g. {left, right})   | can enumerate actions and pick the best | Q-learning / DQN            |
| Continuous (e.g. joint torques) | cannot enumerate                        | policy gradient (PPO, etc.) |

### Transition Dynamics $P$

$$P(s' \mid s, a) = \text{probability of transitioning to } s' \text{ after taking action } a \text{ in state } s$$

In a bandit, the state never changes. In CartPole, the dynamics $P$ are determined by Newtonian mechanics. Suppose the current pole angle is $\theta = 0.05$ (a small tilt) and the action is "push right":

$$P(\theta' = 0.03 \mid \theta = 0.05, \text{push right}) \approx 0.7 \qquad (\text{angle decreases; pole returns toward upright})$$

$$P(\theta' = 0.08 \mid \theta = 0.05, \text{push right}) \approx 0.3 \qquad (\text{angle increases further; push is not enough})$$

A key distinction is whether $P$ is known. If $P$ is known, we can compute an optimal policy via dynamic programming. If $P$ is unknown, we must learn by interacting with the environment. In most real RL problems, $P$ is effectively unknown: Go has about $10^{170}$ states; LLM generation involves astronomically many token sequences. **Not knowing $P$ is a core reason RL differs from classical optimization.**

### Reward Function $R$

$$R(s, a) = \text{reward received after taking action } a \text{ in state } s$$

Reward is the **only learning signal** in RL. Without reward, an agent has nothing to learn from. Reward signals can look very different across tasks:

| Task     | Reward signal                 | Notes            |
| -------- | ----------------------------- | ---------------- |
| Bandit   | win +1, lose -1               | immediate, dense |
| CartPole | survive +1 per step           | immediate, dense |
| Go       | terminal +1 (win) / -1 (lose) | delayed, sparse  |

Richard Sutton proposed the **reward hypothesis**: any goal can be described as "maximizing expected cumulative reward" (Sutton & Barto, 2018) [^2]. Whether you want an agent to win a game, drive a car, or chat politely, if you can encode it as a reward signal, RL can in principle learn it.

But designing a good reward function is one of the hardest parts of RL engineering. The classic failure mode is **reward hacking**: the agent finds loopholes in the reward and achieves a high score in unintended ways. In LLM alignment, even defining "what deserves a high score" is itself a research problem, which is exactly what RLHF and DPO are trying to address.

### Discount Factor $\gamma$

$$\gamma \in [0, 1]$$

$\gamma$ controls how much the agent values delayed gratification. Consider two policies. Policy A gets 10 points at step 1 and then nothing. Policy B gets 0 points for the first 9 steps and then 20 points at step 10. Which is better?

| $\gamma$ | $G_0$ of policy A | $G_0$ of policy B         | Better policy               |
| -------- | ----------------- | ------------------------- | --------------------------- |
| 0.5      | 10                | $0.5^9 \times 20 = 0.39$  | A (very short-sighted)      |
| 0.9      | 10                | $0.9^9 \times 20 = 7.7$   | A (still prefers near-term) |
| 0.99     | 10                | $0.99^9 \times 20 = 18.3$ | B (long-horizon preference) |

The closer $\gamma$ is to 1, the more long-horizon the agent becomes and the more it values delayed rewards. We will discuss its mathematical role in more detail in the next section.

### Comparing Three Tasks

Let's put bandits, CartPole, and LLM alignment side by side and see what their MDP components look like:

| Component | Bandit                                 | CartPole         | LLM alignment                         |
| --------- | -------------------------------------- | ---------------- | ------------------------------------- |
| S         | {initial} (single state)               | R^4 (continuous) | generated token prefix                |
| A         | {choose A, choose B}                   | {left, right}    | next token in vocabulary              |
| P         | payout probability (fixed but unknown) | physics engine   | LM sampling dynamics                  |
| R         | ±1                                     | +1 per step      | human preference / reward model score |
| $\gamma$  | n/a (single-step)                      | 0.99             | often unused (sequence-level scoring) |

Line by line:

**S**: a bandit has a single state: every round looks the same. CartPole's state is a 4D continuous vector. An LLM's state can be viewed as the entire generated token prefix so far; generating one more token changes the state.

**A**: in a bandit you choose A or B. In CartPole you choose push left or push right. In LLM generation, the action is "which token to emit next", chosen from a vocabulary of tens of thousands of candidates.

**P**: for a bandit, the dynamics are just the payout probabilities, fixed but unknown. For CartPole, dynamics come from the physics engine. LLMs are special: here $P$ is essentially the language model itself. Given the current token prefix, the distribution of the next token comes from the model's softmax. In that sense, in LLM training $P$ and the policy $\pi$ are often very closely coupled.

**R**: bandits give immediate ±1. CartPole gives +1 per step. In LLM alignment, reward is more complex: it is often not a fixed hand-written function, but a separate reward model network that scores the entire response (RLHF), or implicit preference supervision as in DPO.

**$\gamma$**: bandits do not use discounting (single-step). CartPole often uses 0.99. In many LLM alignment setups discounting is not used because reward is applied to the whole response rather than each token, so $G_t$ effectively collapses to a single terminal reward.

These three tasks look very different on the surface, but underneath they share the same modeling language.

## Discounted Return $G_t$: Summing Future Rewards

The MDP tuple specifies the rules. But rules alone are not enough: we must define **what the agent is optimizing**. In episodic tasks, an **episode** is one complete run from `reset` until termination or a time limit. In CartPole, an episode starts with an upright pole and ends when it falls, the cart goes out of bounds, or the time limit is reached.

To evaluate how well an episode went, we aggregate rewards across steps. $G_t$ is that objective: it sums rewards along the interaction and discounts future rewards, defining "how many points you can still get starting from now".

The tuple defines the one-step rules, but RL is not a one-step game. CartPole runs for up to 200 steps; an LLM may generate hundreds of tokens. Each step yields a state, action, and reward. Chaining them forms a **trajectory**:

$$s_0, a_0, r_0, \; s_1, a_1, r_1, \; s_2, a_2, r_2, \; \ldots$$

In episodic tasks, a full episode usually corresponds to a full trajectory. But "trajectory" is a data-centric term: it can refer to a whole episode, a segment of experience sampled for training, or a slice of interaction from a continuing task.

A bandit has only one step, so the trajectory is $(s_0, a_0, r_0)$ and the score is just that single reward. What if CartPole survives for 4 steps and gets +1 each step? The most direct score is the sum: $1 + 1 + 1 + 1 = 4$. But is 1 point now really equal to 1 point 10 steps later?

Not necessarily. Rewards further in the future are often treated as less valuable. The discount factor $\gamma$ controls this. With discounting, we define the discounted return $G_t$:

$$G_t = r_t + \gamma \, r_{t+1} + \gamma^2 \, r_{t+2} + \gamma^3 \, r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k}$$

A concrete CartPole example: reward is +1 per step, $\gamma = 0.9$, and the pole falls after 4 steps:

$$G_0 = \underbrace{1}_{r_0} + \underbrace{0.9 \times 1}_{r_1} + \underbrace{0.9^2 \times 1}_{r_2} + \underbrace{0.9^3 \times 1}_{r_3} = 1 + 0.9 + 0.81 + 0.729 = 3.439$$

Even though each reward is +1, the reward at step 3 is discounted to $0.729$. Multiplying by $\gamma=0.9$ three times reduces a "future point" to less than three quarters of its original value.

### Why Discounting Helps

**Mathematical reason.** In an infinite-horizon game, if CartPole could survive forever and get +1 each step:

$$G_t = 1 + 1 + 1 + \cdots = \infty$$

then any policy that never terminates has $G_t = \infty$, making policies impossible to compare. Discounting makes the infinite sum converge to a finite value. When $\gamma < 1$ and $|R| \le R_{\max}$:

$$G_t \leq R_{\max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{\max}}{1 - \gamma}$$

| $\gamma$ | Upper bound     | Interpretation                       |
| -------- | --------------- | ------------------------------------ |
| 0.9      | $10 R_{\max}$   | mostly cares about ~10 future steps  |
| 0.99     | $100 R_{\max}$  | cares about ~100 future steps        |
| 0.999    | $1000 R_{\max}$ | almost equal-weight over ~1000 steps |

**Economic interpretation.** Even in finite-horizon tasks, discounting can be meaningful. In economics this is the time value of money: 100 dollars today is worth more than 100 dollars next year because you can invest the money in the meantime. $\gamma$ quantifies the preference for receiving reward earlier rather than later.

### The Recursive Structure

$G_t$ can also be written in a recursive form:

$$G_t = r_t + \gamma \, G_{t+1}$$

To see why, expand $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$. Factor out $\gamma$ from the tail: $G_t = r_t + \gamma (r_{t+1} + \gamma r_{t+2} + \cdots) = r_t + \gamma G_{t+1}$.

Meaning: **return from time $t$ equals the immediate reward plus the discounted return from time $t+1$**. The future is summarized by $G_{t+1}$.

This recursion is the foundation of Bellman equations. Later you will see that Bellman equations are built directly on this relationship. Keep this identity in mind; it will reappear repeatedly.

## The Policy $\pi$: How Actions Are Chosen

The MDP tuple defines the rules, and $G_t$ defines the objective. But rules and objectives do not produce behavior by themselves. We still need a **decision rule** that chooses actions in each state. The policy $\pi$ is that decision rule. The goal of training is to find an optimal policy $\pi^*$ that maximizes $G_t$.

Policies come in two forms.

**Deterministic policy**: always outputs the same action for a given state:

$$a = \pi(s)$$

Q-learning and DQN often end up with a deterministic policy: after learning $Q(s,a)$, choose $\arg\max_a Q(s,a)$. The advantage is simplicity. The downside is lack of exploration: if early estimates are wrong, the agent can get stuck exploiting a suboptimal action. In practice DQN is usually combined with $\epsilon$-greedy exploration.

**Stochastic policy**: outputs a probability distribution over actions:

$$\pi(a \mid s) = P(a \mid s)$$

Stochastic policies naturally support exploration: there is always some probability of trying non-greedy actions without adding a separate exploration mechanism.

Back to bandits: Policy 1 (uniform random) is $\pi(\text{choose A}) = 0.5, \pi(\text{choose B}) = 0.5$. Policy 2 (always A) is a deterministic policy $\pi(s) = \text{choose A}$.

In CartPole, the policy takes a 4D state vector as input and outputs probabilities for "left" and "right". Early in training it is close to uniform (e.g. $\pi(\text{left}) \approx 0.5$). After learning, it outputs something like $\pi(\text{right}) \approx 0.95$ when the pole tilts right. This corresponds to the Chapter 1 training curve moving from random behavior to stable balancing.

In LLM alignment, the policy is the language model itself: given the generated token prefix (state), it outputs a distribution over the next token (action). In Chapter 2, the model you trained via DPO is exactly a policy network.

### Why Stochastic Policies Are Common

| Reason            | Deterministic policy                                  | Stochastic policy                                  |
| ----------------- | ----------------------------------------------------- | -------------------------------------------------- |
| Exploration       | needs extra mechanisms (e.g. $\epsilon$-greedy)       | built-in                                           |
| Differentiability | outputs a discrete action; gradient is zero/undefined | outputs a distribution; gradients are well-defined |
| Unpredictability  | predictable, exploitable by opponents                 | introduces variation                               |

In 2014, Silver et al. proposed the deterministic policy gradient theorem [^3], showing that deterministic policies can also be optimized with gradients in continuous action spaces. In practice, however, mainstream algorithms such as PPO and A3C still use stochastic policies, because they make both exploration and optimization easier.

## Section Summary

This section introduced the **modeling language** of RL. Three descriptive tools translate almost any sequential decision problem into a unified mathematical object:

**1. The MDP tuple $(S, A, P, R, \gamma)$ specifies the rules.** For any RL problem, ask five questions: where am I (S), what can I do (A), what happens next (P), how am I scored (R), and how much does the future count ($\gamma$). Bandits, CartPole, and LLM alignment look different, but they share this same language.

**2. Discounted return $G_t$ defines the overall objective.** Distant rewards count less, and $\gamma < 1$ ensures infinite-horizon returns converge to finite values. The recursion $G_t = r_t + \gamma G_{t+1}$ is the foundation of Bellman equations.

**3. The policy $\pi$ defines how actions are chosen.** Deterministic policies $a = \pi(s)$ are simple (DQN), while stochastic policies $\pi(a \mid s)$ support exploration and are differentiable (PPO). Training aims to find an optimal policy $\pi^*$ that maximizes $G_t$.

At this point we have a full language to **describe** problems, but we still lack tools to **solve** them. $G_t$ is a global metric; it does not tell you whether a particular intermediate situation is good. If you knew the "value" of each state, you could compare options and make better decisions. In chess, if one move leads to a position worth 80 points and another leads to 60, you would choose the first. Value functions $V(s)$ and Bellman equations play that role in RL, and that is the topic of the next section.

Next we introduce $V(s)$ and the core tool for computing it: the Bellman equation. See: [Value functions and the Bellman equation](./value-bellman).

## References

[^1]: Bellman, R. (1957). _Dynamic Programming_. Princeton University Press.

[^2]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.

[^3]: Silver, D., et al. (2014). Deterministic Policy Gradient Algorithms. _ICML 2014_.
