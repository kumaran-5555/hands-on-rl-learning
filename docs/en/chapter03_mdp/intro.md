# Chapter 3: MDP, Value Functions, and Policy Optimization

## Chapter Overview

**Core Content**

- Master the unified language of the MDP tuple, discounted cumulative return, value functions, and the Bellman equation.
- Understand how DP, MC, and TD estimate value functions under different assumptions.
- Distinguish the $Q(s,a)$ route from the $J(\theta)$ route, and understand how the reward function determines the optimization objective.

**Core Formulas**

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle \quad \text{(MDP tuple: defining the rules of the environment)}
$$

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} \quad \text{(Discounted cumulative return: defining the long-term objective)}
$$

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \quad Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a] \quad \text{(State value and action value: evaluating states and actions)}
$$

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right] \quad \text{(Policy optimization objective: measuring a policy's average return)}
$$

**Role of This Chapter's Formulas**

Chapter 3 establishes a unified formulation of reinforcement learning through a set of foundational formulas. The MDP tuple is used to characterize the sequential decision-making environment the agent operates in; the discounted cumulative return $G_t$ defines the long-term optimization objective; the state value function and action value function evaluate the long-term return of states and actions; the policy objective $J(\theta)$ formulates the optimization problem for parameterized policies. Subsequent topics — DQN, policy gradient, Actor-Critic, and PPO — all build on these basic objects.

In Chapter 1, we trained an agent to balance a pole on the CartPole inverted pendulum — a typical **classical control task** where states and actions are low-dimensional vectors and rewards are directly given by physics rules. In Chapter 2, we turned to **language model preference alignment**, using DPO to teach a large language model to distinguish good responses from bad ones, without needing to manually write a reward function. These two chapters approached from very different scenarios, demonstrating the basic usage of reinforcement learning.

The experience from the first two chapters let us run the code, but left some unresolved questions: What exactly is the reward in CartPole optimizing? Why does the preference learning behind DPO work? To answer these questions, we need to go deeper from "how to use" to "why" — that is the goal of this chapter.

Back to the most fundamental question: **What does reinforcement learning study?** The answer is **sequential decision-making** — the agent chooses an action at each step, the environment provides feedback and transitions to the next state, and so on. The key is that the agent pursues not the immediate reward at any single step, but the **cumulative return over the entire process**. Greedily taking the largest immediate reward is often not the optimal strategy.

To formally describe this process requires a unified formal framework — the **Markov Decision Process** (MDP). The MDP packages **states, actions, transition probabilities, reward functions, and discount factors** into a tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$. With it, value functions, the Bellman equation, Q-Learning, policy gradient, PPO — these seemingly diverse algorithms — all share a common language.

After defining the problem, the next step is to define a **measure of "goodness."** The **discounted cumulative return** $G_t$ describes the total reward obtainable from a trajectory starting at a given time, while the **value function** distributes the expectation of this reward to specific states or actions. Building on this, the **Bellman equation** reveals a key recursive structure: the value of a state equals "the immediate reward for the current step" plus "the discounted value of the next state." This seemingly simple equation is the common starting point for dynamic programming, Monte Carlo methods, and temporal difference methods.

Following the value function further naturally splits into **two algorithmic routes**:

- **Value-based route**: Learn $Q(s,a)$, score each action, then choose the action with the highest score — leading to Q-Learning and Deep Q-Networks.
- **Policy-based route**: Directly define a policy objective $J(\theta)$ and optimize the policy parameters through gradient methods — leading to policy gradient, Actor-Critic, and PPO.

This chapter serves as the **theoretical foundation** for the entire book. Chapter 4's Deep Q-Networks depend on $Q(s,a)$, Chapter 5's policy gradient depends on $J(\theta)$, and Chapter 6's Actor-Critic and PPO use both value estimation and policy optimization ideas. After understanding this chapter, many formulas in subsequent algorithms will no longer appear as isolated techniques, but as natural consequences derived from **the same decision modeling framework**.

## Section Outline

| Section | Core Content |
| --- | --- |
| [Two-Armed Bandit](./bandit) | Understand exploration, exploitation, and expected return from the smallest decision problem |
| [Markov Decision Process](./mdp) | Define the MDP tuple, discounted return, and policy |
| [Value Functions and the Bellman Equation](./value-bellman) | Introduce the state value function and derive its recursive structure |
| [DP, MC, and TD](./dp-mc-td) | Compare three value estimation methods by assumptions, data requirements, and update rules |
| [From Q to Q-Learning](./value-q) | Use GridWorld to illustrate action value, TD targets, exploration, and tabular boundaries |
| [From Value to Policy](./policy-objective) | Define the objective function from the perspective of directly optimizing the policy |
| [Where Does Data Come From](./algorithm-taxonomy) | Discuss On-policy vs. Off-policy, Online vs. Offline |
| [Reward Function Design](./reward-design) | Discuss how rewards express task objectives and the problems that incorrect rewards can cause |
| [Chapter Summary](./panorama) | Summarize core formulas, algorithmic routes, and connections to subsequent chapters |

## Learning Objectives

After reading this chapter, you should be able to:

- Formally describe a reinforcement learning problem using the MDP tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$;
- Understand how the **Bellman equation** writes long-term return as a recursive structure, and grasp the core differences among dynamic programming, Monte Carlo, and temporal difference methods for value estimation;
- Articulate the distinction between two algorithmic routes — the **$Q(s,a)$ route** scores actions and selects the best, while the **$J(\theta)$ route** directly optimizes policy parameters — leading respectively to Deep Q-Networks and PPO.

It is recommended to complete the multi-armed bandit experiment before entering the formal definition of MDP. This way, you can first observe the exploration-exploitation tradeoff in a sufficiently simple environment, and then elevate the intuition into general mathematical language. The next section starts from the smallest reinforcement learning problem: [Hands-on: Exploration and Exploitation — Multi-Armed Bandit](./bandit).
