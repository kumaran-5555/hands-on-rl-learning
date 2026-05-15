---
title: '3.9 Chapter Summary: MDP, Value, and Policy'
---

# 3.9 Chapter Summary: MDP, Value, and Policy

## Overview

This chapter develops the core language of reinforcement learning around sequential decision-making: how we model an environment, how we define return, how we evaluate states and actions via value functions, how Bellman equations provide the recursive structure, how we estimate values from data (DP/MC/TD), how tabular Q-Learning works, how we optimize parameterized policies, and why reward design matters.

As the closing summary of Chapter 3, this section collects the key formulas from Sections 3.1 to 3.8 and explains where each one sits in the chapter's logical structure.

The main takeaways of this chapter can be summarized in eight points:

1. A reinforcement learning problem can be formalized as an MDP five-tuple.
2. The agent optimizes discounted cumulative return, not a one-step immediate reward.
3. The state-value function and action-value function evaluate long-term return at the level of states and actions.
4. Bellman equations reveal the recursive structure of value functions.
5. DP, MC, and TD are three fundamental classes of value estimation methods.
6. A parameterized policy can be optimized directly through a policy objective.
7. Algorithms can be categorized by data source into on-policy/off-policy and online/offline.
8. The reward function defines the learning problem itself; reward design shapes the final behavior learned by the agent.

These ideas form the shared theoretical foundation for later topics such as Deep Q-Networks, policy gradients, Actor-Critic methods, PPO, and reinforcement learning methods for large language models.

## Index Of Core Formulas

Below we list the core formulas from Sections 3.1 to 3.8 in one place. Each formula is annotated with its name, what it is used for, and where it was explained.

### 3.1 Two Slot Machines

$$
\mathbb{E}[R_a] = p_a \cdot (+1) + (1-p_a)\cdot(-1) = 2p_a - 1
\quad \text{(expected reward of a single arm; role: compare the average payoff of one action; see 3.1)}
$$

$$
\mathbb{E}[R_T] = \mathbb{E}[R_{a_1}] + \mathbb{E}[R_{a_2}] + \cdots + \mathbb{E}[R_{a_T}] = \sum_{t=1}^{T} \mathbb{E}[R_{a_t}]
\quad \text{(expected total return over T rounds; role: measure the cumulative performance of a whole strategy; see 3.1)}
$$

$$
\mathrm{Regret}(T) = T\mu^* - \sum_{t=1}^{T}\mu_{a_t}, \qquad \mu^*=\max_a \mu_a
\quad \text{(regret; role: quantify how much is lost due to exploration compared with the best arm; see 3.1)}
$$

### 3.2 MDP

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
\quad \text{(MDP five-tuple; role: specify the complete rules of a sequential decision problem; see 3.2)}
$$

$$
P(s' \mid s,a), \qquad R(s,a), \qquad \gamma \in [0,1]
\quad \text{(transition, reward, and discount; role: define dynamics, immediate feedback, and the weight on the future; see 3.2)}
$$

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k} = r_t + \gamma G_{t+1}
\quad \text{(discounted cumulative return; role: define the long-term objective from time t; see 3.2)}
$$

$$
a = \pi(s), \qquad \pi(a\mid s)=P(a\mid s)
\quad \text{(deterministic and stochastic policies; role: describe how the agent chooses actions; see 3.2)}
$$

### 3.3 V(s) And The Bellman Equation

$$
V^\pi(s)=\mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s\right]
\quad \text{(state-value function; role: evaluate the long-term expected return of a state; see 3.3)}
$$

$$
V^\pi(s)=\sum_{a\in\mathcal{A}}\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)V^\pi(s')\right]
\quad \text{(Bellman expectation equation; role: recursively compute value under a fixed policy; see 3.3)}
$$

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)V^*(s')\right]
\quad \text{(Bellman optimality equation; role: define the optimal state value; see 3.3)}
$$

$$
\text{Target}=r+\gamma V(s'), \qquad \delta=\text{Target}-V(s)
\quad \text{(Bellman target and the prototype of TD error; role: turn Bellman recursion into a sample-based learning signal; see 3.3)}
$$

### 3.4 DP, MC, TD

$$
V(s) \leftarrow \sum_a \pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\right]
\quad \text{(DP policy evaluation update; role: iterate values when the model is known; see 3.4)}
$$

$$
\pi'(s)=\arg\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]
\quad \text{(policy improvement; role: build a better greedy policy from the current value; see 3.4)}
$$

$$
V(s) \leftarrow V(s)+\alpha\left[G_t-V(s)\right]
\quad \text{(MC value update; role: correct value estimates using complete returns; see 3.4)}
$$

$$
V(s) \leftarrow V(s)+\alpha\left[r+\gamma V(s')-V(s)\right]
\quad \text{(TD(0) value update; role: bootstrap from one-step targets for online updates; see 3.4)}
$$

$$
\delta = r+\gamma V(s')-V(s)
\quad \text{(TD error; role: measure how much the current estimate violates the one-step Bellman relation; see 3.4)}
$$

### 3.5 Q(s, a)

$$
Q^\pi(s,a)=\mathbb{E}_\pi\left[G_t\mid s_t=s,a_t=a\right]
\quad \text{(action-value function; role: evaluate long-term return starting with action a at state s; see 3.5)}
$$

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)
\quad \text{(V-Q relationship; role: obtain state value as the policy-weighted average of action values; see 3.5)}
$$

$$
Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)\sum_{a'\in\mathcal{A}}\pi(a'\mid s')Q^\pi(s',a')
\quad \text{(Bellman expectation equation for Q; role: recursively compute action values under a fixed policy; see 3.5)}
$$

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)\max_{a'}Q^*(s',a')
\quad \text{(Bellman optimality equation for Q; role: recursively define the optimal action value; see 3.5)}
$$

$$
\pi^*(s)=\arg\max_a Q^*(s,a)
\quad \text{(greedy optimal policy; role: induce an optimal policy from the optimal action-value function; see 3.5)}
$$

### 3.5 Q-Learning

$$
\text{TD Target}=r+\gamma\max_{a'}Q(s',a')
\quad \text{(Q-Learning TD target; role: construct a one-step learning target for Q from experience; see 3.5)}
$$

$$
\delta=r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\quad \text{(Q-Learning TD error; role: measure the gap between current Q estimate and the TD target; see 3.5)}
$$

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
\quad \text{(Q-Learning update; role: incrementally correct the state-action value table; see 3.5)}
$$

$$
a_t=
\begin{cases}
\text{random action} & \text{with probability }\varepsilon\\
\arg\max_a Q(s_t,a) & \text{with probability }1-\varepsilon
\end{cases}
\quad \text{(}\varepsilon\text{-greedy; role: trade off exploration and exploitation; see 3.5)}
$$

### 3.6 Policy Objective

$$
\pi_\theta(a\mid s)=P_\theta(a\mid s)
\quad \text{(parameterized stochastic policy; role: represent an action distribution with parameters theta; see 3.6)}
$$

$$
J(\theta)=\mathbb{E}_{\pi_\theta}\left[G_t\right]
=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
\quad \text{(policy objective; role: measure the expected long-term return of a parameterized policy; see 3.6)}
$$

$$
\theta^*=\arg\max_\theta J(\theta)
\quad \text{(optimal policy parameters; role: pose policy learning as a maximization problem; see 3.6)}
$$

$$
\nabla_\theta J(\theta)\propto
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot G_t\right]
\quad \text{(policy gradient estimator; role: increase the probability of actions that lead to high return; see 3.6)}
$$

### 3.8 Reward Design

$$
R(s,a)=
\begin{cases}
+1 & \text{reach the goal}\\
0 & \text{otherwise}\\
-1 & \text{failure}
\end{cases}
\quad \text{(sparse reward; role: provide learning signal only at success/failure; see 3.8)}
$$

$$
R_{\text{shaping}}(s,a,s')=-\left(\text{dist}(s',\text{goal})-\text{dist}(s,\text{goal})\right)
\quad \text{(distance-based shaping; role: provide intermediate rewards from progress toward the goal; see 3.8)}
$$

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
\quad \text{(potential-based shaping; role: strengthen intermediate signals without changing the optimal policy; see 3.8)}
$$

$$
r_t^{\text{intrinsic}}=\left\|f(s_t,a_t)-s_{t+1}\right\|^2
\quad \text{(prediction-error intrinsic reward; role: encourage exploration where the model predicts poorly; see 3.8)}
$$

$$
r_t^{\text{RND}}=\left\|\hat{\phi}(s_t)-\phi(s_t)\right\|^2
\quad \text{(RND intrinsic reward; role: measure novelty via random network distillation; see 3.8)}
$$

$$
r_t^{\text{total}}=r_t^{\text{extrinsic}}+\beta r_t^{\text{intrinsic}}
\quad \text{(total reward combination; role: combine task reward and exploration reward; see 3.8)}
$$

## Scalar And Matrix Forms

All formulas in this chapter are presented in a per-state (scalar) form. If we stack all states into vectors and write transitions as matrices, the $n$ scalar equations can be compressed into a single line of matrix form.

### Notation

To avoid overly long dimension expressions, we write $n=|\mathcal{S}|$ for the number of states and $n_A=|\mathcal{A}|$ for the number of actions.

| Symbol               | Shape           | Meaning                                                                              |
| -------------------- | --------------- | ------------------------------------------------------------------------------------ |
| $\boldsymbol{v}_\pi$ | $n \times 1$    | values of all states                                                                 |
| $\boldsymbol{r}_\pi$ | $n \times 1$    | expected immediate reward for each state                                             |
| $P_\pi$              | $n \times n$    | policy-induced transition matrix, $P_\pi[i,j]=\sum_a \pi(a\mid s_i)p(s_j\mid s_i,a)$ |
| $\boldsymbol{q}_\pi$ | $nn_A \times 1$ | Q values for all $(s,a)$ pairs                                                       |
| $P$                  | $nn_A \times n$ | transition matrix, $P[(s,a),s']=P(s'\mid s,a)$                                       |
| $\Pi_\pi$            | $n \times nn_A$ | policy matrix, $\Pi_\pi[s,(s,a)]=\pi(a\mid s)$                                       |

### Master Comparison Table

**Bellman expectation equation**

Per-state form:

$$
V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]
$$

Matrix form:

$$
\boldsymbol{v}_\pi
=
\boldsymbol{r}_\pi+\gamma P_\pi\boldsymbol{v}_\pi
$$

**Bellman optimality equation**

Per-state form:

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right]
$$

Matrix form:

$$
\boldsymbol{v}_*
=
\boldsymbol{r}_*+\gamma P_*\boldsymbol{v}_*
\quad\text{(row-wise max)}
$$

**Closed-form solution**

Matrix form:

$$
\boldsymbol{v}=(I-\gamma P)^{-1}\boldsymbol{r}
$$

**V-Q relationship**

Per-state form:

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)
$$

Matrix form:

$$
\boldsymbol{v}_\pi=\Pi_\pi\boldsymbol{q}_\pi
$$

**Bellman expectation equation for Q**

Per-state form:

$$
Q^\pi(s,a)
=
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
$$

Matrix form:

$$
\boldsymbol{q}_\pi
=
\boldsymbol{r}+\gamma P\Pi_\pi\boldsymbol{q}_\pi
$$

**Bellman optimality equation for Q**

Per-state form:

$$
Q^*(s,a)
=
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a')
$$

Matrix form:

$$
\boldsymbol{q}_*
=
\boldsymbol{r}+\gamma P\cdot\mathrm{rowmax}(\boldsymbol{q}_*)
$$

**DP policy evaluation**

Per-state form:

$$
V(s)\leftarrow
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')
\right]
$$

Matrix form:

$$
\boldsymbol{v}_{k+1}
=
\boldsymbol{r}_\pi+\gamma P_\pi\boldsymbol{v}_k
$$

MC and TD update individual states from samples, so they do not have a direct matrix-form counterpart here.

### From Q To V

Substitute $\boldsymbol{v}_\pi = \Pi_\pi \boldsymbol{q}_\pi$ into $\boldsymbol{q}_\pi = \boldsymbol{r} + \gamma P \boldsymbol{v}_\pi$, and then left-multiply both sides by $\Pi_\pi$:

$$
\Pi_\pi \boldsymbol{q}_\pi = \Pi_\pi \boldsymbol{r} + \gamma \Pi_\pi P \boldsymbol{v}_\pi
\quad\Longrightarrow\quad
\boldsymbol{v}_\pi = \underbrace{\Pi_\pi \boldsymbol{r}}_{\boldsymbol{r}_\pi} + \gamma \underbrace{\Pi_\pi P}_{P_\pi} \boldsymbol{v}_\pi
$$

In the matrix view, the Q-form retains the action dimension (the policy averaging is handled separately by $\Pi_\pi$), while the V-form has already absorbed policy averaging into $\boldsymbol{r}_\pi$ and $P_\pi$. This is exactly the matrix-language expression of the statement that "Q carries finer-grained information than V."

## Dependency Structure Of The Formulas

The formulas in this chapter are not isolated; they form a layered sequence of definitions and consequences.

| Layer               | Core Question                                               | Key Objects                                                    |
| ------------------- | ----------------------------------------------------------- | -------------------------------------------------------------- |
| Problem modeling    | What are the environment, actions, feedback, and discount?  | $\mathcal{M}=\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$ |
| Optimization goal   | How do we measure long-term return from a time point?       | $G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}$                      |
| Behavior rule       | How does the agent choose actions in a state?               | $\pi(s)$, $\pi(a\mid s)$, $\pi_\theta(a\mid s)$                |
| State evaluation    | What is a state's long-term value?                          | $V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$                       |
| Recursive form      | How does long-term return decompose into reward and value?  | Bellman expectation equation; Bellman optimality equation      |
| Learning from data  | How do we estimate value when the model is unknown?         | DP, MC, TD, $\delta$                                           |
| Action evaluation   | After fixing the first action, what is the long-term value? | $Q^\pi(s,a)$, $Q^*(s,a)$                                       |
| Tabular control     | How do we learn Q from samples and derive a policy?         | Q-Learning, TD target, $\varepsilon$-greedy                    |
| Policy optimization | How do we optimize a parameterized policy directly?         | $J(\theta)$, $\nabla_\theta J(\theta)$                         |
| Objective design    | What reward signal is the algorithm maximizing?             | $R(s,a)$, $F(s,a,s')$, $r_t^{\text{total}}$                    |

This hierarchy reflects the chapter's logic: we define the environment before defining return, and define return before defining value. Value recursion underlies DP/MC/TD; state and action values support policy improvement; and the reward signal ultimately determines what every optimization objective means.

## The Main Thread Of The Chapter

### From Return To Bellman Recursion

The most important mathematical structure in Chapter 3 is recursion. Discounted return can be written as an infinite sum:

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

The same quantity can be equivalently written in a one-step recursive form:

$$
G_t=r_t+\gamma G_{t+1}
$$

This recursion decomposes long-term return into the current immediate reward and the next-step return. Bellman equations lift this trajectory-level recursion to the expected value $V^\pi(s)$.

### From State Values To Sample-Based Learning

If the environment model $P$ and $R$ is known, we can update values directly using the Bellman expectation equation (DP). If the model is unknown, we must estimate values from sampled trajectories:

- MC uses the complete return $G_t$ as its target. It is unbiased, but has high variance.
- TD uses the bootstrapped target $r+\gamma V(s')$. It has lower variance and can be updated online.
- The TD error $\delta=r+\gamma V(s')-V(s)$ measures the gap between the current estimate and the one-step Bellman target.

This idea becomes the foundation for later techniques such as critics, DQN targets, and GAE.

### From State Values To Action Values

$V^\pi(s)$ evaluates states, but it does not directly tell us how good each action is at state $s$. To capture long-term return at the action level, Section 3.5 introduces the action-value function:

$$
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
$$

This definition fixes the first action and evaluates the long-term return obtained by following policy $\pi$ thereafter. As a result, $Q$ contains more direct information for action selection than $V$. When the optimal action-value function $Q^*(s,a)$ is known, an optimal policy can be induced by $\arg\max_a Q^*(s,a)$.

### From Action Values To Q-Learning

Section 3.5 applies the TD idea to a table of action values. Each experience tuple $(s,a,r,s')$ yields a one-step TD target:

$$
r+\gamma\max_{a'}Q(s',a')
$$

and uses it to correct the current estimate $Q(s,a)$. This allows the agent to learn a decision-ready Q-table without a full environment model and without waiting for an episode to end. Tabular Q-Learning fits small, discrete state spaces; once the state space is large or continuous, we need function approximation methods, which we discuss in Chapter 4.

### From Policy Representation To Policy Optimization

Section 3.6 provides another perspective on learning: instead of explicitly learning values for each action first, we can represent the policy as a parameterized distribution $\pi_\theta(a\mid s)$ and maximize

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[G_t]
$$

The policy gradient expression shows that the update direction has two components. The term $\nabla_\theta\log\pi_\theta(a\mid s)$ describes how to increase the probability of the chosen action, while $G_t$ serves as the return-weight that tells us how strongly to push in that direction. Chapter 5 will derive and refine this result.

### The Reward Function Defines The Objective

Every value function, policy objective, and update rule ultimately depends on the accumulation of rewards. Rewards that are too sparse lead to weak learning signals; poorly designed rewards can cause the agent to optimize something misaligned with the task intention. The reward shaping and intrinsic reward ideas in Section 3.8 aim to strengthen learning signals while keeping the original task objective as unchanged as possible.

## Review Questions

After completing this chapter, you should be able to answer the following questions.

1. Given a task, how do you write its MDP five-tuple?

::: details Reference Answer
Represent the task as $\mathcal{M}=\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$. Here $\mathcal{S}$ is the set of possible states; $\mathcal{A}$ is the set of available actions; $P(s'\mid s,a)$ describes the transition dynamics after taking an action; $R(s,a)$ (or $R(s,a,s')$) defines the immediate reward; and $\gamma$ is the discount factor for future rewards. When writing an MDP, you should explain what each component means in the concrete task, rather than only listing symbols.
:::

2. Why does RL optimize discounted cumulative return rather than only immediate reward?

::: details Reference Answer
Reinforcement learning studies sequential decision-making. An action not only affects the current reward, but also changes future states, which in turn affects future rewards. Therefore, optimizing immediate rewards alone can lead to short-sighted policies. The discounted cumulative return

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

unifies present and future rewards into a single long-term objective, and uses $\gamma$ to control how important the future is. For continuing tasks, $\gamma<1$ also ensures the return remains finite.
:::

3. What do $G_t$, $V^\pi(s)$, $Q^\pi(s,a)$, and $J(\theta)$ evaluate, respectively?

::: details Reference Answer
$G_t$ is the discounted cumulative return starting from time $t$ along a particular trajectory. $V^\pi(s)$ is the expectation of $G_t$ when starting from state $s$ and following policy $\pi$, and is used to evaluate state value. $Q^\pi(s,a)$ is the expectation of $G_t$ when first taking action $a$ in state $s$ and then following policy $\pi$, and is used to evaluate action value. $J(\theta)$ is the overall expected return of the parameterized policy $\pi_\theta$, and is used to measure and optimize the policy itself.
:::

4. What is the difference between the Bellman expectation equation and the Bellman optimality equation?

::: details Reference Answer
The Bellman expectation equation evaluates a given policy $\pi$, where action selection is averaged using $\pi(a\mid s)$:

$$
V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right].
$$

The Bellman optimality equation defines the optimal value. It does not fix any particular policy, but instead takes a maximum over actions:

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right].
$$

The former answers "what is the value if we act according to this policy?", while the latter answers "what is the highest value achievable under optimal actions?"
:::

5. What are the key differences between DP, MC, and TD?

::: details Reference Answer
DP assumes the environment model is known, and updates values by taking expectations over actions and next states. Its errors mainly come from incomplete convergence of the iterative procedure or function approximation errors. MC does not require an environment model, but it must wait until an episode ends to update using the full return $G_t$; it targets the true return, so it is unbiased, but it can have high variance. TD does not require an environment model and does not need to wait for an episode to finish; it can update step-by-step using $r+\gamma V(s')$. Its variance is lower, but because it bootstraps from an estimate $V(s')$, it introduces bias.
:::

6. Why does TD error become the shared learning signal in later critics, deep Q-networks, and GAE?

::: details Reference Answer
The TD error

$$
\delta=r+\gamma V(s')-V(s)
$$

measures the gap between the current value estimate and the one-step Bellman target. Critics can use it to update state-value functions; deep Q-networks use the same bootstrapping idea to construct training targets for Q-functions; and GAE forms advantage estimates by taking weighted sums of TD errors across multiple time steps. Therefore, TD error is the basic mechanism that turns Bellman recursion into a learnable, sample-based training signal.
:::

7. Why can $Q(s,a)$ directly induce action selection?

::: details Reference Answer
$Q(s,a)$ represents the long-term expected return after choosing action $a$ in state $s$. If the action values are known, we can directly compare $Q$ across actions at the same state. When the optimal action values $Q^*(s,a)$ are known, the optimal policy can be written as

$$
\pi^*(s)=\arg\max_a Q^*(s,a).
$$

So the Q-function not only evaluates actions, but also yields an action selection rule by choosing the action with maximal value.
:::

8. Why do parameterized policies need an objective function $J(\theta)$?

::: details Reference Answer
For a parameterized policy $\pi_\theta(a\mid s)$, the object we learn is the parameter vector $\theta$. To optimize $\theta$, we need an objective function with $\theta$ as the variable:

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[G_t].
$$

$J(\theta)$ measures the policy's expected long-term return. Policy gradient methods estimate $\nabla_\theta J(\theta)$ and adjust parameters to increase the probability of actions in high-return trajectories, thereby improving the policy.
:::

9. Why can reward shaping accelerate learning, and why can it also cause objective drift?

::: details Reference Answer
Reward shaping accelerates learning by adding intermediate rewards, so the agent receives learning signals even before reaching the final goal, mitigating the difficulty of sparse rewards. For example, giving additional reward when the agent gets closer to the goal can guide exploration. However, if the shaping reward is poorly designed, the agent may optimize the shaping signal rather than the original task objective, causing objective drift. Potential-based shaping

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
$$

can theoretically preserve the optimal policy, and is therefore a relatively safer form of shaping.
:::

These questions emphasize the conceptual roles behind the formulas. Mastery of this chapter is not just memorizing symbolic forms, but understanding what each object does in a reinforcement learning problem.

## How Later Chapters Use These Ideas

| Later Chapter              | Objects From This Chapter                                            | How They Are Used                                                                                                       |
| -------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Chapter 4: Deep Q-Networks | $Q(s,a)$, $Q^*(s,a)$, $\arg\max_a Q(s,a)$, TD target                 | Approximate action values with neural networks; update Q via bootstrapped targets                                       |
| Chapter 5: Policy Gradient | $\pi_\theta(a\mid s)$, $J(\theta)$, $\nabla_\theta J(\theta)$, $G_t$ | Directly optimize a parameterized policy; raise probability of high-return actions                                      |
| Chapter 6: Actor-Critic    | $V(s)$, TD error, $J(\theta)$                                        | Use a value function as a critic to provide low-variance signals for policy updates                                     |
| Chapter 7: PPO             | $V(s)$, advantage function, TD error, policy objective               | Use a critic to estimate advantages and constrain the update size                                                       |
| Chapters 8+ (LLM RL)       | policy, reward, return, objective                                    | Treat token generation as sequential decisions; convert preference or verification signals into optimization objectives |

So, the formulas in Chapter 3 are not only for this chapter's exercises. They reappear repeatedly in later algorithms. As representations shift from tables to function approximation, these objects re-emerge as neural networks, loss functions, training targets, advantage estimates, and KL constraints.

## Summary

Chapter 3 establishes the basic structure of reinforcement learning theory:

1. Define a sequential decision problem with the MDP five-tuple.
2. Define the long-term objective using discounted cumulative return $G_t$.
3. Evaluate states and actions via $V^\pi(s)$ and $Q^\pi(s,a)$.
4. Reveal the recursive structure of value via Bellman equations.
5. Explain how value can be computed or estimated using DP, MC, and TD.
6. Use $J(\theta)$ to cast parameterized policy learning as an optimization problem.
7. Distinguish algorithm families by how data is collected (on/off-policy, online/offline).
8. Use reward design to explain where the objective comes from and why the objective definition itself shapes learning outcomes.

The next chapter starts from $Q(s,a)$ and introduces the first complete algorithm family: [Chapter 4: Deep Q-Networks](../chapter04_dqn/intro).
