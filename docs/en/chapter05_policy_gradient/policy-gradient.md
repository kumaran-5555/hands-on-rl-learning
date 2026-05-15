---
title: 5.2 The Policy Gradient Theorem and REINFORCE
---

# 5.2 The Policy Gradient Theorem and REINFORCE

In the previous section, we personally ran the bandit experiment and watched the policy network evolve from "randomly choosing" to "firmly choosing B". The core code was only one line:
`loss = -log_prob * reward`.

But that leaves a deeper question: why can this simple formula achieve that? Why does multiplying `log_prob` by `reward` make the network "prefer good actions"? This is not obvious, at least not in the same way as "compute the $Q$ value of every action and take the maximum" feels intuitive (review: [Q(s,a): scoring actions](../chapter03_mdp/value-q)).

This section is devoted to unpacking that line of code. You will see that behind it sits an elegant theorem: the **policy gradient theorem**. It is the theoretical foundation of policy-based RL, and also the mathematical starting point for later algorithms such as PPO and GRPO in large-model alignment.

## Value-Based vs. Policy-Based

Before we dive into the math, it is worth stepping back and clarifying how the methods in this chapter differ, in substance, from DQN in Chapter 4. These are not just two parallel "styles"; they differ fundamentally in how they solve problems, what kinds of problems they can handle, and what their engineering tradeoffs look like.

### The Core Idea

**Value-based** methods (DQN in Chapter 4) are, at their core, about **scoring**: learn a $Q(s,a)$ that assigns a score to each action, then choose the action with the highest score. The policy is implicit: we do not learn "what to do" directly; instead we obtain it indirectly via $\arg\max_a Q(s,a)$ (review: [Q(s,a) and greedy policies](../chapter03_mdp/value-q)).

**Policy-based** methods (this chapter) are, at their core, about **learning the policy directly**: parameterize $\pi_\theta(a|s)$, output a probability for each action, then optimize parameters $\theta$ to maximize expected return (review: [the policy objective $J(\theta)$](../chapter03_mdp/policy-objective)). The policy is explicit: there is no intermediate scoring step.

|                        | Value-Based (DQN)                                                                              | Policy-Based (Policy Gradient)                          |
| ---------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| What it learns         | $Q(s,a)$: how many points each action is worth                                                 | $\pi_\theta(a\|s)$: the probability assigned to actions |
| How it chooses actions | $\arg\max_a Q(s,a)$ (take the highest score)                                                   | Sample from $\pi_\theta(\cdot\|s)$                      |
| Policy form            | Deterministic (always pick the max)                                                            | Stochastic (outputs a distribution)                     |
| Math tools             | [Bellman equations](../chapter03_mdp/value-bellman) + [TD learning](../chapter03_mdp/dp-mc-td) | Policy gradient theorem + gradient ascent               |

### Action Spaces

Value-based methods have a hard constraint: **they can only handle a finite set of discrete actions**. The $\arg\max$ rule requires comparing $Q$ values across all actions. This is fine in CartPole (only 2 actions). But a robot arm's joint torques are continuous values; if the action is, say, $[-10, 10]^6$, then $\arg\max$ over infinitely many candidates is not something you can compute.

Text generation in large language models is even more extreme in a different sense: each step chooses among tens of thousands of tokens, and the full sequence space is combinatorial. You cannot maintain a $Q$ table over all combinations.

Policy-based methods do not compare action scores. They output a probability distribution directly: a Softmax for discrete actions, a Gaussian distribution for continuous actions. With essentially the same policy-gradient machinery, changing the output head can switch you from "left/right" to "continuous torque".

### Exploration

DQN's policy is deterministic (it always takes $\arg\max$), so exploration must be injected externally. In Chapter 4, we used $\varepsilon$-greedy: with probability $\varepsilon$ act randomly; otherwise choose the action with the highest $Q$ value. The $\varepsilon$ schedule must be hand-tuned: too large wastes experience; too small fails to explore (review: [the three components of DQN](../chapter04_dqn/dqn-components)).

Policy gradients naturally output a probability distribution, so exploration is built in. If the network believes an action is worth trying with probability 30%, it will try it 30% of the time. In the previous bandit experiment, the policy started from a uniform distribution and gradually converged to "firmly choose B". There was no $\varepsilon$ schedule; the transition from exploration to exploitation emerged naturally.

### Data Reuse

This is the most practical engineering difference between the two routes. DQN is **off-policy**: a replay buffer stores old data, which can be reused over and over for training (review: [experience replay](../chapter04_dqn/dqn-components)). One transition can appear in many batches, so data efficiency is high.

Policy gradients are **on-policy**: the expectation $\mathbb{E}_{\pi_\theta}$ in the gradient estimator requires data generated by the current policy $\pi_\theta$. Once the policy updates, old data no longer matches the current distribution. This means policy gradients are inherently less data-efficient than DQN, and this is their largest engineering drawback.

### Summary

|               | Value-Based                           | Policy-Based                            |
| ------------- | ------------------------------------- | --------------------------------------- |
| Action space  | Discrete only                         | Discrete + continuous                   |
| Policy type   | Deterministic                         | Stochastic (a probability distribution) |
| Exploration   | External (e.g., $\varepsilon$-greedy) | Built in (sampling from the policy)     |
| Data reuse    | Off-policy (replay buffer)            | On-policy (must use fresh data)         |
| Typical issue | Continuous actions are hard           | High variance, sample inefficiency      |

We now return to the central question of this chapter: if we choose to represent the policy directly as $\pi_\theta(a|s)$, what objective should we optimize, and how do we compute its gradient?

## The Policy Objective $J(\theta)$

In the MDP framework of Chapter 3, we defined the [state value function](../chapter03_mdp/value-bellman) $V^\pi(s)$ as "starting from state $s$, if we follow policy $\pi$, how many points do we expect to earn?" Now we widen the lens from a single state to the entire policy. Instead of asking "is it good from this state?", we ask "how good is this policy overall?"

This is exactly what the [policy objective](../chapter03_mdp/policy-objective) $J(\theta)$ from Chapter 3 means. The answer is natural: across all possible starting points, under policy $\pi_\theta$, what [discounted return](../chapter03_mdp/mdp) do we expect to accumulate?

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

Let us interpret every symbol:

| Symbol                    | Role              | Intuition                                                           |
| ------------------------- | ----------------- | ------------------------------------------------------------------- |
| $\theta$                  | Policy parameters | Neural network weights: changing them changes the policy’s behavior |
| $\pi_\theta$              | Policy            | Given a state, outputs a probability distribution over actions      |
| $J(\theta)$               | Objective         | The policy’s "report card": how many points it earns on average     |
| $\mathbb{E}_{\pi_\theta}$ | Expectation       | Run the policy many times, then average                             |
| $\gamma^t r_t$            | Discounted reward | Reward at time $t$; farther-future rewards are worth less           |

$J(\theta)$ is our north star. The goal is simple: find parameters $\theta$ that maximize $J(\theta)$. The whole policy-gradient route is really answering one question: "how do we find such a $\theta$?"

## Gradient Ascent

How do we make $J(\theta)$ larger? The most classic tool in deep learning: move in the direction of the gradient.

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

| Symbol                    | Role          | Plain meaning                                                           |
| ------------------------- | ------------- | ----------------------------------------------------------------------- |
| $\nabla_\theta J(\theta)$ | Gradient      | "Which way should we adjust parameters to improve the policy the most?" |
| $\alpha$                  | Learning rate | "How big is each step?" Too large oscillates; too small crawls          |
| $+$                       | Ascent        | Note the plus sign: we maximize, not minimize                           |

But now the hard part appears: how do we compute $\nabla_\theta J(\theta)$?

The objective $J(\theta)$ contains an expectation $\mathbb{E}$. In theory, that means enumerating all possible trajectories and averaging, but in any realistic environment the number of possible trajectories is astronomical. This is like wanting the average height of an entire school: you cannot measure everyone, but you can estimate it by randomly sampling 100 students.

## The Policy Gradient Theorem

This is where the policy gradient theorem enters. In 1992, Ronald Williams proved in his REINFORCE paper that the seemingly intractable gradient $\nabla_\theta J(\theta)$ can be transformed into a form that can be estimated by sampling [^1]. Later, Sutton and collaborators further generalized and systematized this result [^2].

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

Do not be intimidated by the length of the formula. Let us read it piece by piece:

| Symbol                                      | Role                 | Plain meaning                                                                   |
| ------------------------------------------- | -------------------- | ------------------------------------------------------------------------------- |
| $\nabla_\theta$                             | Take gradient        | "How should we adjust parameters?"                                              |
| $\log \pi_\theta(a_t \| s_t)$               | Log-probability      | Log-probability that the policy chooses action $a_t$ in state $s_t$             |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | Gradient of log-prob | "How does changing parameters change the probability of selecting this action?" |
| $G_t$                                       | Return               | Total reward from time $t$ to the end: "how many points did we end up with?"    |
| Outer $\mathbb{E}$                          | Expectation          | "Run many times and average" (approximate with samples)                         |

In one sentence: **if an action leads to a good outcome (large $G_t$), increase the probability of taking that action again; if it leads to a bad outcome (small $G_t$), decrease that probability.**

This matches exactly what you observed in the bandit experiment: choosing B tends to win (large $G_t$), so its probability rises; choosing A may occasionally win, but not often enough, so its probability does not dominate. The policy gradient theorem is simply this intuition expressed in precise mathematics.

### The Log-Derivative Trick

You might wonder: why not write $\nabla_\theta \pi_\theta(a_t|s_t) \cdot G_t$ directly, and why introduce a $\log$?

This is a clever mathematical technique called the **log-derivative trick**. By the chain rule,

$$\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$$

That division by $\pi$ cancels the implicit $\pi$ factor that appears in the expectation, making the final estimator clean and computable. From an engineering perspective, $\pi \in (0,1)$; gradients of raw probabilities can be numerically tiny and harm training stability. The $\log$ maps $(0,1)$ to $(-\infty, 0)$, often yielding gradients that behave more nicely in practice.

<details>
<summary>Math derivation: from the objective to the policy gradient theorem</summary>

To take the gradient of the objective, we differentiate with respect to the trajectory distribution:

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \sum_t r_t(\tau)$$

Here $\tau = (s_0, a_0, s_1, a_1, \ldots)$ is a trajectory, and $P(\tau; \theta)$ is the probability that the policy induces $\tau$. The gradient only acts on $P(\tau; \theta)$ (rewards do not depend on $\theta$):

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \sum_t r_t(\tau)$$

The key step is the identity $\nabla_\theta P = P \cdot \nabla_\theta \log P$:

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \left( \nabla_\theta \log P(\tau; \theta) \right) \sum_t r_t(\tau)$$

The trajectory probability factorizes as
$P(\tau; \theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$.
Taking the log and differentiating w.r.t. $\theta$, the environment dynamics term $P(s'|s,a)$ drops out because it does not depend on $\theta$, leaving only the policy terms:

$$\nabla_\theta \log P(\tau; \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Substituting back into the expectation yields the policy gradient theorem. The most beautiful point here is that the environment dynamics are eliminated during differentiation. That is why policy gradients **do not need a model of the environment**; this is a fundamental reason they are much more flexible than dynamic programming methods.

</details>

## The REINFORCE Algorithm

The policy gradient theorem gives the form of the gradient. **REINFORCE** is its most straightforward implementation: use [Monte Carlo sampling](../chapter03_mdp/dp-mc-td) to estimate the expectation (review: the essence of MC is "run a full episode, then look back"). The algorithm is surprisingly simple:

1. Run a complete episode with the current policy $\pi_\theta$, recording states, actions, and rewards at each step.
2. For each time step, compute the return from that step to the end: $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$.
3. Estimate the gradient with samples: $\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$.
4. Update parameters along the gradient direction: $\theta \leftarrow \theta + \alpha \nabla_\theta J$.

In PyTorch, this becomes one line:

```python
loss = -log_prob * G_t  # The minus sign is because PyTorch does gradient descent (minimization), but we need gradient ascent (maximization)
```

Looking back at the previous bandit code, `loss = -log_prob * reward` is exactly the single-step special case of REINFORCE ($G_t = r_t$ because the bandit has only one step and no future).

```python
# REINFORCE core (multi-step version)
for t in range(len(rewards)):
    G_t = sum(gamma ** k * rewards[t + k] for k in range(len(rewards) - t))
    loss += -log_probs[t] * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## The Variance Problem of REINFORCE

REINFORCE looks concise and elegant, but it has a problem severe enough to make it almost "unusable" in practice: **its variance is too large**.

Why? Because $G_t$ is the accumulated return from time $t$ to the end of the episode, which includes all the randomness along the rest of the trajectory. For the same action, different sampled rollouts can produce drastically different $G_t$:

| Case      | What actually happened               | $G_t$ |
| --------- | ------------------------------------ | ----- |
| Good luck | Later steps happened to score highly | Large |
| Bad luck  | Later steps happened to score poorly | Small |

The problem is that the policy gradient uses $G_t$ to judge whether "this action was good". If $G_t$ fluctuates widely, then **the same good action might be punished due to bad luck, and the same bad action might be rewarded due to good luck**. This is like judging a student's true ability from a single exam score: a poor result does not necessarily mean poor understanding; it could just be an off day.

In the previous bandit experiment you already saw the concrete manifestation of this: the training curve was not a smooth upward line, but full of jagged oscillations. If you increase the learning rate (for example, from 0.01 to 0.1), the policy will swing violently between A and B and never stabilize. That is the direct consequence of high variance.

## Discrete vs. Continuous Action Spaces

Our experiments in this chapter use a discrete action space (choose A or choose B), but the policy gradient theorem applies equally to continuous action spaces. This is not a minor theoretical detail; it determines how you design the "output head" of the policy network:

|                           | Discrete action space                     | Continuous action space                           |
| ------------------------- | ----------------------------------------- | ------------------------------------------------- |
| Example                   | CartPole left/right, LLM token choice     | Robot joint angles, steering angle                |
| Output head               | Softmax (probability of each action)      | Gaussian parameters (mean $\mu$ and std $\sigma$) |
| Sampling                  | Sample according to Softmax probabilities | Sample from $\mathcal{N}(\mu, \sigma^2)$          |
| How to compute $\log \pi$ | `log_softmax`                             | Log-density formula of a Gaussian                 |

This difference matters: the same PPO algorithm can be used for LLM alignment (discrete token selection) or for robot control (continuous torque output) by swapping only the output head. This is one of the major reasons policy-gradient methods are far more flexible than value-based methods. DQN’s $\arg\max$ is simply not computable in continuous spaces (you cannot compare infinitely many continuous candidates), whereas policy gradients differentiate the probability density directly and are naturally compatible with continuous actions.

<details>
<summary>Question: What is the essential difference between REINFORCE and Q-Learning updates?</summary>

Q-Learning updates a value function $Q(s,a)$ ("how many points is this action worth?"), and the policy is obtained implicitly via $\arg\max Q$: first you build a score table, then you pick the best action from it. REINFORCE updates the policy parameters $\theta$ directly, skipping the $Q$ step entirely: it does not ask "what is the score?", but learns "what should I do?"

This difference has two important consequences. Q-Learning is off-policy (old data can be reused many times), while REINFORCE is on-policy (it must use fresh data generated by the current policy). Q-Learning can only handle discrete actions (it needs to enumerate actions to take a max), while REINFORCE can handle continuous actions (it differentiates the probability density directly).

</details>

<details>
<summary>Question: Why does REINFORCE need to finish an entire episode before it can update?</summary>

Because $G_t$ requires all rewards from time $t$ until the episode ends. If you do not reach the terminal step, you do not know the full value of $G_t$. This is like not being able to fairly judge a movie until you watch the final scene; stopping halfway prevents a complete evaluation.

This also hints at an optimization direction. If we can replace $G_t$ with a more stable estimate, then we could update without waiting for the episode to finish. Where could such a "more stable estimate" come from? From the [value function](../chapter03_mdp/value-bellman) in Chapter 3. Using $V(s)$ as a baseline turns an "absolute return" into a "relative return". This is exactly what the next baseline experiment and the next chapter on Actor-Critic will do.

</details>

REINFORCE can work, but its "high variance" makes it nearly impractical. Fortunately, the policy gradient theorem has a remarkable property: you can subtract a "baseline" that does not depend on the action from the gradient estimator. This does not change the expected gradient direction, but can dramatically reduce variance. This observation directly leads to the Actor-Critic architecture. In the next chapter, [Actor-Critic](../chapter06_actor_critic/actor-critic), we will see how that works.

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
