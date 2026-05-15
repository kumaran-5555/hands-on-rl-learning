---
title: 6.1 The Advantage Function
---

# 6.1 The Advantage Function

At the end of Chapter 5, we saw a useful fact: subtracting a baseline $V(s)$ can reduce the variance of policy gradients without changing the expected gradient direction. In this section, we push that insight one step further and introduce the advantage function, the conceptual bridge that connects the Actor and the Critic.

::: tip Prerequisites Used in This Section

- [REINFORCE policy gradient](../chapter05_policy_gradient/reinforce): $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$ and where the baseline is inserted
- [State value $V(s)$](../chapter03_mdp/value-bellman): what makes a good baseline
- [Action value $Q(s,a)$](../chapter03_mdp/value-q): the definition of the advantage depends on the difference between $Q$ and $V$
- [TD error](../chapter03_mdp/dp-mc-td): $\delta = r + \gamma V(s') - V(s)$, a practical estimator of the advantage
  :::

## From a Baseline to the Advantage Function

Recall the REINFORCE [policy gradient](../chapter05_policy_gradient/reinforce):

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$$

$G_t$ is the total return from the current time step to the end of the episode (review: [discounted return](../chapter03_mdp/mdp)). The difficulty is that $G_t$ can fluctuate drastically: under the same policy, from the same state, two rollouts can yield very different $G_t$.

After subtracting the baseline $V(s)$:

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot (G_t - V(s))$$

The term in parentheses, $G_t - V(s)$, is already an estimate of an **advantage**: it measures how much better (or worse) this particular outcome was compared with what the state “normally” yields. The formal definition of the advantage function is:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) \tag{6.1}$$

Here $Q^\pi(s,a)$ is the [action-value function](../chapter03_mdp/value-q) (the expected return if we start from state $s$, take action $a$ first, and then follow the policy), and $V^\pi(s)$ is the [state-value function](../chapter03_mdp/value-bellman) (the expected return if we start from $s$ and follow the policy immediately). Their difference answers a very specific question: “Because I chose action $a$ in state $s$, how many extra points did I gain compared with the state’s average outcome?”

In words, the advantage says:

**How much better is this action than what we would typically get in this state?**

- $A > 0$: the action is better than expected; we should choose it more often
- $A < 0$: the action is worse than expected; we should choose it less often
- $A \approx 0$: the action is about as good as expected

An analogy from board games: $V(s)$ is “this position has a 60% win rate overall,” while $Q(s, \text{play rook})$ is “after playing the rook move, the win rate becomes 75%.” The advantage is $A = 75\% - 60\% = 15\%$, meaning the rook move is 15 percentage points better than the average outcome of the position, so it is a strong choice.

## Advantage Versus Cumulative Return

Why does the advantage reduce variance? The key is that it **subtracts the points you would have gotten anyway**, leaving only the part attributable to the specific action.

Consider a concrete numerical example. Suppose at some state $s$, the policy’s average return is $V(s) = 10$. Two episodes yield returns $G_t^{(1)} = 15$ and $G_t^{(2)} = 5$.

|           | Using $G_t$                         | Using $A = G_t - V(s)$                      |
| --------- | ----------------------------------- | ------------------------------------------- |
| Episode 1 | gradient signal $= 15$ (very large) | gradient signal $= 5$ (moderate)            |
| Episode 2 | gradient signal $= 5$ (medium)      | gradient signal $= -5$ (negative direction) |

If we use $G_t$ directly, both episodes produce a positive gradient signal (only the magnitude differs), so both look “good.” If we use $A$, Episode 1 produces a positive signal (“5 above average”), while Episode 2 produces a negative signal (“5 below average”). The advantage converts an _absolute_ return into a _relative_ return. That makes the learning signal more precise and, in practice, lower-variance.

## Estimating the Advantage with the TD Error

The definition $A = Q - V$ is clean, but in practice we usually do not compute $Q$ explicitly. Using the [TD error](../chapter03_mdp/dp-mc-td), we can estimate the advantage in a more efficient way:

$$A(s,a) \approx r + \gamma V(s') - V(s) = \delta \tag{6.2}$$

This is exactly the TD error introduced in Chapter 3. It measures “after taking one step, how much better (or worse) was the outcome compared with what we predicted?” Replacing $G_t$ with $\delta$ as the policy-gradient signal has two practical benefits:

1. **No need to wait until the end of the episode**: we can update after every step (using $G_t$ requires a full episode, which is a limitation of [Monte Carlo methods](../chapter03_mdp/dp-mc-td))
2. **Lower variance**: $\delta$ involves randomness from only a single step, while $G_t$ aggregates randomness along the entire trajectory

This is the same MC-to-TD transition, now appearing in the policy-optimization setting: REINFORCE uses $G_t$ (MC), while Actor-Critic uses $\delta$ (TD).

|                    | **REINFORCE (MC)**                      | **Actor-Critic (TD)**                                      |
| ------------------ | --------------------------------------- | ---------------------------------------------------------- |
| Advantage estimate | $G_t - V(s)$ (requires full trajectory) | $r + \gamma V(s') - V(s) = \delta$ (update after one step) |
| Update timing      | after the episode ends                  | every step                                                 |
| Variance           | high                                    | low                                                        |
| Cost               | none                                    | requires training a Critic                                 |

## Implementing the Critic Network

To compute $\delta = r + \gamma V(s') - V(s)$, we need $V(s)$ and $V(s')$. But in real problems, $V$ is unknown, so we approximate it with a function approximator. This network is the **Critic**.

```text
Actor (policy network)             Critic (value network)
  input:  state s                   input:  state s
  output: π_θ(a|s) distribution      output: V_φ(s) scalar
  role:   choose actions             role:   evaluate state value
  params: θ                          params: φ
```

The Actor and the Critic share the same input (the state $s$), but produce different outputs: the Actor produces a probability distribution over actions, while the Critic produces a scalar value estimate. They cooperate through the advantage estimate $A \approx \delta$: the Critic provides an evaluation signal, and the Actor adjusts its behavior based on that evaluation.

But how do we train the Critic? How does it learn to estimate $V(s)$ accurately? The next section revisits the three methods briefly surveyed in Chapter 3, and shows how [DP, MC, and TD](../chapter03_mdp/dp-mc-td) are applied concretely when training the Critic. See: [Critic training methods](./critic-training).
