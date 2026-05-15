---
title: 5.2 The Policy Gradient Theorem and REINFORCE
---

# 5.2 The Policy Gradient Theorem and REINFORCE

In the previous section, we clarified why we need policy-based methods: DQN
relies on an $\arg\max$ over actions, which breaks down in continuous action
spaces. In that setting, it is more natural to learn the policy
$\pi_\theta(a|s)$ directly. This section answers two questions:

1. What metric should we use to measure "how good" a policy is?
2. How do we optimize that metric?

## The Policy Objective

Back in Chapter 3, we introduced the
[policy objective](../chapter03_mdp/policy-objective) $J(\theta)$ as a way to
measure "how good this policy is overall". The answer is as natural as it
sounds: across all possible starting points, under policy $\pi_\theta$, how much
[discounted return](../chapter03_mdp/mdp) do we expect to accumulate?

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

| Symbol                    | Role              | Meaning                                                             |
| ------------------------- | ----------------- | ------------------------------------------------------------------- |
| $\theta$                  | Policy parameters | Neural network weights: changing them changes the policy's behavior |
| $\pi_\theta$              | Policy            | Given a state, outputs a probability distribution over actions      |
| $J(\theta)$               | Objective         | A policy "report card": the average score achieved by the policy    |
| $\mathbb{E}_{\pi_\theta}$ | Expectation       | Run the policy many times and average                               |
| $\gamma^t r_t$            | Discounted reward | Reward at time $t$; farther-future rewards are worth less           |

$J(\theta)$ is our north star. The goal is simple: find parameters $\theta$ that
maximize $J(\theta)$.

## Gradient Ascent

How do we make $J(\theta)$ larger? The classic move in deep learning is to walk
in the direction of the gradient:

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

| Symbol                    | Role          | Meaning                                                                 |
| ------------------------- | ------------- | ----------------------------------------------------------------------- |
| $\nabla_\theta J(\theta)$ | Gradient      | Which direction should we change parameters to improve the policy most? |
| $\alpha$                  | Learning rate | How big is one step? Too large oscillates; too small is slow            |
| $+$                       | Ascent        | Notice the plus sign: we maximize, not minimize                         |

The difficulty is that $J(\theta)$ contains an expectation $\mathbb{E}$, which
in principle asks us to average over all possible trajectories. The number of
possible trajectories is astronomical; we cannot enumerate them. It is like
trying to compute the average height of every student in a country: you cannot
measure everyone, but you can sample a subset and estimate the average.

## The Policy Gradient Theorem

This is exactly where the policy gradient theorem enters. In 1992, Williams
showed in the REINFORCE paper that the seemingly intractable gradient
$\nabla_\theta J(\theta)$ can be rewritten into a form that can be estimated by
sampling. [^1] Sutton and colleagues later generalized and systematized this
result. [^2]

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

Let's unpack it term by term:

| Symbol                                      | Role                 | Meaning                                                                             |
| ------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------- |
| $\nabla_\theta$                             | Take gradient        | How should we change the parameters?                                                |
| $\log \pi_\theta(a_t \| s_t)$               | Log probability      | Under state $s_t$, the log-probability of choosing action $a_t$                     |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | Gradient of log-prob | How do parameters change the probability of choosing this action?                   |
| $G_t$                                       | Return               | Total reward from time $t$ to the end: how many points did we get after doing this? |
| Outer $\mathbb{E}$                          | Expectation          | Run many episodes and average, approximated by sampling                             |

In one sentence: if an action leads to a good outcome (large $G_t$), increase
the probability of taking it again; if it leads to a bad outcome (small $G_t$),
decrease that probability.

### The Log-Derivative Trick

Why don't we write something like $\nabla_\theta \pi_\theta(a_t|s_t) \cdot G_t$,
and instead introduce a $\log$?

This is a mathematical technique known as the **log-derivative trick**. By the
chain rule:

$$\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$$

That division by $\pi$ cancels the $\pi$ factor hidden inside the expectation,
making the expression clean and estimable. From an engineering point of view,
since probabilities lie in $(0, 1)$, gradients of raw probabilities can become
extremely small and destabilize training. The $\log$ maps $(0, 1)$ to
$(-\infty, 0)$ and typically leads to better numerical behavior.

<details>
<summary>Math derivation: from the objective to the policy gradient theorem</summary>

The gradient of the objective differentiates through the trajectory
distribution:

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \sum_t r_t(\tau)$$

Here $\tau = (s_0, a_0, s_1, a_1, \ldots)$ denotes a trajectory, and
$P(\tau; \theta)$ is the probability that policy $\pi_\theta$ generates
trajectory $\tau$. The gradient can only act on $P(\tau; \theta)$, since the
reward does not depend on $\theta$:

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \sum_t r_t(\tau)$$

The key step is the identity $\nabla_\theta P = P \cdot \nabla_\theta \log P$:

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \left( \nabla_\theta \log P(\tau; \theta) \right) \sum_t r_t(\tau)$$

The trajectory probability factorizes as
$P(\tau; \theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$. Taking
logs and differentiating with respect to $\theta$, the environment transition
probability $P(s'|s,a)$ disappears because it does not depend on $\theta$,
leaving only the policy terms:

$$\nabla_\theta \log P(\tau; \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Substituting back into the expectation yields the policy gradient theorem. The
most beautiful part of this derivation is that the environment dynamics, the
state transition probabilities, cancel out during differentiation. This means
policy gradients do not need a model of the environment, which is the
fundamental reason they are much more flexible than dynamic programming
approaches.

</details>

## The REINFORCE Algorithm

The policy gradient theorem gives us the form of the gradient. **REINFORCE** is
the most straightforward implementation of the theorem: it uses
[Monte Carlo sampling](../chapter03_mdp/dp-mc-td) to estimate the expectation.
The algorithm flow is:

1. Run one full episode with the current policy $\pi_\theta$, recording state,
   action, and reward at each step.
2. For each time step, compute the return from that time until the end of the
   episode: $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$.
3. Estimate the gradient with samples:
   $\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$.
4. Update parameters along the gradient:
   $\theta \leftarrow \theta + \alpha \nabla_\theta J$.

In PyTorch, the core update can be written in one line:

```python
loss = -log_prob * G_t  # minus sign because PyTorch does gradient descent (minimize), while we want ascent (maximize)
```

A full multi-step version:

```python
# REINFORCE core (multi-step version)
for t in range(len(rewards)):
    G_t = sum(gamma ** k * rewards[t + k] for k in range(len(rewards) - t))
    loss += -log_probs[t] * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### A Minimal Example: Multi-Armed Bandits

Before we dive into CartPole, let's use a minimal setting to understand what
`loss = -log_prob * reward` is really doing.

Imagine a bandit with two arms: arm A wins with probability 30%, and arm B wins
with probability 70%. The policy network is just a Softmax layer that outputs
the probability of choosing A and B. The core training code looks like:

```python
probs = policy(state)
dist = torch.distributions.Categorical(probs)
action = dist.sample()  # sample an action by probability
log_prob = dist.log_prob(action)  # log π(a|s)

reward = pull_arm(action.item())  # take the action

loss = -log_prob * reward  # REINFORCE core
```

After 300 episodes, the probability of choosing B will typically climb from
around 0.5 to something like 0.85-0.95: the policy learns to prefer the arm with
the higher win rate. But the curve will not be smooth; it will be jagged and
noisy. If you increase the learning rate from 0.01 to 0.1, the policy may swing
violently between A and B.

This is the core pain point of REINFORCE: **high variance**.

## The Variance Problem in REINFORCE

$G_t$ is the cumulative return from time $t$ until the episode ends, which
includes all randomness along that future trajectory. For the same action,
different sampled trajectories can produce very different $G_t$ values:

| Case      | What actually happened               | $G_t$ |
| --------- | ------------------------------------ | ----- |
| Good luck | Many high rewards happened to follow | Large |
| Bad luck  | Many low rewards happened to follow  | Small |

Policy gradients use $G_t$ to decide whether "this action was good". But when
$G_t$ fluctuates heavily, a good action can be penalized due to bad luck, and a
bad action can be rewarded due to good luck. It is like judging a student's
ability by a single exam: a bad score does not necessarily mean poor mastery; it
may just be an off day.

In the bandit experiment, this shows up as jagged learning curves and
oscillations. In more complex environments, such as CartPole, high variance
makes training even more unstable: sometimes the policy improves nicely, then a
run of bad luck throws it off course.

## Discrete vs. Continuous Action Spaces

In this chapter's experiments we use discrete action spaces, choose A vs. B and
CartPole left vs. right. But the policy gradient theorem applies equally to
continuous action spaces:

|                    | Discrete action space                 | Continuous action space                                         |
| ------------------ | ------------------------------------- | --------------------------------------------------------------- |
| Example            | CartPole left/right, LLM token choice | Robot joint torque, steering wheel angle                        |
| Output layer       | Softmax, probability for each action  | Gaussian parameters, mean $\mu$ and standard deviation $\sigma$ |
| Sampling           | Sample according to Softmax           | Sample from $\mathcal{N}(\mu, \sigma^2)$                        |
| Compute $\log \pi$ | `log_softmax`                         | Log-density formula of a Gaussian distribution                  |

With the same policy gradient formula, changing only the output layer lets us
move from left/right to continuous torques. This is where policy gradients are
more flexible than value-based methods: DQN's $\arg\max$ is not computable in
continuous spaces, while policy gradients differentiate through a probability
density, which is naturally compatible with continuous actions.

<details>
<summary>Thinking question: what is the essential difference between REINFORCE and Q-Learning updates?</summary>

Q-Learning updates the value function $Q(s,a)$, how many points is this action
worth, and the policy is obtained implicitly via $\arg\max Q$. REINFORCE updates
the policy parameters $\theta$ directly, skipping the intermediate step of
learning $Q$ values.

This difference leads to two key consequences: Q-Learning is off-policy, it can
reuse old data many times, while REINFORCE is on-policy, it needs fresh data
generated by the current policy. Q-Learning is naturally limited to discrete
actions, it needs to take a max over actions, while REINFORCE can handle
continuous actions, it differentiates the log probability density directly.

</details>

REINFORCE can work, but its high variance makes it nearly unusable in practice.
Fortunately, the policy gradient theorem has a remarkable property: we can
subtract a baseline that does not depend on the action from the gradient
estimator, without changing the expected direction of the gradient, while
significantly reducing variance. We will develop this in Section 5.4. In the
next section, we will first run vanilla REINFORCE on CartPole:
[Hands-on: CartPole](./cartpole).

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
