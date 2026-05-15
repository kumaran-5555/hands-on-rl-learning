---
title: E.2.1 Probability Basics
---

# E.2.1 Probability Basics: Probability, Conditional Probability, and Expectation

> **Prerequisites**: This article does not require prior probability theory, but it is useful to read the two-state running example in the [appendix introduction](./intro) first.

---

## Start With "What Could Happen?"

Probability theory does not begin with policies. It begins with a more basic question: what can a random event look like?

The **sample space** $\Omega$ is the set of all possible outcomes. For example, when rolling one die:

$$
\Omega=\{1,2,3,4,5,6\}.
$$

An **event** is a subset of the sample space. For example, the event "the die shows an even number" is:

$$
A=\{2,4,6\}.
$$

A **random variable** is a function that maps random outcomes to numbers. If $X$ denotes the die result, then $X(\omega)=\omega$. In reinforcement learning, the reward $R_{t+1}$, the return $G_t$, and the next state $S_{t+1}$ can all be viewed as random variables.

Only after we have random variables can we talk about expectation, variance, and value functions. A value function is essentially the conditional expectation of a random return.

---

## Probability as Long-Run Frequency

Suppose a policy has two possible actions in state $s$:

| Action | Probability |
| ------ | ----------- |
| left   | $0.3$       |
| right  | $0.7$       |

This means that if the agent visits state $s$ many times, it chooses left about $30\%$ of the time and right about $70\%$ of the time.

In RL notation:

$$
\pi(\text{left} \mid s)=0.3, \qquad \pi(\text{right} \mid s)=0.7.
$$

The vertical bar $\mid$ reads as "conditioned on." The expression $\pi(a \mid s)$ means: given that the current state is $s$, what is the probability of choosing action $a$?

---

## Conditional Probability and State Transitions

The policy can be random when it chooses actions, and the environment can also be random when it returns the next state. The vertical bar $\mid$ in $\pi(a\mid s)$ expresses this conditional relationship. To see the environment's randomness more concretely, suppose the agent executes action right:

| Next state | Probability |
| ---------- | ----------- |
| $s_1$      | $0.2$       |
| $s_2$      | $0.8$       |

This can be written as:

$$
p(s_1 \mid s, \text{right})=0.2, \qquad p(s_2 \mid s, \text{right})=0.8.
$$

It means: given that the current state is $s$ and the action is right, these are the probabilities of the possible next states.

The state transition probability in a Markov decision process is exactly this kind of conditional probability:

$$
p(s' \mid s, a).
$$

Do not rush to memorize the symbol. Read it as a sentence: **after taking action $a$ in state $s$, the probability of moving to next state $s'$**.

---

## Expectation as a Weighted Average

Consider a simple example. An action has two possible outcomes:

| Outcome | Probability | Reward |
| ------- | ----------- | ------ |
| success | $0.8$       | $10$   |
| failure | $0.2$       | $-5$   |

The average reward is not the simple average $(10-5)/2=2.5$. It is weighted by probability:

$$
\mathbb{E}[R] = 0.8 \times 10 + 0.2 \times (-5) = 7.
$$

$\mathbb{E}$ denotes expectation, and $R$ inside the brackets is the random variable. The line reads "$R$ has expectation $7$." In plain language: if we repeat this action many times, the average reward per trial approaches $7$.

The structure of this formula is:

- $\mathbb{E}$ reads as "expectation." It is not a new number; it names the operation of adding all possible outcomes after weighting them by probability.
- The brackets $[\cdot]$ contain the random variable being averaged.
- $\sum_x p(x)x$ is the expanded form of expectation: multiply every possible outcome $x$ by its probability $p(x)$, then add them all.

The state value function in reinforcement learning is also an expectation:

$$
v_\pi(s)=\mathbb{E}_\pi[G_t \mid S_t=s].
$$

The subscript $\pi$ means "under policy $\pi$," and the expression after the vertical bar is the condition. The full line reads: starting from state $s$, following policy $\pi$, what is the average future discounted return $G_t$?

## Joint Probability, Marginal Probability, and the Law of Total Probability

So far we have looked at conditional probability $p(s'\mid s,a)$: it describes which next state appears after a particular action is known. But many RL calculations need to combine conditional probabilities. For example, what is the total probability of reaching state $s'$ from state $s$? This total probability must add over all possible actions. To perform this fuller probability reasoning, we need three tools: joint probability, marginal probability, and the law of total probability.

**Joint probability** describes the probability that two events happen together. For example:

$$
P(A,B)=P(A)P(B\mid A).
$$

If state $s$ appears with probability $0.4$, and the probability of choosing action right in this state is $0.7$, then the probability of "being in $s$ and choosing right" is:

$$
P(s,\text{right})=0.4\times0.7=0.28.
$$

**Marginal probability** removes variables we do not care about by summing them out. For example, suppose next state $s'$ can be caused by two actions:

| Action | $\pi(a\mid s)$ | $p(s'\mid s,a)$ |
| ------ | -------------- | --------------- |
| left   | $0.3$          | $0.2$           |
| right  | $0.7$          | $0.8$           |

Then, under policy $\pi$, the total probability of moving from $s$ to $s'$ is:

$$
p_\pi(s'\mid s)=0.3\times0.2+0.7\times0.8=0.62.
$$

This is the law of total probability:

$$
p_\pi(s'\mid s)=\sum_a \pi(a\mid s)p(s'\mid s,a).
$$

The "sum over actions first, then sum over next states" pattern that often appears in Bellman equations is fundamentally a combination of the law of total probability and expectation.

---

## Conditional Expectation: The Mathematical Core of Value Functions

Ordinary expectation asks "what is the average?" over all possible cases. In reinforcement learning, however, we usually do not care about the average over every possible situation. We care about the average return given that the current state is $s$. This is like asking not "what is the average score in the whole school?" but "given that this student is in the advanced class, what score should we expect?" **Conditional expectation** answers the question "what is the average when a condition is already known?" It is the mathematical core of value functions.

Suppose state $s$ has two actions:

| Action | Probability | Average return after the action |
| ------ | ----------- | ------------------------------- |
| left   | $0.4$       | $3$                             |
| right  | $0.6$       | $8$                             |

If we know the current state is $s$, but the action is still sampled from the policy, then the average return starting from $s$ is:

$$
\mathbb{E}[G\mid S=s]=0.4\times3+0.6\times8=6.
$$

The state value function

$$
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
$$

is a conditional expectation. It is not the result of one trajectory. It is the average over all possible future trajectories under the condition "starting from state $s$."

Once this is clear, the randomness of value functions becomes clear as well: trajectories may differ, returns may differ, but the state value is the conditional average of those returns.

---

## Summary

This article built five basic concepts from probability theory:

| Concept                 | Definition                            | RL role                                                |
| ----------------------- | ------------------------------------- | ------------------------------------------------------ |
| Sample space            | The set of all possible outcomes      | Defines possible states and actions of the environment |
| Random variable         | Maps random outcomes to numbers       | Reward $R$, return $G$, state $S$                      |
| Probability             | Long-run frequency of an outcome      | Probability that a policy chooses an action            |
| Conditional probability | Probability given partial information | State transition $p(s'\mid s,a)$                       |
| Expectation             | Probability-weighted average          | Value function $v_\pi(s)=\mathbb{E}[G\mid s]$          |

Probability describes randomness, conditional probability describes randomness under a known condition, and expectation compresses randomness into a representative number. Together, these three tools form the mathematical foundation of value functions.

> **Next**: [E.2.2 Random Variables, Returns, and State Values](./probability-value) -- applying expectation to returns and value functions.
