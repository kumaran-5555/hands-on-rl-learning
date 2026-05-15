---
title: E.4.1 Entropy and Exploration
---

# E.4.1 Information Theory Basics: Self-Information, Entropy, and Exploration

> **Prerequisites**: This article does not require prior knowledge of information theory, but it is recommended that you first read the [appendix introduction](./intro) and [E.2.1 Probability Basics](./probability-basics).

---

## What Question Is Information Theory Answering?

In reinforcement learning, a policy is essentially a probability distribution. Given a state $s$, the policy assigns a probability to each action:

$$
\pi(\cdot\mid s).
$$

Once the policy becomes a probability distribution, three questions naturally appear:

1. How random is this policy? (Answer: entropy.)
2. How far is the model's predicted distribution from the target distribution? (Answer: cross-entropy.)
3. How far is the new policy from the old policy? (Answer: KL divergence.)

These three quantities run through the whole path from policy gradient to DPO. We will start from the simplest probability event and move step by step toward their use in RL.

---

## The Smaller the Probability, the Larger the Information

Imagine a number-guessing game: the other person chooses a number from 1 to 8, and you need to identify it using yes-or-no questions. If the chosen number is 3, how many questions do you need? "Is it less than 4?" "Is it greater than 2?" "Is it 3?" Exactly 3. If the range were only 1 to 2, one question would be enough.

The larger the range, the harder the outcome is to guess, and the more information you gain after guessing it correctly. This is the intuition behind self-information: **the less likely an event is, the more "surprise" it brings when it occurs**.

Mathematically, self-information uses a compact formula to describe this degree of surprise:

$$
I(x)=-\log_2 p(x).
$$

Here $\log_2$ is the logarithm with base 2, and $p(x)$ is the probability that event $x$ occurs. The minus sign makes the result positive, because the logarithm of a number smaller than 1 is negative.

For example, if $p(x)=1/2$, as in a fair coin landing heads:

$$
I(x)=-\log_2(1/2)=1.
$$

If $p(x)=1/8$, as in guessing one specific number in the game above:

$$
I(x)=-\log_2(1/8)=3.
$$

When the probability drops from $1/2$ to $1/8$, self-information rises from 1 to 3. This property, "rarer means more surprising," is the foundation for entropy and KL divergence later.

---

## Entropy: Measuring How Undecided a Policy Is

Self-information measures the surprise of a single event, but a policy faces an entire set of action probabilities. We need one overall quantity to measure "how uncertain the policy is as a whole." This is the problem entropy solves.

Entropy, denoted by $H$, is the weighted average of the self-information of all actions:

$$
H(P)=-\sum_x p(x)\log_2 p(x).
$$

Here $\sum_x$ means "sum over all possible actions $x$," and each term is weighted by its corresponding probability $p(x)$.

Consider two concrete policies. Policy A is completely undecided:

| Action | Probability |
| ------ | ----------- |
| left   | $0.5$       |
| right  | $0.5$       |

Policy B has almost made up its mind:

| Action | Probability |
| ------ | ----------- |
| left   | $0.9$       |
| right  | $0.1$       |

The entropy of policy A is:

$$
H(A)=-0.5\log_2 0.5-0.5\log_2 0.5=1.
$$

The entropy of policy B is approximately:

$$
H(B)=-0.9\log_2 0.9-0.1\log_2 0.1\approx0.47.
$$

A has higher entropy, which means it is more undecided and more willing to explore different actions. B has lower entropy, which means it leans toward one action and behaves more predictably.

---

## Entropy Bonus: Keeping the Policy from Deciding Too Early

Knowing how to compute entropy is not enough; the key is what it can do. In practice, a policy can easily lock onto one action too early. This is premature convergence. To address it, we add an entropy bonus to the training objective, encouraging the policy to keep an open mind early in training.

One common countermeasure is to add an entropy bonus to the training objective:

$$
J(\pi)=\mathbb{E}[G]+\beta H(\pi).
$$

Here $\mathbb{E}[G]$ is the expected return, $\mathbb{E}$ denotes an average, $H(\pi)$ is the entropy of the policy, and $\beta$ is a coefficient that controls how strongly we encourage exploration.

Consider a numerical example. Two policies have the following average returns and entropies:

| Policy | Average return | Entropy | Objective when $\beta=0.5$ |
| ------ | -------------- | ------- | -------------------------- |
| A      | $10$           | $1.0$   | $10+0.5\times1=10.5$       |
| B      | $10.3$         | $0.1$   | $10.3+0.5\times0.1=10.35$  |

B has the higher raw return, but A is more willing to explore. After adding the entropy bonus, A's combined score exceeds B's.

The reason is simple: early in training, value estimates are still inaccurate. Rather than letting the policy bet everything on one or two actions, it is better to try several choices. Once the value estimates become reliable, the policy will naturally become more certain.

---

## Units of Entropy: Bit and Nat

The entropy formula above uses a logarithm, and the base of the logarithm determines the unit of entropy. If we use $\log_2$, the unit is called a bit. If we use the natural logarithm $\ln$, with base $e \approx 2.718$, the unit is called a nat.

The two differ by a constant factor: $1\text{ nat} = \log_2 e \approx 1.443\text{ bit}$. Machine learning papers and code almost always use natural logarithms, because they match the mathematical forms of exponentials, softmax, and cross-entropy loss.

For example, the entropy of a fair coin using $\log_2$ is:

$$
H=-0.5\log_2 0.5-0.5\log_2 0.5=1\text{ bit}.
$$

Using the natural logarithm:

$$
H=-0.5\ln0.5-0.5\ln0.5\approx0.693\text{ nat}.
$$

The numerical values differ, but they express the same physical quantity with different units, just like 1 kilometer and 1000 meters. Unless stated otherwise, we will use natural logarithms from now on.

---

## Why Maximum Entropy Corresponds to the Uniform Distribution

A natural follow-up question is: when is entropy largest? The answer is intuitive: when all options have equal probability.

If the probabilities of two actions are $[0.5,0.5]$, we cannot guess which one the policy will choose. If they are $[0.99,0.01]$, we can almost be sure it will choose the first one. Entropy captures exactly this degree of unpredictability:

$$
H([0.5,0.5])=1\text{ bit},
$$

while

$$
H([0.9,0.1])\approx0.47\text{ bit}.
$$

So encouraging entropy does not mean encouraging the policy to act randomly forever. It means keeping the policy open early in training so it tries different choices. Once value estimates become more reliable, the policy naturally narrows toward high-return actions.

---

## From Softmax to Policy Distributions

So far, we have assumed that the policy probability distribution is known. In practice, however, the policy is a neural network. It does not output probabilities directly; it outputs a set of raw scores. We need a way to convert these scores into a valid probability distribution. The softmax function is introduced to solve exactly this problem.

For example, for two actions, the network might output logits $z=[2,1]$. The softmax function converts these scores into valid probabilities: every value lies between 0 and 1, and all values sum to 1.

$$
\pi(a_i\mid s)=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Here $e^{z_i}$ is an exponential, and the denominator $\sum_j e^{z_j}$ sums the exponentials of all actions for normalization. The result is:

$$
\pi(a_1\mid s)=\frac{e^2}{e^2+e^1}\approx0.73,
\qquad
\pi(a_2\mid s)\approx0.27.
$$

Softmax is the most common final layer for policy networks and language models. All later uses of entropy, cross-entropy, and KL divergence will act on probability distributions produced by softmax. Once you understand this layer, it becomes easier to see why policy gradient, PPO, and DPO repeatedly compare the log probabilities of different actions or different answers.

---

## Summary

This article started from "how surprising is a probability event?" and gradually built the core information-theoretic toolchain used in RL:

| Concept          | Problem it solves                                                    | Core formula                                    | Role in RL                                          |
| ---------------- | -------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------- |
| Self-information | How much "surprise" a single event contains                          | $I(x)=-\log p(x)$                               | Low-probability events contain more information     |
| Entropy          | How uncertain the policy is overall                                  | $H(P)=-\sum_x p(x)\log p(x)$                    | Measures exploration and encourages diversity       |
| Entropy bonus    | Preventing the policy from locking onto one action too early         | $J=\mathbb{E}[G]+\beta H(\pi)$                  | Adds an exploration incentive to the objective      |
| Maximum entropy  | What the distribution looks like when entropy is largest             | Entropy is largest for the uniform distribution | "Encouraging entropy" means encouraging openness    |
| Softmax          | Turning neural network outputs into a valid probability distribution | $\pi(a_i\mid s)=e^{z_i}/\sum_j e^{z_j}$         | Output layer of policy networks and language models |

This article discussed only the properties of a single distribution. The next article compares two distributions, such as model predictions vs. true labels or a new policy vs. an old policy. For that, we need cross-entropy and KL divergence.

> **Next**: [E.4.2 Cross-Entropy and KL Divergence](./information-cross-entropy-kl) -- measuring the distance between two distributions.
