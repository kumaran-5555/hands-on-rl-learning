---
title: E.4.4 Mutual Information
---

# E.4.4 Mutual Information and Representation Learning

> **Prerequisites**: [E.4.1 Entropy](./information-basics) and [E.4.2 KL Divergence](./information-cross-entropy-kl). You need to know the definitions of entropy and KL divergence.

---

The previous three sections discussed the properties of a single distribution, entropy, and the distance between two distributions, KL divergence. But sometimes we want to ask a subtler question: how much information do two random variables share? This requires **mutual information**. It measures how much the uncertainty of one variable decreases after we know another variable, and in representation learning it is used to evaluate whether a state representation keeps task-relevant information.

## Mutual Information: How Much Knowing One Variable Helps

The central question mutual information answers is: after knowing $Y$, how much does the uncertainty of $X$ decrease? The formula is:

$$
I(X;Y)=H(X)-H(X\mid Y).
$$

Here $H(X)$ is the uncertainty of $X$ itself, namely its entropy, and $H(X\mid Y)$ is "how much uncertainty remains in $X$ after $Y$ is known." The difference is the uncertainty removed because we know $Y$.

Consider an intuitive example. Suppose $X$ is "whether the next step succeeds," and $Y$ is "the current state representation." If we do not know the state representation, success and failure are equally likely:

$$
H(X)=1\text{ bit}.
$$

After knowing the state representation, we can almost determine whether the step will succeed, and the uncertainty drops to:

$$
H(X\mid Y)=0.2\text{ bit}.
$$

Then the mutual information is:

$$
I(X;Y)=1-0.2=0.8\text{ bit}.
$$

$0.8$ bit means that $Y$ removes $80\%$ of the uncertainty in $X$. It helps a great deal in predicting $X$.

Mutual information can also be defined using KL divergence:

$$
I(X;Y)=D_{KL}(P_{XY}\|P_XP_Y).
$$

This form says that mutual information is the KL divergence between the joint distribution $P_{XY}$ of $X$ and $Y$ and the distribution $P_XP_Y$ that would hold if they were independent. If $X$ and $Y$ are truly independent, these two distributions are the same and the mutual information is $0$.

In reinforcement learning, mutual information is often used in representation learning. A good state representation $\phi(s)$ should preserve information related to future return while discarding task-irrelevant noise:

$$
I(\phi(s);G_t)\ \text{larger means the representation contains more return-relevant information.}
$$

Formulas like this do not necessarily appear directly in basic algorithms, but they are common in research on exploration, representation learning, world models, and unsupervised RL.

---

## Summary

This article introduced a tool for measuring how much information two random variables share:

| Concept                             | Problem it solves                                    | Core formula                    | Role in RL                                                       |
| ----------------------------------- | ---------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------- |
| Mutual information                  | How much one variable reduces uncertainty in another | $I(X;Y)=H(X)-H(X\mid Y)$        | Evaluates whether representations keep task-relevant information |
| KL definition of mutual information | Expressing mutual information with KL divergence     | $I(X;Y)=D_{KL}(P_{XY}\|P_XP_Y)$ | Mutual information is 0 under independence                       |

Mutual information connects the entropy and KL divergence from the previous articles: it uses KL to measure the difference between a joint distribution and the independence assumption, and it uses the reduction in entropy to measure information gain. The next article summarizes all complete formulas in module E.4.

> **Next**: [E.4.5 Complete Information Theory Formulas](./information-advanced-formulas) -- full expressions for KL, RLHF, DPO, and mutual information.
