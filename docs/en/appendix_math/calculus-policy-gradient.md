---
title: E.3.2 Policy Gradient
---

# E.3.2 Policy Gradients and Advantage Functions

> **Prerequisite**: [E.3.1 Derivatives, Gradients, and the Chain Rule](./calculus-basics). You need to know what a gradient is.

---

## Where Policy Gradients Come From

In the previous section, we learned how to use gradients to adjust parameters and optimize an objective function. Now we apply that idea to reinforcement learning. The core idea of policy gradients can be understood in one sentence: **if an action leads to a high return, increase its probability in the corresponding state; if it leads to a low return, decrease its probability**.

Suppose that in some state $s$, the policy chooses action right with probability

$$
\pi_\theta(\text{right}\mid s)=0.2.
$$

This time the agent chose right and eventually received return $G=10$. As an analogy, if a basketball player makes a shot from a certain position, the coach will encourage the player to shoot more often from that position. Policy gradients do something similar.

A common form of the policy gradient is

$$
\nabla_\theta J(\theta) \approx G_t \nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

It has two parts:

- $G_t$ (the cumulative return): how good the outcome was, which determines the strength and direction of the update.
- $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ (the gradient of the log probability): how to adjust the parameters so that this action becomes more likely, or less likely, to occur.

If $G_t$ is positive, we update in the direction that increases the probability of this action. If $G_t$ is negative, we update in the opposite direction.

---

## Advantage Functions: A More Precise Judgment Than "Good or Bad"

Policy gradients give us the broad direction: increase the probabilities of good actions and decrease the probabilities of bad actions. But there is a practical problem. If we use the raw return $G_t$ as the signal, every action with a positive return will be encouraged, even when some of those actions merely happened to get a positive score and are not actually better than average. This is like an exam where the class average is 90 and a student scores 80. The score is not bad in absolute terms, but relatively speaking it is not worth praising. The advantage function is introduced to solve this problem of relative quality.

$$
V(s)=8.
$$

This action received return $10$. It is not merely "good"; it is $2$ better than the average level:

$$
A(s,a)=10-8=2.
$$

If another action receives $6$, then even though the return is still positive, it is below average:

$$
A(s,a)=6-8=-2.
$$

The advantage function answers the question: **relative to the average level in the current state, is this action better or worse?**

In policy gradients, we often replace $G_t$ with an advantage estimate $\hat{A}_t$, so the information about "how much better" or "how much worse" becomes more precise:

$$
\nabla_\theta J(\theta) \approx \hat{A}_t \nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

Replacing the raw return with the advantage function can greatly reduce the variance of the gradient estimate and make training more stable.

---

## Summary

This page applied gradient tools to policy optimization:

| Concept             | Formula                                                                      | Role                                              |
| ------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------- |
| Policy gradient     | $\nabla_\theta J \approx G_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)$       | Good outcomes increase the probability of actions |
| Advantage function  | $A(s,a)=R-V(s)$                                                              | Turns "absolute quality" into "relative quality"  |
| Advantage weighting | $\nabla_\theta J \approx \hat{A}_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)$ | Actions better than average are strengthened      |

Policy gradients provide the direction, and advantage functions make the signal more precise. But one unresolved question remains: how large should each update step be? The next page discusses how PPO clipping and the Adam optimizer control update size.

> **Next**: [E.3.3 PPO Clipping and Adam](./calculus-ppo), which controls policy update size and gradient noise.
