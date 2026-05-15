---
title: 'Chapter 5: Policy-Based Methods'
---

# Chapter 5: Policy-Based Methods: Policy Gradients and REINFORCE

In Chapter 4, we followed Route 1: learn $Q(s,a)$ to score each action, then pick the action with the highest score (review: [Q(s,a) and the greedy policy](../chapter03_mdp/value-q)). DQN performs well on CartPole and Atari, but it has a fundamental limitation:

**it can only handle a finite set of discrete actions.**

CartPole has only two choices, "push left" and "push right". DQN computes a $Q$ value for each action and takes the maximum. But what if we want to control a robotic arm? The shoulder, elbow, and wrist joints each have multiple degrees of freedom, and each degree can apply a continuous torque. The set of possible action combinations is infinite, so it is impossible to compute a $Q$ value for every combination. The situation is even more obvious in large language model text generation: at every step we sample from tens of thousands of tokens. The policy itself is a continuous probability distribution, so the $\arg\max$ mindset does not really apply.

## Learning The Policy Directly

If "score first, then choose" is not viable, we take a different route:

**skip $Q$ values and learn the policy $\pi_\theta(a|s)$ directly.**

Instead of asking "how many points is each action worth?", we learn "what to do in what situation".

This is exactly the core idea of Chapter 3's [Route 2: the policy objective $J(\theta)$](../chapter03_mdp/policy-objective): define a policy objective function $J(\theta)$, then directly optimize the parameters $\theta$ to maximize $J(\theta)$. In this chapter, we will go deeper along this route, moving from the policy gradient theorem to the REINFORCE algorithm, and then to variance reduction via baselines.

::: tip Prerequisites (Quick Review)
We will repeatedly use the following concepts in this chapter. If any of them feels fuzzy, click through for a quick refresh before continuing:

- [Policy $\pi_\theta$ and objective $J(\theta)$](../chapter03_mdp/policy-objective): how do we represent a parameterized policy, and how do we define "how good" a policy is?
- [Monte Carlo (MC) methods](../chapter03_mdp/dp-mc-td): "finish an entire episode, then look back", the sampling foundation behind REINFORCE
- [State value $V(s)$](../chapter03_mdp/value-bellman): "how many points on average can I get?", the source of baselines
  :::

## Main Thread Of This Chapter

We will develop the chapter along two parallel threads. The first is **theory**: from the policy gradient theorem to the REINFORCE algorithm, then to baseline variance reduction and the advantage function. The second is **practice**: we will first get vanilla REINFORCE running on CartPole, observe the high-variance behavior, and then add a value baseline to compare the results.

| Section                                              | Core Question                                                                          |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [Why Policy Gradients Are Necessary](./pg-necessity) | Where does DQN's $\arg\max$ break down? Why learn the policy directly?                 |
| [The REINFORCE Algorithm](./reinforce)               | What is the mathematical form of the policy gradient theorem? How implement REINFORCE? |
| [Hands-on: CartPole](./cartpole)                     | How does REINFORCE perform on a real control task? What does high variance look like?  |
| [Improving Policy Gradients](./pg-improvements)      | Why do baselines reduce variance? Why is $V(s)$ an optimal baseline?                   |
| [Hands-on: CartPole Ablation](./cartpole-baseline)   | What is the practical effect of a value baseline? Look from reward and variance.       |

Let's begin with the motivation for policy gradients: [from DQN to policy gradients](./pg-necessity).
