---
title: 4. Deep Q-Networks
---

# Deep Q-Networks

## Chapter Guide

**Core ideas**

- Understand why tabular Q-Learning only works for small, enumerable discrete state spaces, and why it breaks for continuous control and pixel-based games.
- Learn how Deep Q-Networks (DQN) approximate $Q(s,a)$ with a neural network, turning a Q-table into a generalizable function.
- See why **experience replay** and the **target network** are not optional engineering tricks, but the key reasons DQN is trainable.
- Through LunarLander and visual game tasks, develop intuition for the stability, exploration, and function-approximation issues that appear once we move beyond low-dimensional state vectors.

**Key formulas**

$$
Q(s,a;\theta)\approx Q^*(s,a)
\quad \text{(Function approximation: replace a Q-table with a neural network)}
$$

$$
y = r+\gamma\max_{a'}Q(s',a';\theta^-)
\quad \text{(DQN TD target: estimate next-step optimal value with a target network)}
$$

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}
\left[
\left(
r+\gamma\max_{a'}Q(s',a';\theta^-)
-Q(s,a;\theta)
\right)^2
\right]
\quad \text{(DQN loss: minimize the squared TD error)}
$$

**What these formulas are doing**

Chapter 4 continues along the **$Q(s,a)$ line** from Chapter 3. The first formula states the central substitution: we no longer store one number per state-action pair. Instead, we output action values from a network with parameters $\theta$. The second formula keeps the TD idea from Q-Learning: the immediate reward $r$ plus the discounted return of the best action in the next state. The third formula turns that TD error into an optimization objective a neural network can actually train on, where $\mathcal{D}$ is the replay buffer and $\theta^-$ denotes the target network parameters.

So the chapter is not about changing the RL goal. It is about converting the Chapter 3 action-value estimation idea into a deep learning problem that can handle high-dimensional states.

At the end of Chapter 3, we already had the Q-Learning intuition:

1. the agent takes action $a$ in state $s$
2. it observes reward $r$ and transitions to $s'$
3. it corrects $Q(s,a)$ using a TD target

In a small GridWorld, this is almost too natural. With a small number of states and actions, we can keep a tiny table, update one cell per step, and watch the values propagate backward from the goal.

The real issue is a hidden assumption: **the state space must be small enough to enumerate, name, and revisit**. LunarLander's state is an 8D continuous vector. Atari states are frames of pixels. Tiny differences in position, velocity, and pixels produce new states. At that point, the problem is no longer "how do I write the TD target", but "should we still keep a separate table entry for each possible state?"

Look at it from another angle: what is truly valuable in Q-Learning is not the table, but the idea of "use current experience to correct long-term value estimates." DQN keeps that idea and replaces the table with a neural network. Similar states can now share parameters and share learning through generalization.

But that substitution does not automatically work. Supervised learning typically has fixed labels. In DQN, the "label" (the TD target) depends on the model's own estimates of the future, and consecutive samples are highly correlated along a trajectory. DQN therefore solves two linked problems:

1. **the table does not scale**
2. **training the network is unstable**

The chapter's main line can be summarized as:

**How do we extend tabular Q-Learning into a deep RL algorithm that can handle continuous state spaces, pixel inputs, and unstable bootstrapped targets?**

We will first locate the boundary of Q-tables, then unpack DQN's three components, then run LunarLander end-to-end, and finally discuss DQN variants (Double, Dueling, PER, Rainbow) and a full project path toward visual games.

## Chapter Map

| Section                                                             | Main question                                                                                              |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [Why DQN is needed](./from-q-to-dqn)                                | Why can't the Q-table scale? What changes when we replace it with a network?                               |
| [DQN architecture](./dqn-components)                                | What problems do the Q-network, replay buffer, and target network each solve? How do formulas map to code? |
| [Hands-on: LunarLander](./lunar-lander)                             | How do we train, analyze, and evaluate DQN on a low-dimensional control task?                              |
| [The DQN family](./dqn-family)                                      | What do Double/Dueling/Rainbow-style improvements fix?                                                     |
| [Project: from LunarLander to visual games](./visual-game-projects) | How do we move from vector states to pixel inputs and more complex game settings?                          |

## Learning Objectives

After finishing this chapter, you should be able to:

- explain why tabular Q-Learning is not viable for continuous states and pixel observations
- explain how DQN approximates $Q(s,a)$ with a neural network
- write down the DQN TD target and the MSE TD-error loss
- explain why replay buffers and target networks stabilize training
- trace one full parameter update step: batch sampling → forward pass → target computation → loss → backprop → parameter update
- distinguish training vs evaluation modes (epsilon-greedy exploration vs greedy evaluation)
- recognize the main ideas behind common DQN variants and what failure modes they target

Next: [Why DQN is needed](./from-q-to-dqn).
