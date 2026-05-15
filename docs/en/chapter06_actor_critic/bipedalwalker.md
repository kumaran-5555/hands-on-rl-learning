---
title: '6.6 Hands-On: BipedalWalker'
---

# 6.6 Hands-On: BipedalWalker (Bipedal Walking)

> **Goal of This Section**: Train `BipedalWalker-v3` with Actor-Critic and observe how the policy gradually learns to coordinate four continuous joints to achieve stable bipedal walking. This is a classic benchmark that highlights why Actor-Critic methods are strong at high-dimensional continuous control.

> **Code for This Section**: [actor_critic_bipedalwalker.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/actor_critic_bipedalwalker.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/requirements.txt)

In the previous section, Pendulum has a 1D continuous action and a 3D state. BipedalWalker increases the complexity by an order of magnitude: a 24D state (joint angles, angular velocities, ground-contact sensors, and more) and a 4D continuous action (two joints per leg: hip and knee). The goal is simple to state but hard to achieve: teach a two-legged robot to walk forward without falling.

## Environment: BipedalWalker-v3

```
        O          ← head
       /|\
      / | \        ← torso
     /  |  \
    🔶   🔶       ← hip joints
    |     |        ← thighs
    🔷   🔷       ← knee joints
    |     |        ← shins
   ___   ___       ← feet
```

| Property     | Value                                                                           |
| ------------ | ------------------------------------------------------------------------------- |
| State dim    | 24 (torso angle, angular velocity, joint states, 10 lidar-like distance rays)   |
| Action dim   | 4 (torques at left hip, left knee, right hip, right knee; continuous $[-1, 1]$) |
| Reward       | forward progress + survival term - energy cost                                  |
| Termination  | falling (head touches the ground) or reaching the end                           |
| "Solved" tag | average return > 300                                                            |

## Run Training

```bash
pip install -r code/chapter06_actor_critic/requirements.txt
python code/chapter06_actor_critic/actor_critic_bipedalwalker.py
```

BipedalWalker is substantially harder than Pendulum and CartPole. Training often takes 1000-3000 episodes. A practical success signal is that the robot can walk stably without falling, and the average return exceeds 300.

## From Pendulum to BipedalWalker

|                | Pendulum             | BipedalWalker                              |
| -------------- | -------------------- | ------------------------------------------ |
| State dim      | 3                    | 24                                         |
| Action dim     | 1                    | 4                                          |
| Training time  | a few minutes        | tens of minutes to a few hours             |
| "Solved" bar   | return near 0        | return > 300                               |
| Main challenge | single-joint control | multi-joint coordination + dynamic balance |

The real challenge of BipedalWalker is **coordination**. All four joints must apply force in the right way at the right time. A single joint moving out of sync can destabilize the body and cause a fall.

This is exactly where Actor-Critic shines:

- The Actor can output a 4D continuous action simultaneously (one value per joint).
- The Critic evaluates how good the overall state is, providing learning signals that reflect the global outcome (walking forward, staying upright, and spending less energy).

As the two networks improve together, the policy gradually discovers coordinated motion patterns that look like walking.

## Chapter Summary

This chapter started from the high-variance problem of REINFORCE and introduced the Actor-Critic architecture: use a Critic network to estimate $V(s)$ and provide a lower-variance advantage signal, while the Actor network updates the policy that actually makes decisions.

From CartPole (discrete actions), to Pendulum (1D continuous control), and then to BipedalWalker (4D continuous control), we saw that Actor-Critic remains effective as task complexity increases.

In the next chapter, we will address another issue of Actor-Critic, namely training instability, and motivate PPO: [Chapter 7: PPO](../chapter07_ppo/intro).
