---
title: '6.5 Hands-on: Pendulum Swing-Up and Balance'
---

# 6.5 Hands-on: Pendulum Swing-Up and Balance

> **Goal of this section**: Train `Pendulum-v1` with Actor-Critic, understand how a policy network outputs **continuous actions** (via a Gaussian distribution), and experience why AC is a natural fit for continuous control.

> **Code for this section**: [actor_critic_pendulum.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/actor_critic_pendulum.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/requirements.txt)

In the previous chapters, our experiments all used **discrete actions**: CartPole is "left/right", LunarLander chooses among four thrusters, and so on. But the real strength of Actor-Critic shows up when the action space becomes **continuous**, which is exactly where DQN starts to break down.

`Pendulum-v1` is one of the simplest continuous-control tasks. A rod is attached to a pivot, and the agent can apply a **continuous torque** in $[-2, 2]$ to swing the rod up to the upright position and keep it there. The state is only 3-dimensional ($\cos\theta$, $\sin\theta$, angular velocity), and the action is only 1-dimensional (torque), but that single dimension is continuous. That continuity is the key difficulty: you cannot compute a $Q$ value for every torque and then take $\arg\max$.

## A Policy Network For Continuous Actions

For discrete actions, the policy network outputs Softmax probabilities. For continuous actions, the policy network typically outputs the parameters of a **Gaussian distribution**, namely the mean $\mu$ and the standard deviation $\sigma$:

```python
class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head: output the mean of the Gaussian
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        # Actor head: output log std (use log to ensure std > 0)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic head: output V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        mu = torch.tanh(self.mu_head(features)) * 2  # scale to [-2, 2]
        std = torch.exp(self.log_std)
        value = self.critic_head(features)
        return mu, std, value
```

The key difference is this: the Actor no longer outputs a probability vector over discrete actions. Instead, it outputs $\mu$ and $\sigma$ for a Gaussian distribution. An action is sampled from $\mathcal{N}(\mu, \sigma^2)$, and then passed through `tanh` to squash it into $[-2, 2]$.

## Run Training

```bash
pip install -r code/chapter06_actor_critic/requirements.txt
python code/chapter06_actor_critic/actor_critic_pendulum.py
```

Training typically converges within 200 to 300 episodes. A practical sign of success is that the rod can stay steadily near the upright position: the return rises from around -1000 (a random policy) to close to 0 (a near-optimal policy; in Pendulum, the maximum achievable reward is 0).

## Why DQN Cannot Handle Pendulum Well

DQN selects actions by computing $\arg\max_a Q(s,a)$. But in Pendulum, the action is a continuous value in $[-2, 2]$. There are infinitely many candidates, so you cannot compare them one by one.

A common workaround is to discretize $[-2, 2]$ into, say, 20 bins, but the resulting loss of precision is usually severe.

Actor-Critic, by contrast, samples actions directly from $\mathcal{N}(\mu, \sigma^2)$, which makes it naturally suitable for continuous spaces. This is exactly what we meant at the beginning of the chapter by "the Actor directly outputs an action distribution, which makes it naturally applicable to continuous action spaces."

In the next section, we will challenge a more complex continuous-control task: [Hands-on: BipedalWalker Locomotion](./bipedalwalker).
