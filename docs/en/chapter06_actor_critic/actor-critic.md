---
title: 6.3 Actor-Critic Architecture
---

# 6.3 Actor-Critic Architecture

In the previous two sections, we met the [advantage function](./advantage-function) $A(s,a)$ and the [training method for the Critic](./critic-training). Now let's assemble all the parts and see how the Actor and the Critic collaborate.

::: tip Prerequisites for This Section

- [Advantage function $A(s,a) = Q(s,a) - V(s)$](./advantage-function): "How much better is this action than the average?"
- [TD Error $\delta = r + \gamma V(s') - V(s)$](./critic-training): a practical estimate of the advantage
- [Policy gradient $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$](../chapter05_policy_gradient/reinforce): the Actor's update formula
- [REINFORCE and baselines](../chapter05_policy_gradient/pg-improvements): why we move from $G_t$ to $G_t - V(s)$
  :::

## From REINFORCE to Actor-Critic

Recall the gradient formula of REINFORCE from Chapter 5 (review: [policy gradient theorem](../chapter05_policy_gradient/reinforce)):

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

$G_t$ is the cumulative return over the full trajectory. This is exactly why REINFORCE has high variance. The [baseline analysis](../chapter05_policy_gradient/pg-improvements) in Chapter 5 tells us that subtracting $V(s)$ can reduce variance. In the previous section, we also found that we do not need to wait until the end of the episode. Using the [TD Error](./critic-training) $\delta = r + \gamma V(s') - V(s)$, we can replace $G_t - V(s)$ as an advantage estimate:

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta$$

This substitution changes the nature of the algorithm:

|                    | REINFORCE                      | Actor-Critic                                                           |
| ------------------ | ------------------------------ | ---------------------------------------------------------------------- |
| Advantage estimate | $G_t$ (MC, needs full episode) | $\delta = r + \gamma V(s') - V(s)$ (TD, update after one step)         |
| Update timing      | after the episode ends         | after every step                                                       |
| Variance           | high                           | low                                                                    |
| Bias               | unbiased                       | biased (bias introduced by [bootstrapping](../chapter03_mdp/dp-mc-td)) |
| Cost               | none                           | must train a Critic                                                    |

## Actor-Critic Architecture

If we integrate the advantage function with Critic training, we obtain one of the most classic architectures in reinforcement learning. The Actor is responsible for choosing actions; the Critic is responsible for evaluating how good those actions are. The two work together through the advantage function $A(s,a)$:

```
Actor-Critic Data Flow

  state s
    │
    ├──→ Actor (policy network)
    │      π(a|s) → choose action a
    │                  │
    │              execute action a
    │                  │
    │                  ▼
    │            environment → returns r, s'
    │                  │
    ├──→ Critic (value network) │
    │      V(s)  ───────────────┤
    │      V(s') ───────────────┤
    │                           │
    │      δ = r + γV(s') - V(s)
    │            │
    │            ▼
    │      Actor update:  θ ← θ + α·∇log π(a|s)·δ
    │      Critic update: V(s) ← V(s) + α·δ
    │
    └──→ next step, repeat
```

Both networks share the same input (state $s$), but each has a different job:

| Network | Role               | Input     | Output                           | Learning objective               |
| ------- | ------------------ | --------- | -------------------------------- | -------------------------------- |
| Actor   | choose action      | state $s$ | action probabilities $\pi(a\|s)$ | maximize cumulative reward       |
| Critic  | evaluate situation | state $s$ | value estimate $V(s)$            | predict future return accurately |

If you look carefully at the Critic update rule, $V(s) \leftarrow V(s) + \alpha \cdot \delta$, isn't this exactly [TD learning](../chapter03_mdp/dp-mc-td) from Chapter 3? **The Critic is, in essence, a neural-network implementation of the value function $V(s)$ from Chapter 3** (see: [value function $V(s)$](../chapter03_mdp/value-bellman)). It independently learns "how many points each state is worth." The Actor is a neural-network implementation of the [policy $\pi(a|s)$](../chapter03_mdp/policy-objective): it adjusts its behavior based on the evaluation provided by the Critic.

Two function approximators work together: the Critic helps the Actor judge "how much better this action is than the average," and the Actor updates its policy based on that judgment. The new policy then generates new data, which in turn helps the Critic learn better. This feedback loop is where the name Actor-Critic comes from.

### Implementing Actor-Critic in PyTorch

Compared with REINFORCE, Actor-Critic adds a Critic network, but the overall structure remains quite clear:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# ==========================================
# 1. Actor-Critic network (shared feature extractor)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # Actor head: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic head: outputs state value
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

# ==========================================
# 2. Training loop (update every step; no need to wait for episode end)
# ==========================================
env = gym.make("CartPole-v1")
model = ActorCritic(state_dim=4, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99

reward_history = []

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    while True:
        state_t = torch.FloatTensor(state)

        # Actor chooses action; Critic evaluates state
        probs, value = model(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        # Critic evaluates the next state
        with torch.no_grad():
            _, next_value = model(torch.FloatTensor(next_state))
            next_value = 0 if done else next_value

        # TD Error = advantage estimate (review: Section 6.1 A ≈ δ)
        td_target = reward + gamma * next_value
        td_error = td_target - value

        # Actor loss: policy gradient × advantage
        actor_loss = -log_prob * td_error.detach()

        # Critic loss: make V(s) close to TD target (review: Section 6.2 L = δ²)
        critic_loss = td_error.pow(2)

        # Total loss
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break

    reward_history.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {episode+1} | Avg Reward: {avg:.1f}")
```

Compared with the REINFORCE code in Chapter 5, the key differences are: there is an additional Critic network (outputting $V(s)$); TD Error (`td_target - value`) replaces $G_t$; the Critic has its own loss function (MSE); and we do not need to finish an episode before updating.

### Actor-Critic Training Curve on CartPole

```
Training Curve of Actor-Critic on CartPole

 500 ┤
     │                              ━━━━━━━━━━━━━━━
 400 ┤                         ━━━━
     │                    ━━━━
 300 ┤              ━━━━━
     │         ━━━━
 200 ┤    ━━━━
     │ ━━
 100 ┤╱
     └────────────────────────────────────────────
     0    50   100  150  200  250  300  350  400  450  500
                    Episode

 Compare with the typical curve of REINFORCE (more jagged, slower convergence)
```

On CartPole, Actor-Critic typically stabilizes at 500 points (the maximum) within about 200-300 episodes, whereas REINFORCE may need 500+ episodes and shows a visibly jagged curve. This is the payoff of "trading bias for variance": every step provides a more stable gradient signal, and policy updates are no longer driven by luck.

## Further Evolution of Actor-Critic

Actor-Critic is not the destination; it is a skeleton. In later chapters, you will see various extensions:

| Chapter                                                              | Variant                          | Key improvement                                                                                  |
| -------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| [Chapter 7 PPO](../chapter07_ppo/intro)                              | PPO-Clip                         | limit the size of policy updates to avoid "taking steps that are too big"                        |
| [Chapter 7 GAE](../chapter07_ppo/gae-reward-model)                   | generalized advantage estimation | exponentially weighted sum of multi-step TD errors; precisely control the bias-variance tradeoff |
| [Chapter 9 DPO](../chapter09_alignment/intro)                        | implicit Actor-Critic            | replace the Critic with preference data; remove the on-policy constraint                         |
| [Chapter 9 GRPO](../chapter09_grpo_rlvr/grpo-practice-and-mechanism) | remove the Critic                | replace $V(s)$ with an in-group mean; save one network                                           |

All variants share the same skeleton: one network that chooses, plus one signal that evaluates. What changes is only "where the evaluation signal comes from" and "how the selection network is updated."

<details>
<summary>Question to think about: if Actor-Critic is better than REINFORCE, why not use a pure Critic (only V)?</summary>

Because with only a Critic, we cannot directly output a policy. The Critic learns $V(s)$ or $Q(s,a)$. To derive a policy, we need $\arg\max_a Q(s,a)$ (review: [greedy optimal policy](../chapter03_mdp/value-q)). But in a continuous action space, this $\arg\max$ generally has no closed-form solution. You cannot compare infinitely many continuous values one by one.

The Actor matters because it outputs action probabilities (or action parameters) directly, and thus naturally handles continuous action spaces. This is why we need two networks: the Critic provides "evaluation," and the Actor provides "choice." Both are necessary.

</details>

<details>
<summary>Question to think about: where does the "bias" in Actor-Critic come from, and is it harmful?</summary>

The bias comes from the Critic's [bootstrapping](../chapter03_mdp/dp-mc-td): the Critic uses its own estimate $V(s')$ to update $V(s)$. If $V(s')$ is inaccurate, the error can propagate backward. It is like using an inaccurate ruler to calibrate another ruler: the error accumulates.

But this bias is not necessarily bad. A moderate amount of bias can buy a much lower variance, and overall it can converge faster than the unbiased but high-variance REINFORCE. In Chapter 7, GAE is exactly about controlling this "bias-variance tradeoff": with a parameter $\lambda$, it smoothly interpolates between pure TD (high bias, low variance) and pure MC (unbiased, high variance).

</details>

Now let's look at how Actor-Critic performs in large-scale applications: [Frontiers of large-scale Actor-Critic applications](./ac-frontier).

---

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
