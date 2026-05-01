import random

import torch
import torch.nn as nn


torch.manual_seed(2)
random.seed(2)


def target_action(state):
    return 0.35 * torch.sin(2.5 * state)


def behavior_mean(state):
    return 0.20 * target_action(state)


def reward_fn(state, action):
    return -((action - target_action(state)) ** 2)


def make_dataset(num_samples=6000):
    state = torch.empty(num_samples, 1).uniform_(-1.0, 1.0)
    action = behavior_mean(state) + 0.04 * torch.randn_like(state)
    action = action.clamp(-1.0, 1.0)
    reward = reward_fn(state, action)
    return state, action, reward


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


def train_q(data, cql_alpha=0.0, steps=1500, batch_size=256, num_random_actions=20):
    states, actions, rewards = data
    q = QNetwork()
    optimizer = torch.optim.Adam(q.parameters(), lr=3e-4)

    for _ in range(steps):
        index = torch.randint(0, states.shape[0], (batch_size,))
        state = states[index]
        action = actions[index]
        reward = rewards[index]

        q_data = q(state, action)
        loss = ((q_data - reward) ** 2).mean()

        if cql_alpha > 0.0:
            repeated_state = state[:, None, :].expand(-1, num_random_actions, -1)
            repeated_state = repeated_state.reshape(-1, 1)
            random_action = torch.empty(batch_size, num_random_actions, 1).uniform_(-1.0, 1.0)
            random_action = random_action.reshape(-1, 1)

            q_random = q(repeated_state, random_action).reshape(batch_size, num_random_actions)
            conservative_penalty = torch.logsumexp(q_random, dim=1).mean() - q_data.mean()
            loss = loss + cql_alpha * conservative_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return q


def train_actor(data, q, bc_weight=0.0, steps=1200, batch_size=256):
    states, actions, _ = data
    actor = Actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    for _ in range(steps):
        index = torch.randint(0, states.shape[0], (batch_size,))
        state = states[index]
        data_action = actions[index]

        policy_action = actor(state)
        loss = -q(state, policy_action).mean()

        if bc_weight > 0.0:
            loss = loss + bc_weight * ((policy_action - data_action) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return actor


@torch.no_grad()
def evaluate(name, actor):
    state = torch.linspace(-1.0, 1.0, 1000).view(-1, 1)
    action = actor(state).clamp(-1.0, 1.0)
    actual_return = reward_fn(state, action).mean().item()
    support_gap = (action - behavior_mean(state)).abs().mean().item()
    print(f"{name:16s} return={actual_return: .4f} support_gap={support_gap:.4f}")


def main():
    data = make_dataset()
    states, actions, rewards = data

    behavior_return = rewards.mean().item()
    optimal_return = reward_fn(states, target_action(states)).mean().item()
    print(f"behavior_policy  return={behavior_return: .4f} support_gap=0.0000")
    print(f"oracle_policy    return={optimal_return: .4f} support_gap=unknown")

    naive_q = train_q(data, cql_alpha=0.0)
    cql_q = train_q(data, cql_alpha=0.20)

    evaluate("naive_q_actor", train_actor(data, naive_q, bc_weight=0.0))
    evaluate("cql_actor", train_actor(data, cql_q, bc_weight=0.0))
    evaluate("td3bc_actor", train_actor(data, naive_q, bc_weight=3.0))


if __name__ == "__main__":
    main()
