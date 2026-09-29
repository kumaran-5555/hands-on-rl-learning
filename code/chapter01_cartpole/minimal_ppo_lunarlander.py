
import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# Part 1: Actor-Critic network (separate heads + orthogonal init)
# ==========================================
class ActorCritic(nn.Module):
    """
    Separate Actor-Critic network (aligned with SB3 MlpPolicy):
    - Actor and Critic use their own hidden layers to avoid gradient conflicts
    - Orthogonal init: actor output layer gain=0.01 keeps the initial policy close to the mean action

    Same continuous-action setup as the Pendulum version: the actor outputs the
    MEAN of a Gaussian, and log_std is a free parameter. The only change is that
    there are now TWO action dimensions (main engine, side engine) instead of one.
    """

    def __init__(self, obs_dim=8, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # std is state-independent and learned in log space so it stays positive
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization, matching SB3 defaults"""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # actor output layer uses a small gain → initial policy close to zero-mean
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        # critic output layer gain=1
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

    def forward(self, x):
        mean = self.actor(x)
        value = self.critic(x)
        return mean, value.squeeze(-1)

    def get_dist(self, mean):
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_action(self, obs, deterministic=False):
        mean, value = self.forward(obs)
        dist = self.get_dist(mean)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        # one log_prob per action dimension → sum to get the joint log_prob
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


# ==========================================
# Part 2: Collect trajectories (Rollout)
# ==========================================
def collect_rollout_2(model, env, num_steps=4096, act_low=-1.0, act_high=1.0):
    """
    Collect trajectories and output the following for each step

    ** observation and action taken **
    - observation
    - action taken (unclipped, this is what the log_prob refers to)
    - value estimate of the state
    - log probability of the action

    ** feedback from the environment **
    - reward received
    - next observation (only if truncated but not terminated)
    - terminated (crashed or came to rest): V(s')=0
    - truncated (reached 1000 step limit): V(s') needs bootstrap

    ** bootstrap value **
    - bootstrap value [if the rollout steps ended but the last step is not truncated or terminated]

    Unlike Pendulum, LunarLander really does terminate: the lander either crashes
    or settles on the ground. So both the terminated and truncated branches get used.
    """

    obs, _ = env.reset()

    rollout_data = []
    last_step_bootstrap_value = None

    for step in range(num_steps):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs)

        # the Gaussian is unbounded, but the env only accepts throttles in [-1, 1].
        # we clip what we SEND to the env but keep the raw action for the log_prob.
        clipped_action = np.clip(action.numpy(), act_low, act_high)
        next_obs, reward, terminated, truncated, info = env.step(clipped_action)
        rollout_data.append({
            "obs": obs,
            "action": action.numpy(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": reward,
            "next_obs": next_obs if truncated and not terminated else None,
            "terminated": terminated,
            "truncated": truncated,
        })

        if terminated or truncated:
            # there is no next observation to bootstrap from if the episode ended
            next_obs, _ = env.reset()  # reset the environment for the next episode

        obs = next_obs

    # If the rollout ended but the last step is not truncated or terminated, we need to bootstrap
    if not terminated and not truncated:
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            _, _, last_step_bootstrap_value = model.get_action(obs)
        last_step_bootstrap_value = last_step_bootstrap_value.item()

    return rollout_data, last_step_bootstrap_value


# ==========================================
# Part 3: Compute GAE advantages
# ==========================================
def compute_gae_2(model, rollout_data, last_step_bootstrap_value, gamma=0.999, lam=0.98):
    """
    Compute GAE advantages.
    Return
    - advantages for each step (normalized)
    - returns for each step (reward + value of the next state)

    - Advantages are computed as:
        - (reward + expected value of state t+1  - expected value of state t)
        - expected future value is discounted by gamma and lam

    - Returns are computed as:
        - (reward + expected value of state t+1)
        - expected future value is discounted by gamma

    gamma is high (0.999) because the big +100/-100 payoff arrives at the very end
    of an episode that can run 1000 steps. A lower gamma would discount the landing
    bonus into irrelevance.
    """

    deltas = []

    # simple forward loop to compute delta for each step

    for i in range(len(rollout_data)):
        data = rollout_data[i]

        if data["terminated"]:
            # no future value since this is a terminal state
            # advantage becomes just delta
            delta = data["reward"] - data["value"]

        elif data["truncated"]:
            # the episode was cut off by the time limit, not by crashing or landing.
            # the lander is still flying, so bootstrap from V(next_obs).
            with torch.no_grad():
                next_obs = torch.FloatTensor(data["next_obs"])
                _, next_value = model(next_obs)
            delta = data["reward"] + gamma * next_value.item() - data["value"]

        else:
            if i < len(rollout_data) - 1:
                # value of state t+1 is the value of the next step in the rollout or the bootstrap value if this is the last step
                next_value = rollout_data[i + 1]["value"]
            else:
                next_value = last_step_bootstrap_value

            delta = data["reward"] + gamma * next_value - data["value"]

        deltas.append(delta)

    advantages = [None] * len(deltas)
    returns = [None] * len(deltas)

    gae = 0
    for i in reversed(range(len(rollout_data))):
        data = rollout_data[i]
        if data['terminated'] or data['truncated']:
            # no backpropagation is needed across an episode boundary
            gae = deltas[i]
        else:
            # gae = current steps error + discounted future gae
            # there are two discounting factors
            # gamma - discounts all future events
            # lambda - controls smoothing of current advantage with future. gae_t = (delta_t + lambda * delta_t+1 + lambda**2 * delta_t+2 ...)

            # we are doing it in reverse, gae on right will refer to next time step in the trajectory
            gae = deltas[i] + gamma * lam * gae

        advantages[i] = gae
        # actual return can be derived from advantage:
        # gae = reward_t + value_t+1 - value_t
        # return_t = reward_t + value_t+1
        # return_t = gae + value_t

        returns[i] = gae + data["value"]

    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    # keeps the policy gradient step size stable as the reward scale grows
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ==========================================
# Part 4: PPO update
# ==========================================
def ppo_update_2(
    model,
    optimizer,
    transitions,
    advantages,
    returns,
    clip_eps=0.2,
    epochs=4,
    batch_size=64,
    ent_coef=0.01,
):
    """PPO clipped-objective update"""

    total_policy_loss = 0
    total_value_loss = 0
    n_updates = 0

    # convert to arrays for batch level sampling
    obs = torch.FloatTensor(np.array([t["obs"] for t in transitions]))
    actions = torch.FloatTensor(np.array([t["action"] for t in transitions]))
    old_log_probs = torch.FloatTensor(np.array([t["log_prob"] for t in transitions]))

    for e in range(epochs):
        # sample a batch
        indices = np.random.permutation(len(transitions))
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_prob = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            # compute new log probability from updated model

            mean, values = model(batch_obs)  # call forward
            dist = model.get_dist(mean)
            new_log_prob = dist.log_prob(batch_actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # PPO clipped objective: ratio = new_prob / old_prob
            # adjust ratio in the direction of advantage till clipped threshold
            # if advantage > 0, maximize the ratio till clip (increase abs ppo_loss)
            # if advantage < 0, minimize the ratio till clip (decrease abs ppo_loss)
            ratio = torch.exp(new_log_prob - batch_old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

            # value regression loss against return value
            value_loss = ((values - batch_returns) ** 2).mean()

            # total loss = ppo loss + value loss - entropy bonus
            # the entropy bonus matters here: without it the policy shuts both engines
            # off early (fuel costs reward) and learns to crash gently forever
            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

            # backpropagate and update model parameters
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Enable gui")
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    env = gym.make("LunarLanderContinuous-v3")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = float(env.action_space.low[0])
    act_high = float(env.action_space.high[0])

    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    print("Model architecture:")
    print("=" * 50)
    print(model)

    print("Environment info:")
    print("=" * 50)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Observation high:", env.observation_space.high)
    print("Observation low:", env.observation_space.low)

    total_iterations = 150
    steps_per_rollout = 4096

    for i in range(total_iterations):
        rollout_data, last_step_bootstrap_value = collect_rollout_2(
            model, env, num_steps=steps_per_rollout, act_low=act_low, act_high=act_high
        )

        advantages, returns = compute_gae_2(model, rollout_data, last_step_bootstrap_value)
        metrics = ppo_update_2(model, optimizer, rollout_data, advantages, returns)

        # Compute episode rewards and lengths
        ep_rewards = []
        ep_lengths = []
        ep_reward = 0
        ep_length = 0
        for t in rollout_data:
            ep_reward += t["reward"]
            ep_length += 1
            if t["terminated"] or t["truncated"]:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward = 0
                ep_length = 0

        print(f"Iteration {i+1}/{total_iterations}: Policy Loss = {metrics['policy_loss']:.4f}, Value Loss = {metrics['value_loss']:.4f} : Episode Reward = {np.mean(ep_rewards):.2f}, Episode Length = {np.mean(ep_lengths):.2f}, Std = {model.log_std.exp().mean().item():.3f}")

    print("=" * 50)
    print("Evaluation ")

    try:
        vis_env = gym.make("LunarLanderContinuous-v3", render_mode="human")
        for ep in range(5):
            obs, _ = vis_env.reset()
            done, truncated, score = False, False, 0
            while not (done or truncated):
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    action, _, _ = model.get_action(obs_tensor, deterministic=True)
                action = np.clip(action.numpy(), act_low, act_high)
                obs, reward, done, truncated, _ = vis_env.step(action)
                score += reward
            print(f"Episode {ep + 1} score: {score:.2f}")
        vis_env.close()
    except Exception as e:
        print("GUI not available. Skipping evaluation render.")


if __name__ == "__main__":
    train()
