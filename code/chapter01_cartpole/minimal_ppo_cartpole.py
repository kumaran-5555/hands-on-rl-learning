
import argparse
import os

import gymnasium as gym
import numpy as np
import swanlab
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
    - Orthogonal init: actor output layer gain=0.01 keeps the initial policy close to uniform
    """

    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
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
        # actor output layer uses a small gain → initial policy close to uniform
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        # critic output layer gain=1
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ==========================================
# Part 2: Collect trajectories (Rollout)
# ==========================================
def collect_rollout(model, env, num_steps=2048):
    """
    Collect trajectories and output the following for each step
    
    ** observation and action taken **
    - observation
    - action taken 
    - value estimate of the state    
    - log probability of the action

    ** feedback from the environment **
    - reward received
    - next observation (only if truncated but not terminated)
    - terminated (pole fell): V(s')=0
    - truncated (reached step limit): V(s') needs bootstrap

    ** bootstrap value **
    - bootstarp value [if the rollout steps ended but the last step is not truncted or terminated]

    """

    print(f"Collecting rollout for {num_steps} steps...")

    obs, _ = env.reset()

    rollout_data = []
    last_step_bootstrap_value = None

    for step in range(num_steps):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs) 

        next_obs, reward, terminated, truncated, info = env.step(action.item())
        rollout_data.append({
            "obs": obs,
            "action": action.item(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": reward,
            "next_obs": next_obs if not (terminated or truncated) else None,
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
            _,_, last_step_bootstrap_value = model.get_action(obs)
    
    print(f"Rollout collection completed. Collected {len(rollout_data)} steps.")
    return rollout_data, last_step_bootstrap_value
            













    


# ==========================================
# Part 3: Compute GAE advantages
# ==========================================
def compute_gae(model, rollout_data, last_step_bootstrap_value, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages. 
    Return 
    - advantages for each step
    - returns for each step (reward + value of the next state)    

    - Avantages are computed as: 
        - (reward + expected value of state t+1  - expected value of state t)
        - expected future value is discounted by gamma and lam

    - Returns are computed as:
        - (reward + expected value of state t+1)
        - expected future value is discounted by gamma
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
            # no future value since this is a terminal state
            # advantage becomes just delta         
            delta = data["reward"] - data["value"]
            
        else:
            if i < len(rollout_data) - 1:
                # value of state t+1 is the value of the next step in the rollout or the bootstrap value if this is the last step
                next_value = rollout_data[i+1]["value"]
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
            # no backpropagation is needed at the terminal states
            advantages[i] = deltas[i]
            gae = deltas[i]
        else:
            # gae = current steps error + discounted future gae
            # there are two discounting factors 
            # gamma - discounts all future 
            # lambda - controls smoothing of current advantage with future. gae_t = (delta_t + lambda * delta_t+1 + lambda**2 * delta_t+2)
            gae = deltas[i] + gamma * lam * gae # we are doing it in reverse, gae on right will refer to next step in trajectory

        advantages[i] = gae
        # actual return can be derived from advantage: 
        # gae = reward_t + value_t+1 - value_t
        # return_t = reward_t + value_t+1
        # return_t = gae + value_t

        returns[i] = gae + data["value"] 

    return advantages, returns





# ==========================================
# Part 4: PPO update
# ==========================================
def ppo_update(
    model,
    optimizer,
    transitions,
    advantages,
    returns,
    clip_eps=0.2,
    epochs=10,
    batch_size=64,
):
    """PPO clipped-objective update"""
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Enable gui")
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    env = gym.make("CartPole-v1")
    

    model = ActorCritic()
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

    for i in range(2):
        rollout_data, last_step_bootstrap_value = collect_rollout(model, env, num_steps=2048)


        
    





if __name__ == "__main__":
    train()
