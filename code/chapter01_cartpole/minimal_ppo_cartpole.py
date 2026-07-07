
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
def collect_rollout_2(model, env, num_steps=2048):
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
    
    return rollout_data, last_step_bootstrap_value
            













    


# ==========================================
# Part 3: Compute GAE advantages
# ==========================================
def compute_gae_2(model, rollout_data, last_step_bootstrap_value, gamma=0.99, lam=0.95):
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
    advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.FloatTensor(returns)


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
    epochs=10,
    batch_size=64,
):
    """PPO clipped-objective update"""

    total_policy_loss = 0
    total_value_loss = 0
    n_updates = 0

    # convert to arrays for batch level sampling
    obs = torch.FloatTensor(np.array([t["obs"] for t in transitions]))
    actions = torch.LongTensor(np.array([t["action"] for t in transitions]))
    old_log_probs = torch.FloatTensor(np.array([t["log_prob"] for t in transitions]))

    for e in range(epochs):
        # sample a batch 
        indices = np.random.permutation(len(transitions))
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start+batch_size]
            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_prob = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]


            # compute new log probability from updated model 

            logits, values = model(batch_obs) # call forward
            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(batch_actions)

            # PPO clipped objective: ratio = new_prob / old_prob
            # adjust ratio in the direction of advantage till clipped threshold
            # if advantage > 0, maximize the ratio till clip (increase abs ppo_loss)
            # if advantage < 0, mimimize the ratio till clip (decrease abs ppo_loss)
            ratio = torch.exp(new_log_prob - batch_old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
            
            # value regression loss against return value
            value_loss = ((values - batch_returns) ** 2).mean()

            # total loss = ppo loss + value loss
            loss = policy_loss + 0.5 * value_loss

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

    total_iterations = 40
    steps_per_rollout = 2048


    for i in range(total_iterations):
        rollout_data, last_step_bootstrap_value = collect_rollout_2(model, env, num_steps=steps_per_rollout)





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


        print(f"Iteration {i+1}/{total_iterations}: Policy Loss = {metrics['policy_loss']:.4f}, Value Loss = {metrics['value_loss']:.4f} : Episode Reward = {np.mean(ep_rewards):.2f}, Episode Length = {np.mean(ep_lengths):.2f}")


    print("=" * 50)
    print("Evalution ")

    try:
        vis_env = gym.make("CartPole-v1", render_mode="human")
        for ep in range(5):
            obs, _ = vis_env.reset()
            done, truncated, score = False, False, 0
            while not (done or truncated):
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    action, _, _ = model.get_action(obs_tensor, deterministic=True)
                obs, reward, done, truncated, _ = vis_env.step(action.item())
                score += reward
            print(f"Episode {ep + 1} score: {score}")
        vis_env.close()        
    except Exception as e:
        print("GUI not available. Skipping evaluation render.")

        





        
    





if __name__ == "__main__":
    train()
