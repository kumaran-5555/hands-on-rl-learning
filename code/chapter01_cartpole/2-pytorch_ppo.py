"""
Chapter 1: Cracking open the black box — implementing PPO in pure PyTorch to train CartPole
Shows the core logic behind SB3's model.learn()

Metrics during training (reward curve, losses, etc.) are logged via SwanLab.
After training, optionally pop up a GUI window to showcase the learned results.

Usage:
    # Default: train + SwanLab curves (no GUI, faster)
    python 2-pytorch_ppo.py

    # Open the GUI demo (pops up the cart animation window after training)
    python 2-pytorch_ppo.py --gui

About the --gui flag:
    The training phase is always headless (no rendering); speed is unaffected by the GUI.
    --gui only controls whether the post-training demo pops up the CartPole animation window.
    With the GUI on, each demo frame waits for the screen refresh (~16ms), noticeably slower;
    with the GUI off, the demo is pure computation and finishes within seconds.
"""

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
    Collect trajectories, correctly handling terminated vs truncated:
    - terminated (pole fell): V(s')=0
    - truncated (reached step limit): V(s') needs bootstrap
    - end of rollout not done: needs bootstrap
    """
    obs, _ = env.reset()
    transitions = []

    for _ in range(num_steps):
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        # truncated but not terminated → store next_obs for bootstrap
        transitions.append(
            {
                "obs": obs,
                "action": action.item(),
                "log_prob": log_prob.item(),
                "value": value.item(),
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "next_obs": next_obs if truncated and not terminated else None,
            }
        )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # End-of-rollout bootstrap: if the last episode didn't finish, compute V(s_last)
    if not (terminated or truncated):
        with torch.no_grad():
            _, _, bootstrap_value = model.get_action(torch.FloatTensor(obs))
        last_bootstrap = bootstrap_value.item()
    else:
        last_bootstrap = 0.0

    return transitions, last_bootstrap


# ==========================================
# Part 3: Compute GAE advantages
# ==========================================
def compute_gae(model, transitions, last_bootstrap, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation, correctly handling:
    - terminated (truly done): don't propagate GAE, V(s')=0
    - truncated (time cutoff): don't propagate GAE, but use V(next_obs) as bootstrap
    - normal step: propagate GAE normally
    """
    n = len(transitions)
    rewards = [t["reward"] for t in transitions]
    values = [t["value"] for t in transitions]

    # Precompute the bootstrap value for each truncated step
    bootstrap_values = [0.0] * n
    for i, t in enumerate(transitions):
        if t["truncated"] and not t["terminated"] and t["next_obs"] is not None:
            with torch.no_grad():
                _, _, bv = model.get_action(torch.FloatTensor(t["next_obs"]))
            bootstrap_values[i] = bv.item()

    advantages = []
    gae = 0
    next_value = last_bootstrap

    for step in reversed(range(n)):
        t = transitions[step]

        if t["terminated"]:
            # Truly done: V(s') = 0
            delta = rewards[step] - values[step]
            gae = delta
        elif t["truncated"]:
            # Time cutoff: bootstrap with V(next_obs), but don't propagate GAE
            delta = rewards[step] + gamma * bootstrap_values[step] - values[step]
            gae = delta
        else:
            # Normal step
            delta = rewards[step] + gamma * next_value - values[step]
            gae = delta + gamma * lam * gae

        next_value = values[step]
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages_norm, returns, advantages



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
    obs = np.array([t["obs"] for t in transitions])
    actions = np.array([t["action"] for t in transitions])
    old_log_probs = np.array([t["log_prob"] for t in transitions])

    obs = torch.FloatTensor(obs)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl = 0
    total_clip_frac = 0
    n_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(len(transitions))

        for start in range(0, len(transitions), batch_size):
            idx = indices[start : start + batch_size]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = ((values - batch_returns) ** 2).mean()

            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.0 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Collect metrics
            with torch.no_grad():
                total_kl += (batch_old_log_probs - new_log_probs).mean().item()
                total_clip_frac += (
                    ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                )

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_kl / n_updates,
        "clip_fraction": total_clip_frac / n_updates,
    }


# ==========================================
# Part 5: Training loop
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="纯 PyTorch PPO CartPole 训练")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="训练结束后弹出 GUI 窗口演示智能体（默认关闭，仅输出得分）",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    env = gym.make("CartPole-v1")

    # Print environment info (state space, action space, boundary thresholds)
    print("=" * 50)
    print("CartPole-v1 环境信息")
    print("=" * 50)
    print(f"  观测空间:  {env.observation_space}")
    print(f"  动作空间:  {env.action_space}")
    print(f"  观测上限:  {env.observation_space.high}")
    print(f"  观测下限:  {env.observation_space.low}")
    print(
        f"  终止条件:  位置 > ±{env.unwrapped.x_threshold}, "
        f"角度 > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
        f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)"
    )
    print("=" * 50)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    total_iterations = 40
    steps_per_rollout = 2048

    # Initialize SwanLab
    swanlab.init(
        project="cartpole-pytorch",
        experiment_name="PPO-PyTorch-CartPole-v1",
        mode="local",
        config={
            "algorithm": "PPO",
            "lr": 3e-4,
            "total_iterations": total_iterations,
            "steps_per_rollout": steps_per_rollout,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "epochs": 10,
            "batch_size": 64,
        },
    )

    print("开始训练（纯 PyTorch PPO + SwanLab）...")
    print("-" * 60)

    total_timesteps = 0

    for iteration in range(total_iterations):
        # Collect data
        transitions, last_bootstrap = collect_rollout(model, env, steps_per_rollout)

        total_timesteps += len(transitions)

        # Compute episode rewards and lengths
        ep_rewards = []
        ep_lengths = []
        ep_reward = 0
        ep_length = 0
        for t in transitions:
            ep_reward += t["reward"]
            ep_length += 1
            if t["terminated"] or t["truncated"]:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward = 0
                ep_length = 0

        # Compute advantages
        advantages, returns, advantages_orig = compute_gae(model, transitions, last_bootstrap)
        advantages_2, returns_2 = compute_gae_2(model, transitions, last_bootstrap)

        # PPO update
        metrics = ppo_update(model, optimizer, transitions, advantages, returns)

        # Explained variance (re-predict with the updated Critic, matching SB3)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(np.array([t["obs"] for t in transitions]))
            _, updated_values = model(obs_tensor)
        return_values = returns.numpy()
        updated_values_np = updated_values.numpy()
        var_returns = np.var(return_values)
        if var_returns < 1e-6:
            # All returns identical (e.g. all 500), EV is meaningless, set to 0
            explained_variance = 0.0
        else:
            explained_variance = (
                1 - np.var(return_values - updated_values_np) / var_returns
            )

        mean_reward = np.mean(ep_rewards) if ep_rewards else 0
        mean_ep_len = np.mean(ep_lengths) if ep_lengths else 0

        # Linear learning-rate decay (matching SB3 default behavior)
        frac = 1.0 - iteration / total_iterations
        lr = 3e-4 * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Log to SwanLab (aligned with SB3 metrics)
        swanlab.log(
            {
                "rollout/ep_rew_mean": mean_reward,
                "rollout/ep_len_mean": mean_ep_len,
                "train/policy_gradient_loss": metrics["policy_loss"],
                "train/value_loss": metrics["value_loss"],
                "train/entropy_loss": -metrics["entropy"],
                "train/approx_kl": metrics["approx_kl"],
                "train/clip_fraction": metrics["clip_fraction"],
                "train/clip_range": 0.2,
                "train/explained_variance": explained_variance,
                "train/learning_rate": lr,
                "train/n_updates": (iteration + 1) * 10 * (steps_per_rollout // 64),
                "time/total_timesteps": total_timesteps,
                "time/iterations": iteration + 1,
            },
            step=iteration,
        )

        print(
            f"  迭代 {iteration + 1:2d}/{total_iterations} | "
            f"回合数: {len(ep_rewards):3d} | "
            f"平均奖励: {mean_reward:6.1f} | "
            f"KL: {metrics['approx_kl']:.4f} | "
            f"clip%: {metrics['clip_fraction']:.1%}"
        )

    print("-" * 60)

    # Final evaluation
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done, truncated, score = False, False, 0
        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action, _, _ = model.get_action(obs_tensor, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            score += reward
        eval_rewards.append(score)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\n训练完成！20 回合评估: {mean_reward:.1f} +/- {std_reward:.1f}")

    swanlab.log(
        {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
        }
    )

    # Save model
    torch.save(model.state_dict(), "output/pytorch_ppo_cartpole.pth")
    print(f"模型已保存到 output/pytorch_ppo_cartpole.pth")

    # GUI demo
    if args.gui:
        try:
            vis_env = gym.make("CartPole-v1", render_mode="human")
            print("\n正在演示学习成果（5 个回合）...")
            for ep in range(5):
                obs, _ = vis_env.reset()
                done, truncated, score = False, False, 0
                while not (done or truncated):
                    obs_tensor = torch.FloatTensor(obs)
                    with torch.no_grad():
                        action, _, _ = model.get_action(obs_tensor, deterministic=True)
                    obs, reward, done, truncated, _ = vis_env.step(action.item())
                    score += reward
                print(f"  演示回合 {ep + 1} 得分: {score}")
            vis_env.close()
            print("\nGUI 演示结束。")
        except Exception:
            print("(跳过 GUI 演示，无图形界面)")
    else:
        print("\n提示: 加 --gui 可弹出小车动画窗口查看演示效果。")

    env.close()
    swanlab.finish()

    print("SwanLab 实验看板: swanlab watch swanlog")


if __name__ == "__main__":
    train()
