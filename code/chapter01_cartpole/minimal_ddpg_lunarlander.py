
"""
Minimal DDPG / TD3 / TD3+BC for LunarLanderContinuous-v3.

Run DDPG:    python minimal_ddpg_lunarlander.py
Run TD3:     python minimal_ddpg_lunarlander.py --td3
Run TD3+BC:  python minimal_ddpg_lunarlander.py --td3 --bc

Every TD3-specific piece of code is marked with a "TD3 TWEAK" comment block so you
can diff the two algorithms by reading. There are exactly three tweaks:

  TD3 TWEAK 1 - Twin critics / clipped double-Q  (fixes Q-value overestimation)
  TD3 TWEAK 2 - Delayed policy updates           (let the critic settle first)
  TD3 TWEAK 3 - Target policy smoothing          (stop the actor exploiting Q spikes)

Everything else is plain DDPG.

The --bc flag adds the behavior-cloning term from TD3+BC (Fujimoto & Gu, NeurIPS
2021), marked with a "TD3+BC" comment block. It is a single extra term in the actor
loss that keeps the policy near the actions actually present in the replay buffer.

IMPORTANT CAVEAT: TD3+BC is an OFFLINE algorithm. Its purpose is to stop the actor
from exploiting critic errors on actions that no logged policy ever took -- errors
that, offline, can never be corrected because you cannot go try the action. Running
--bc here (online, with a buffer that keeps growing) still demonstrates the
mechanism, but the constraint anchors you to a MIXTURE of every past policy,
including the 10k uniform-random warmup steps. Expect it to slow online learning
rather than help it. The setting where it pays off is a frozen buffer.

Also note: the TD3+BC paper pairs the BC term with observation normalization
(running mean/std over the dataset). That is not implemented here.
"""

import argparse
import copy
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# Part 1: Actor and Critic networks
# ==========================================
#
# The biggest conceptual break from PPO. In PPO the actor produced the MEAN of a
# Gaussian and we sampled from it. Here the actor is DETERMINISTIC: it outputs the
# single action it thinks is best. There is no distribution, so there is also no
# log_prob, no ratio, and no clipped objective. Exploration has to be bolted on
# manually by adding noise at action-selection time (see Part 3).
#
# The critic also changes shape. PPO's critic learned V(s) -- "how good is this
# state". DDPG's critic learns Q(s, a) -- "how good is this action in this state".
# That difference is the whole trick: if the critic can score an arbitrary action,
# then the actor can be trained by simply asking the critic "which action scores
# highest here?" and following that gradient.


class Actor(nn.Module):
    """Deterministic policy: obs -> action."""

    def __init__(self, obs_dim=8, act_dim=2, hidden=256, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            # tanh squashes to [-1, 1], which is exactly LunarLander's action range.
            # This is why we never need to clip the actor's output the way we had to
            # clip the unbounded Gaussian samples in the PPO version.
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, obs):
        return self.net(obs) * self.max_action


class Critic(nn.Module):
    """
    Action-value function Q(s, a).

    Holds one Q network for DDPG, two for TD3. Keeping both inside a single module
    means one optimizer and one target-network copy regardless of which algorithm
    is running.
    """

    def __init__(self, obs_dim=8, act_dim=2, hidden=256, twin=False):
        super().__init__()
        self.twin = twin

        def build_q():
            # note the input width: obs AND action are concatenated and fed in together
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = build_q()

        # ---------------- TD3 TWEAK 1 (part a): twin critics ----------------
        # DDPG's single critic systematically OVERESTIMATES Q. The actor is trained to
        # maximize Q, so it actively hunts for whatever states/actions the critic is
        # wrong about, and that error then feeds back into the target. TD3 trains a
        # second, independently initialized critic and later takes the pessimistic
        # min of the two, which cancels most of that bias.
        self.q2 = build_q() if twin else None
        # --------------------------------------------------------------------

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1) if self.twin else None
        return q1, q2

    def q1_only(self, obs, act):
        """Used for the actor loss. Even in TD3 the actor follows q1 alone."""
        return self.q1(torch.cat([obs, act], dim=-1)).squeeze(-1)


# ==========================================
# Part 2: Replay buffer
# ==========================================
#
# This replaces collect_rollout_2 from the PPO file, and it is the other big break.
#
# PPO is ON-POLICY: the 4096 steps it collects were produced by the current policy,
# they are used for a handful of gradient epochs, and then they are thrown away.
# DDPG is OFF-POLICY: every transition ever seen stays in a buffer and can be
# resampled thousands of times. That is why DDPG needs far fewer environment steps
# than PPO -- it squeezes much more learning out of each one.
#
# It works because Q-learning's update target only needs (s, a, r, s', done). It
# never asks "what was the probability of taking a under the policy that generated
# this data?", so stale data is still valid. PPO's importance ratio does ask exactly
# that, which is why PPO cannot reuse old data.


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=200_000):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.ptr = 0  # where the next write goes
        self.size = 0  # how much is actually filled

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        # ring buffer: wrap around and start overwriting the oldest transitions
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=256):
        # uniform random sample. Sampling randomly (rather than in collected order)
        # breaks the temporal correlation between consecutive states, which would
        # otherwise make the gradient estimates wildly biased.
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor(self.dones[idx]),
        )


# ==========================================
# Part 3: Exploration and target networks
# ==========================================


def select_action(actor, obs, noise_std=0.1, max_action=1.0, deterministic=False):
    """
    Pick an action, optionally with exploration noise.

    PPO explored for free: the policy WAS a distribution, so sampling from it
    explored automatically, and the entropy bonus controlled how much. A deterministic
    actor always returns the same action for the same state and would never explore
    at all, so we add Gaussian noise by hand.

    Consequence worth noting: unlike PPO's learned log_std, this noise level is a
    fixed hyperparameter. It does not shrink as the policy improves unless you
    manually anneal it. That is one reason DDPG feels fussier to tune.
    """
    with torch.no_grad():
        action = actor(torch.FloatTensor(obs)).numpy()

    if not deterministic:
        action = action + np.random.normal(0, noise_std * max_action, size=action.shape)

    # the noise can push us outside the valid range, so clip before stepping the env
    return np.clip(action, -max_action, max_action)


def soft_update(net, target_net, tau=0.005):
    """
    Nudge the target network a tiny bit toward the live network (Polyak averaging):
        target = tau * live + (1 - tau) * target

    This is DDPG's answer to the stability problem that PPO solved with ratio
    clipping. The critic's regression target is built from the critic's own
    predictions, so if the target moved every gradient step the network would be
    chasing itself and diverge. With tau=0.005 the target lags ~200 steps behind,
    which makes the target effectively a fixed constant on short timescales.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ==========================================
# Part 4: DDPG / TD3 update
# ==========================================
#
# Replaces both compute_gae_2 and ppo_update_2. Note that GAE is gone entirely:
# there are no advantages, no returns, no lambda. DDPG bootstraps a single step
# (r + gamma * Q(s', a')) instead of blending many-step returns, so there is nothing
# to smooth over.


def ddpg_update(
    actor,
    critic,
    actor_target,
    critic_target,
    actor_optimizer,
    critic_optimizer,
    buffer,
    update_step,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    max_action=1.0,
    use_td3=False,
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5,
    use_bc=False,
    bc_alpha=2.5,
):
    """One gradient step for the critic, and (sometimes) one for the actor."""

    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

    # ---------- Critic update: regress Q(s,a) onto r + gamma * Q_target(s', a') ----------
    with torch.no_grad():
        # the target actor (not the live one) proposes the next action
        next_actions = actor_target(next_obs)

        # ---------------- TD3 TWEAK 3: target policy smoothing ----------------
        # The critic inevitably has narrow spikes of overestimated Q. A deterministic
        # actor will happily drive straight into one. Adding a little noise to the
        # TARGET action forces Q to be high across a small neighbourhood rather than
        # at a single lucky point, which smooths those spikes away.
        # (Note this noise is on the target used for training, NOT on the action we
        # actually send to the environment -- that is the separate noise in Part 3.)
        if use_td3:
            noise = (torch.randn_like(next_actions) * target_noise).clamp(
                -noise_clip, noise_clip
            )
            next_actions = (next_actions + noise).clamp(-max_action, max_action)
        # ----------------------------------------------------------------------

        target_q1, target_q2 = critic_target(next_obs, next_actions)

        # ---------------- TD3 TWEAK 1 (part b): clipped double-Q ----------------
        # Take the pessimistic estimate. Two independently initialized critics rarely
        # overestimate the same action, so the min is much closer to the truth. This
        # is the single most important of the three tweaks.
        if use_td3:
            target_q = torch.min(target_q1, target_q2)
        else:
            target_q = target_q1
        # ------------------------------------------------------------------------

        # (1 - dones) zeroes the future term at real terminal states.
        # dones holds `terminated` only, never `truncated` -- same distinction the PPO
        # file made. A lander that ran out of time is still flying, so its future value
        # is real and must be bootstrapped.
        target = rewards + gamma * (1 - dones) * target_q

    current_q1, current_q2 = critic(obs, actions)
    critic_loss = ((current_q1 - target) ** 2).mean()
    if use_td3:
        # both critics regress onto the same shared target
        critic_loss = critic_loss + ((current_q2 - target) ** 2).mean()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # ---------- Actor update: climb the critic (optionally anchored to the data) ----------
    actor_loss_value = None
    bc_loss_value = None

    # ---------------- TD3 TWEAK 2: delayed policy updates ----------------
    # DDPG updates the actor every step (policy_delay is forced to 1 below). TD3
    # updates it every 2nd step, so the critic gets two gradient steps of head start.
    # Chasing a critic that is itself still badly wrong is a good way to diverge, and
    # this is a cheap way to avoid it. The target networks move on the same schedule.
    if update_step % policy_delay == 0:
        # The actor's whole objective: produce actions the critic scores highly.
        # Gradients flow through the critic's weights into the action, and from there
        # into the actor -- which is why the critic must take `a` as an input rather
        # than just scoring states.
        pi = actor(obs)
        q = critic.q1_only(obs, pi)

        if use_bc:
            # ---------------- TD3+BC ----------------
            # Two terms now:
            #   -q.mean()          "produce actions the critic likes"   (RL)
            #   (pi - actions)^2   "stay near what the data actually did" (imitation)
            #
            # `actions` are the LOGGED actions from the buffer -- what the behavior
            # policy really took at these states. The BC term is ordinary supervised
            # regression toward them, and it is what stops the actor from wandering
            # into regions where Q is fiction because no data was ever collected there.
            #
            # Why lmbda: the two terms live on wildly different scales. The BC term is
            # an MSE over actions in [-1, 1], so it is always O(1). Q is whatever the
            # reward scale and horizon produce -- could be 10, could be 5000. Dividing
            # by mean|Q| renormalizes the RL term to O(1) as well, which is what makes
            # a single bc_alpha work across environments instead of needing a retune
            # every time the reward scale changes.
            #
            # .detach() matters: lmbda is a scaling constant, not part of the graph.
            #
            # bc_alpha is the dial between the two extremes:
            #   bc_alpha -> 0    pure behavior cloning (safe, can never beat the data)
            #   bc_alpha -> inf  pure TD3 (free to exploit critic errors)
            #   bc_alpha = 2.5   the paper's value, used across all D4RL tasks
            lmbda = bc_alpha / q.abs().mean().detach()
            bc_loss = ((pi - actions) ** 2).mean()
            actor_loss = -lmbda * q.mean() + bc_loss
            bc_loss_value = bc_loss.item()
            # ----------------------------------------
        else:
            # Negative because optimizers minimize.
            actor_loss = -q.mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        soft_update(actor, actor_target, tau)
        soft_update(critic, critic_target, tau)

        actor_loss_value = actor_loss.item()
    # ---------------------------------------------------------------------

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss_value,
        "bc_loss": bc_loss_value,
    }


def evaluate(actor, env_id, max_action, episodes=5):
    """Run the policy with noise switched off. This is the number that matters."""
    eval_env = gym.make(env_id)
    scores = []
    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done, truncated, score = False, False, 0.0
        while not (done or truncated):
            action = select_action(actor, obs, max_action=max_action, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            score += reward
        scores.append(score)
    eval_env.close()
    return float(np.mean(scores))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--td3", action="store_true", help="Enable the three TD3 tweaks")
    parser.add_argument("--bc", action="store_true", help="Add the TD3+BC behavior-cloning term")
    parser.add_argument("--bc-alpha", type=float, default=2.5, help="Higher = more RL, less BC")
    parser.add_argument("--gui", action="store_true", help="Render episodes at the end")
    parser.add_argument("--steps", type=int, default=200_000, help="Total env steps")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_id = "LunarLanderContinuous-v3"
    env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(obs_dim, act_dim, max_action=max_action)
    # TD3 builds two critics, DDPG builds one
    critic = Critic(obs_dim, act_dim, twin=args.td3)

    # Target networks start as exact frozen copies and then drift slowly behind
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    for p in actor_target.parameters():
        p.requires_grad = False
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = ReplayBuffer(obs_dim, act_dim, capacity=200_000)

    # policy_delay=1 makes the actor update every step, i.e. plain DDPG
    policy_delay = 2 if args.td3 else 1

    start_steps = 10_000  # pure random actions first, to fill the buffer with variety
    batch_size = 256

    algo_name = ("TD3" if args.td3 else "DDPG") + ("+BC" if args.bc else "")
    print(f"Algorithm: {algo_name}")
    if args.bc:
        print(f"BC alpha: {args.bc_alpha} (higher = more RL, less imitation)")
    print("=" * 50)
    print("Actor:")
    print(actor)
    print("Critic:")
    print(critic)

    print("Environment info:")
    print("=" * 50)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=args.seed)
    ep_reward, ep_length = 0.0, 0
    recent_rewards = []
    metrics = {"critic_loss": float("nan"), "actor_loss": None, "bc_loss": None}

    for step in range(1, args.steps + 1):
        # Warm-up phase: ignore the policy and act uniformly at random. Without this
        # the buffer's first transitions all come from a barely-initialized actor that
        # outputs nearly the same action everywhere, and the critic overfits to it.
        if step <= start_steps:
            action = env.action_space.sample()
        else:
            action = select_action(actor, obs, noise_std=0.1, max_action=max_action)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        # store `terminated` only -- see the bootstrap comment in ddpg_update
        buffer.add(obs, action, reward, next_obs, float(terminated))

        obs = next_obs
        ep_reward += reward
        ep_length += 1

        if terminated or truncated:
            recent_rewards.append(ep_reward)
            obs, _ = env.reset()
            ep_reward, ep_length = 0.0, 0

        # One gradient update per environment step, once there is enough data.
        # Compare with PPO, which did ~256 updates then collected 4096 fresh steps.
        if step > start_steps:
            metrics = ddpg_update(
                actor,
                critic,
                actor_target,
                critic_target,
                actor_optimizer,
                critic_optimizer,
                buffer,
                update_step=step,
                batch_size=batch_size,
                max_action=max_action,
                use_td3=args.td3,
                policy_delay=policy_delay,
                use_bc=args.bc,
                bc_alpha=args.bc_alpha,
            )

        if step % 10_000 == 0:
            train_reward = np.mean(recent_rewards[-20:]) if recent_rewards else float("nan")
            eval_reward = evaluate(actor, env_id, max_action, episodes=5)
            bc_str = ""
            if args.bc and metrics["bc_loss"] is not None:
                # how far the policy has drifted from the logged actions
                bc_str = f"BC Loss = {metrics['bc_loss']:.4f}, "
            print(
                f"Step {step}/{args.steps}: "
                f"Critic Loss = {metrics['critic_loss']:.3f}, "
                f"{bc_str}"
                f"Train Reward = {train_reward:.2f}, "
                f"Eval Reward = {eval_reward:.2f}, "
                f"Buffer = {buffer.size}"
            )

    print("=" * 50)
    print("Evaluation ")
    print(f"Final deterministic score: {evaluate(actor, env_id, max_action, episodes=10):.2f}")

    if args.gui:
        input("Enter to start GUI")
        try:
            vis_env = gym.make(env_id, render_mode="human")
            for ep in range(5):
                obs, _ = vis_env.reset()
                done, truncated, score = False, False, 0.0
                while not (done or truncated):
                    action = select_action(actor, obs, max_action=max_action, deterministic=True)
                    obs, reward, done, truncated, _ = vis_env.step(action)
                    score += reward
                print(f"Episode {ep + 1} score: {score:.2f}")
            vis_env.close()
        except Exception:
            print("GUI not available. Skipping evaluation render.")


if __name__ == "__main__":
    train()
