"""
Train a DQN agent on common discrete-action Gymnasium environments.

This script is the practical companion for Chapter 4 DQN experiments. It covers the tasks
where DQN is usually a good first choice: Classic Control and discrete Box2D
environments such as CartPole, MountainCar, Acrobot, and LunarLander.

Examples:
    python code/chapter04_dqn/dqn_gym_sb3.py --env-id CartPole-v1 --total-timesteps 100000
    python code/chapter04_dqn/dqn_gym_sb3.py --env-id MountainCar-v0 --total-timesteps 300000
    python code/chapter04_dqn/dqn_gym_sb3.py --env-id LunarLander-v3 --total-timesteps 500000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.monitor import Monitor


def patch_swanlab_cpu_count() -> None:
    """Work around psutil.cpu_count() returning None in restricted sandboxes."""
    try:
        import psutil
    except Exception:
        return

    original_cpu_count = psutil.cpu_count

    def safe_cpu_count(logical: bool = True):
        count = original_cpu_count(logical=logical)
        return 1 if count is None else count

    psutil.cpu_count = safe_cpu_count


def maybe_make_swanlab_callback(args: argparse.Namespace):
    if args.no_swanlab:
        return None

    patch_swanlab_cpu_count()
    try:
        from swanlab.integration.sb3 import SwanLabCallback
    except Exception as exc:
        print(f"SwanLab unavailable, continuing without it: {exc}")
        return None

    return SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=args.swanlab_run_name or f"DQN-{args.env_id}-seed{args.seed}",
        mode=args.swanlab_mode,
    )


class EvaluationCsvCallback(EvalCallback):
    def __init__(self, *args: Any, csv_path: Path, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mean_reward", "std_reward", "mean_ep_length", "std_ep_length"])

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and self.evaluations_results:
            rewards = np.array(self.evaluations_results[-1], dtype=float)
            lengths = np.array(self.evaluations_length[-1], dtype=float)
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    float(rewards.mean()),
                    float(rewards.std()),
                    float(lengths.mean()),
                    float(lengths.std()),
                ])
        return continue_training


class SwanLabEvalCallback(BaseCallback):
    def __init__(self, eval_csv_path: Path):
        super().__init__()
        self.eval_csv_path = eval_csv_path
        self.last_logged_step = -1
        self.swanlab = None

    def _on_training_start(self) -> None:
        try:
            import swanlab
            self.swanlab = swanlab
        except Exception:
            self.swanlab = None

    def _on_step(self) -> bool:
        if self.swanlab is None or not self.eval_csv_path.exists():
            return True
        try:
            rows = list(csv.DictReader(self.eval_csv_path.open("r", encoding="utf-8")))
        except OSError:
            return True
        if not rows:
            return True
        row = rows[-1]
        step = int(float(row["timesteps"]))
        if step <= self.last_logged_step:
            return True
        self.last_logged_step = step
        self.swanlab.log(
            {
                "eval/mean_reward": float(row["mean_reward"]),
                "eval/std_reward": float(row["std_reward"]),
                "eval/mean_ep_length": float(row["mean_ep_length"]),
            },
            step=step,
        )
        return True


class RestoreStdoutLog(BaseCallback):
    """Restore SB3's rolling stdout log table that SwanLabCallback removes.

    SwanLabCallback._init_callback() calls self.model.set_logger(...) and
    replaces SB3's default logger with a SwanLab-only one, dropping the
    HumanOutputFormat that prints the verbose=1 progress table to stdout.
    This callback re-adds an stdout output format after that replacement,
    so the terminal table reappears without affecting SwanLab logging.
    Add it after SwanLabCallback in the callback list.
    """

    def _init_callback(self) -> None:
        self.model.logger.output_formats.append(HumanOutputFormat(sys.stdout))

    def _on_step(self) -> bool:
        return True


def save_eval_plot(csv_path: Path, output_path: Path, title: str) -> None:
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = np.array([float(r["timesteps"]) for r in rows])
    mean_rewards = np.array([float(r["mean_reward"]) for r in rows])
    std_rewards = np.array([float(r["std_reward"]) for r in rows])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(timesteps, mean_rewards, color="#2563EB", linewidth=2.4, marker="o", markersize=4)
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color="#93C5FD", alpha=0.28)
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation mean reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SB3 DQN on a discrete Gymnasium task.")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("output/dqn_gym"))
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="hands-on-modern-rl-dqn")
    parser.add_argument("--swanlab-run-name", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default="local")
    return parser.parse_args()


def ensure_discrete_action_space(env_id: str) -> None:
    env = gym.make(env_id)
    try:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise SystemExit(
                f"{env_id} has action space {env.action_space}. "
                "DQN in Stable-Baselines3 requires gymnasium.spaces.Discrete."
            )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    ensure_discrete_action_space(args.env_id)

    run_dir = args.log_dir / args.env_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(args.env_id, n_envs=1, seed=args.seed, monitor_dir=str(run_dir / "monitor"))
    eval_env = Monitor(gym.make(args.env_id))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix=f"dqn_{args.env_id}",
        save_replay_buffer=True,
    )
    eval_csv_path = run_dir / "eval" / "eval_metrics.csv"
    eval_callback = EvaluationCsvCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        csv_path=eval_csv_path,
    )
    callbacks = [checkpoint_callback, eval_callback]
    swanlab_callback = maybe_make_swanlab_callback(args)
    if swanlab_callback is not None:
        callbacks.append(swanlab_callback)
        callbacks.append(RestoreStdoutLog())
        callbacks.append(SwanLabEvalCallback(eval_csv_path))

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=args.progress_bar,
    )

    model.save(run_dir / "final_model")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    print(f"Evaluation over {args.eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    plot_path = run_dir / "eval" / "eval_curve.png"
    save_eval_plot(eval_csv_path, plot_path, f"{args.env_id} DQN evaluation")
    write_summary(
        run_dir / "summary.json",
        {
            "env_id": args.env_id,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "eval_csv": str(eval_csv_path),
            "eval_curve": str(plot_path),
            "swanlab_project": None if args.no_swanlab else args.swanlab_project,
            "swanlab_mode": None if args.no_swanlab else args.swanlab_mode,
        },
    )
    print(f"Evaluation CSV: {eval_csv_path}")
    print(f"Evaluation curve: {plot_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
