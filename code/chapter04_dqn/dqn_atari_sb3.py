"""
Train a real Atari DQN agent with Stable-Baselines3.

This script is intentionally more practical than the short teaching snippets in
the chapter text. It uses the standard Atari wrapper stack, frame stacking,
delayed learning, periodic evaluation, TensorBoard logging, and checkpoints.

Examples:
    python code/chapter04_dqn/dqn_atari_sb3.py --total-timesteps 200000 --learning-starts 10000
    python code/chapter04_dqn/dqn_atari_sb3.py --total-timesteps 5000000 --learning-starts 100000
    tensorboard --logdir output/dqn_atari
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
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


gym.register_envs(ale_py)


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
        experiment_name=args.swanlab_run_name or args.run_name,
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
    ax.plot(timesteps, mean_rewards, color="#DC2626", linewidth=2.4, marker="o", markersize=4)
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color="#FCA5A5", alpha=0.28)
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
    parser = argparse.ArgumentParser(description="Train DQN on an Atari game.")
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("output/dqn_atari"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--optimize-memory-usage", action="store_true")

    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--no-terminal-on-life-loss", action="store_true")
    parser.add_argument("--no-clip-reward", action="store_true")

    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="hands-on-modern-rl-dqn")
    parser.add_argument("--swanlab-run-name", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default="local")
    return parser.parse_args()


def build_env(args: argparse.Namespace, monitor_dir: Path):
    """Create the Atari environment with SB3's production wrapper stack."""
    wrapper_kwargs = {
        "noop_max": args.noop_max,
        "frame_skip": args.frame_skip,
        "screen_size": args.screen_size,
        "terminal_on_life_loss": not args.no_terminal_on_life_loss,
        "clip_reward": not args.no_clip_reward,
        "action_repeat_probability": args.repeat_action_probability,
    }

    # ALE v5 environments already have frame skipping and sticky actions by
    # default. We disable those here and let AtariWrapper own both choices, so
    # the actual training setup is explicit and does not double-skip frames.
    env_kwargs = {}
    if args.env_id.startswith("ALE/"):
        env_kwargs = {
            "frameskip": 1,
            "repeat_action_probability": 0.0,
        }

    env = make_atari_env(
        args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(monitor_dir),
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs=env_kwargs,
    )
    env = VecFrameStack(env, n_stack=args.frame_stack)
    return VecTransposeImage(env)


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        env_name = args.env_id.replace("/", "_")
        args.run_name = f"{env_name}_dqn_seed{args.seed}"

    run_dir = args.output_dir / args.run_name
    train_monitor_dir = run_dir / "monitor_train"
    eval_monitor_dir = run_dir / "monitor_eval"
    checkpoint_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    tensorboard_dir = run_dir / "tensorboard"

    for path in [train_monitor_dir, eval_monitor_dir, checkpoint_dir, best_model_dir, tensorboard_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if args.total_timesteps <= args.learning_starts:
        print(
            "Warning: total_timesteps is not larger than learning_starts; "
            "the replay buffer will be filled, but DQN will barely update."
        )

    set_random_seed(args.seed)
    try:
        train_env = build_env(args, train_monitor_dir)
        eval_env = build_env(args, eval_monitor_dir)
    except gym.error.Error as exc:
        raise SystemExit(
            "Failed to create the Atari environment. Install the Atari extras "
            "and ROM license first:\n\n"
            "  pip install -r code/chapter04_dqn/requirements.txt\n\n"
            "If you installed dependencies from inside code/, use:\n\n"
            "  pip install -r chapter04_dqn/requirements.txt\n\n"
            f"Original Gymnasium error: {exc}"
        ) from exc

    model = DQN(
        "CnnPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=10,
        optimize_memory_usage=args.optimize_memory_usage,
        replay_buffer_kwargs=(
            {"handle_timeout_termination": False}
            if args.optimize_memory_usage
            else None
        ),
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    callbacks = []
    eval_csv_path = run_dir / "eval" / "eval_metrics.csv"
    if args.eval_freq > 0:
        callbacks.append(
            EvaluationCsvCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(run_dir / "eval"),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
                csv_path=eval_csv_path,
            )
        )
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix="dqn_atari",
            )
        )
    swanlab_callback = maybe_make_swanlab_callback(args)
    if swanlab_callback is not None:
        callbacks.append(swanlab_callback)
        callbacks.append(RestoreStdoutLog())
        callbacks.append(SwanLabEvalCallback(eval_csv_path))

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks) if callbacks else None,
            tb_log_name=args.run_name,
            progress_bar=args.progress_bar,
        )
        final_model_path = run_dir / "final_model"
        model.save(str(final_model_path))
        plot_path = run_dir / "eval" / "eval_curve.png"
        save_eval_plot(eval_csv_path, plot_path, f"{args.env_id} DQN evaluation")
        write_summary(
            run_dir / "summary.json",
            {
                "env_id": args.env_id,
                "total_timesteps": args.total_timesteps,
                "seed": args.seed,
                "eval_csv": str(eval_csv_path),
                "eval_curve": str(plot_path),
                "swanlab_project": None if args.no_swanlab else args.swanlab_project,
                "swanlab_mode": None if args.no_swanlab else args.swanlab_mode,
            },
        )
        print(f"Saved final model to {final_model_path}.zip")
        print(f"TensorBoard logs: {tensorboard_dir}")
        print(f"Evaluation CSV: {eval_csv_path}")
        print(f"Evaluation curve: {plot_path}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
