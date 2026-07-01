"""
Chapter 1: Training CartPole with PPO from Stable-Baselines3

Metrics during training (reward curve, losses, etc.) are logged via SwanLab.
After training, optionally pop up a GUI window to showcase the learned results.

Usage:
    # Default: train + SwanLab curves (no GUI, faster)
    python 1-ppo_cartpole.py

    # Open the GUI demo (pops up the cart animation window after training)
    python 1-ppo_cartpole.py --gui

About the --gui flag:
    The training phase is always headless (no rendering); speed is unaffected by the GUI.
    --gui only controls whether the post-training demo pops up the CartPole animation window.
    With the GUI on, each demo frame waits for the screen refresh (~16ms), noticeably slower;
    with the GUI off, the demo is pure computation and finishes within seconds.
"""

import argparse
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat
from swanlab.integration.sb3 import SwanLabCallback
import swanlab


class LogApproxKL(BaseCallback):
    """Backfill train/approx_kl to SwanLab.

    SB3's PPO.train() records this metric internally via logger.record("train/approx_kl", ...),
    but the value is of type numpy.float32. SwanLab's SB3 callback uses
    isinstance(value, (int, float)) as a type check in write(), and numpy.float32 fails it
    (numpy.float64 and Python float pass), so approx_kl gets silently skipped.

    After each train() call, this callback pulls the approx_kl value from the logger cache,
    converts it to a Python float, and backfills it directly via swanlab.log.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # train() has already finished before _on_rollout_end fires;
        # the logger cache holds all metrics from this round of train.
        logger = self.model.logger
        if hasattr(logger, "name_to_value") and "train/approx_kl" in logger.name_to_value:
            value = float(logger.name_to_value["train/approx_kl"])
            swanlab.log({"train/approx_kl": value}, step=self.num_timesteps)


class RestoreStdoutLog(BaseCallback):
    """Add back the scrolling log table that SB3 prints to the terminal.

    SwanLabCallback._init_callback() internally calls self.model.set_logger(...),
    replacing SB3's default logger entirely with a "SwanLab-only" logger, which also
    removes the HumanOutputFormat responsible for printing the ep_rew_mean / fps /
    approx_kl table to stdout (i.e. the scrolling log at verbose=1).

    This callback runs during the _init_callback phase (after SwanLabCallback has
    replaced the logger), adding a stdout output back to the current logger, so the
    terminal scrolls and prints again while SwanLab logging is unaffected. It must be
    placed after SwanLabCallback in the callback list.
    """

    def _init_callback(self) -> None:
        # SwanLabCallback has swapped the logger to contain only SwanLabOutputFormat;
        # add a stdout output back here to restore the verbose=1 scrolling table.
        self.model.logger.output_formats.append(HumanOutputFormat(sys.stdout))

    def _on_step(self) -> bool:
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 PPO CartPole 训练")
    parser.add_argument(
        "--gui", action="store_true",
        help="训练结束后弹出 GUI 窗口演示智能体（默认关闭，仅输出得分）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    # ==========================================
    # Phase 1: Training
    # ==========================================
    env = gym.make("CartPole-v1")

    # Print environment info (state space, action space, boundary thresholds)
    print("=" * 50)
    print("CartPole-v1 环境信息")
    print("=" * 50)
    print(f"  观测空间:  {env.observation_space}")
    print(f"  动作空间:  {env.action_space}")
    print(f"  观测上限:  {env.observation_space.high}")
    print(f"  观测下限:  {env.observation_space.low}")
    print(f"  终止条件:  位置 > ±{env.unwrapped.x_threshold}, "
          f"角度 > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
          f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)")
    print("=" * 50)

    model = PPO("MlpPolicy", env, verbose=1)

    print("开始训练（带 SwanLab 日志）...")
    swanlab_cb = SwanLabCallback(
        project="cartpole-ppo",
        experiment_name="PPO-CartPole-v1",
        mode="local",
    )
    model.learn(
        total_timesteps=80000,
        callback=[swanlab_cb, RestoreStdoutLog(), LogApproxKL()],
    )

    # Evaluate
    mean_reward, std_reward = evaluate_policy
    print(f"训练完成！平均奖励: {mean_reward} +/- {std_reward}")

    model.save("output/ppo_cartpole")
    env.close()

    # ==========================================
    # Phase 2: Demonstrate the learned results
    # ==========================================
    print("\n正在展示智能体的学习成果...")
    render_mode = "human" if args.gui else None
    vis_env = gym.make("CartPole-v1", render_mode=render_mode)
    model = PPO.load("output/ppo_cartpole")

    for episode in range(5):
        obs, info = vis_env.reset()
        done, truncated, score = False, False, 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = vis_env.step(action)
            score += reward
        print(f"  回合 {episode + 1} 得分: {score}")

    vis_env.close()

    if args.gui:
        print("\nGUI 演示结束。")
    else:
        print("\n提示: 加 --gui 可弹出小车动画窗口查看演示效果。")

    print("SwanLab 实验看板: swanlab watch swanlog")


if __name__ == "__main__":
    main()
