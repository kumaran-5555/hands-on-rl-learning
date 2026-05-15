---
title: Environment Setup Guide
---

# Environment Setup Guide

> **Goal of this section**: build a complete development environment for this course from scratch, covering Python, PyTorch, the RL toolchain, and LLM training frameworks. Follow the steps end to end, and you will be able to run every experiment in the book.

## Minimal Working Setup (Get Running in 5 Minutes)

If you cannot wait to start coding, the following 4 steps are enough to run all experiments in Chapters 1 through 6. The remaining simulators and LLM frameworks can be installed later as needed.

::: code-group

```bash [conda (recommended)]
conda create -n rl-course python=3.10 -y
conda activate rl-course
pip install torch torchvision
pip install gymnasium stable-baselines3[extra]
pip install numpy scipy matplotlib tqdm
```

```bash [venv]
python3.10 -m venv rl-course
source rl-course/bin/activate   # Windows: rl-course\Scripts\activate
pip install torch torchvision
pip install gymnasium stable-baselines3[extra]
pip install numpy scipy matplotlib tqdm
```

:::

After installation, verify with the following code. If a `CartPole` window pops up, everything is ready:

```python
import gymnasium as gym
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print("Minimal environment verified!")
```

::: details No GPU? You can still run the course.
The first half of the course (CartPole, DQN, etc.) can be trained on CPU; a GPU is not required. The second half on LLM fine-tuning (Chapters 7-10) recommends at least 24 GB of VRAM, or you can use the free GPUs on [Google Colab](https://colab.research.google.com/).
:::

---

**The following is the complete installation guide. Consult it as needed.**

## Python Environment

We recommend **Python 3.10+** (3.10, 3.11, or 3.12 are all fine). We suggest using conda to manage environments, as it makes switching CUDA versions easier.

```bash
# Option 1: conda (recommended)
conda create -n rl-course python=3.10 -y
conda activate rl-course

# Option 2: venv (lightweight)
python3.10 -m venv rl-course
source rl-course/bin/activate  # Linux/macOS
# rl-course\Scripts\activate   # Windows
```

::: tip Why Python 3.10?
PyTorch 2.x, transformers 4.x, and gymnasium have the best compatibility with 3.10. Python 3.12 also works, but some older libraries may not yet be fully adapted.
:::

## Installing PyTorch

PyTorch is the foundation for all deep learning experiments in this course. Choose the installation command that matches your hardware.

**Step 1: Check your GPU driver**

```bash
# Check NVIDIA driver and CUDA version
nvidia-smi
```

The upper-right corner of the output shows `CUDA Version: 12.x`, which is the highest CUDA version supported by your installed driver.

**Step 2: Install PyTorch**

```bash
# CUDA 12.4 (recommended for most modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (for older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only (no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (M1/M2/M3/M4 Macs)
pip install torch torchvision
```

**Step 3: Verify PyTorch GPU**

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Core RL Packages

These packages are used throughout the first half of the course (Chapters 1-6).

```bash
# Gymnasium (the successor to OpenAI Gym; the foundation for all environments)
pip install gymnasium

# Stable-Baselines3 (a well-tested RL algorithms library)
pip install stable-baselines3[extra]

# Scientific computing and visualization
pip install numpy scipy matplotlib seaborn pandas

# Progress bars and logging
pip install tqdm tensorboard wandb
```

## Simulator Environments

Different chapters use different simulators. Install them as needed.

```bash
# Chapter 11: PyBullet robotics simulation
pip install pybullet

# Chapter 4: Atari games (requires ale-py)
pip install "gymnasium[atari,accept-rom-license]"
pip install ale-py

# Chapter 4: ViZDoom first-person 3D
pip install vizdoom

# Chapter 4: stable-retro (classic games)
pip install stable-retro
```

**MuJoCo installation (optional, for high-fidelity robotics simulation)**

```bash
# MuJoCo is now free and open source; install directly via pip
pip install mujoco
# To use MuJoCo environments in gymnasium
pip install "gymnasium[mujoco]"
```

::: warning MuJoCo deep learning version
MuJoCo requires GPU rendering support. On headless servers, you need to set `export MUJOCO_GL=egl` or `export MUJOCO_GL=osmesa`.
:::

**Isaac Lab installation (optional, for GPU-parallel robotics simulation)**

Isaac Lab is the successor to NVIDIA Isaac Gym. It supports training thousands of robots in parallel on GPU and is suitable for large-scale robotics RL research. Requires an NVIDIA GPU + Linux.

```bash
# Isaac Lab depends on Isaac Sim; install Isaac Sim first
pip install isaacsim[all]

# Clone the Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# Verify installation
python -c "import isaacsim; print('Isaac Lab ready')"
```

::: warning Isaac Lab system requirements
Isaac Lab requires Linux (Ubuntu 22.04 recommended) and an NVIDIA RTX series GPU (at least 8 GB VRAM). macOS is not supported. If you use macOS, skip Isaac Lab and use PyBullet or MuJoCo as alternatives.
:::

**Unity ML-Agents installation (optional, for 3D game RL)**

Unity ML-Agents lets you train RL agents inside 3D environments built with the Unity engine, suitable for tasks like platform jumping, obstacle avoidance, and other tasks requiring spatial reasoning.

```bash
# Install the Python-side mlagents package
pip install mlagents-envs

# If you need training functionality (includes PyTorch dependencies)
pip install mlagents

# Verify installation
python -c "from mlagents_envs.environment import UnityEnvironment; print('ML-Agents ready')"
```

Using ML-Agents also requires downloading or building Unity environments (`.exe` / `.app` / Linux executables). Pre-built environments are available from [ML-Agents GitHub Releases](https://github.com/Unity-Technologies/ml-agents/releases). For detailed usage instructions, see [Learning Resources and Reproduction Projects](../appendix_game_projects/intro).

::: tip Unity ML-Agents use cases
The unique value of ML-Agents is **3D spatial reasoning**: Atari uses 2D pixels, CartPole uses low-dimensional vectors, while ML-Agents provides a complete 3D physics environment (gravity, collisions, occlusion). If your research involves visual navigation, spatial reasoning, or multi-agent 3D coordination, ML-Agents is a strong complement to Gymnasium/PyBullet.
:::

## LLM Training Frameworks

The second half of the course (Chapters 7-10) involves large model alignment training.

```bash
# Hugging Face ecosystem
pip install transformers datasets accelerate peft

# TRL (Transformer Reinforcement Learning, core of DPO/PPO training)
pip install trl

# Quantized inference (optional, saves VRAM)
pip install bitsandbytes

# Evaluation tools
pip install lm-eval
```

::: tip Version compatibility
If you encounter version conflicts, you can install a tested combination with the following command:

```bash
pip install "transformers==4.57.3" "trl==0.24.0" "datasets==4.4.1" "accelerate==1.10.1" "peft==0.17.1"
```

:::

## GPU Driver Checklist

Before starting training, confirm that all of the following are working properly.

| Check item    | Command                          | Expected result                   |
| ------------- | -------------------------------- | --------------------------------- |
| NVIDIA driver | `nvidia-smi`                     | Shows driver version and GPU info |
| CUDA version  | `nvcc --version`                 | Matches PyTorch compile version   |
| cuDNN         | `torch.backends.cudnn.version()` | Returns a version number          |
| PyTorch GPU   | `torch.cuda.is_available()`      | `True`                            |

```python
# One-click check script
import torch
print("=" * 50)
print("Environment Check Report")
print("=" * 50)
print(f"Python:       {__import__('sys').version}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda or 'CPU'}")
if torch.cuda.is_available():
    print(f"cuDNN:        {torch.backends.cudnn.version()}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Apple MPS:    {torch.backends.mps.is_available()}")
print("=" * 50)
```

## Common Installation Problems and Fixes

### Problem 1: CUDA Version Mismatch

**Symptoms**: `RuntimeError: CUDA out of memory` or `Found no NVIDIA driver`

**Cause**: The CUDA version PyTorch was compiled with does not match the system driver.

```bash
# Check the CUDA version used by the current PyTorch
python -c "import torch; print(torch.version.cuda)"

# If mismatched, reinstall the corresponding version of PyTorch
# Uninstall first
pip uninstall torch torchvision -y
# Then install the matching version (see the PyTorch installation section)
```

### Problem 2: Package Version Conflicts

**Symptoms**: `ImportError` or `AttributeError` indicating an incorrect package version.

```bash
# Check versions of all installed packages
pip list | grep -E "torch|gymnasium|transformers|trl"

# Force reinstall to resolve conflicts
pip install --force-reinstall <package_name>
```

### Problem 3: Apple Silicon (M-series chips) Compatibility

```bash
# M-series chips use MPS acceleration
# Make sure the arm64 version of PyTorch is installed
python -c "import platform; print(platform.processor())"
# Should output arm64

# Some RL environments do not support rendering on Mac; use headless mode
# export PYOPENGL_PLATFORM=egl
```

### Problem 4: MuJoCo Rendering Failure

```bash
# On headless servers, set the environment variable
export MUJOCO_GL=egl
# Or use osmesa
export MUJOCO_GL=osmesa
```

## Global Verification Script

Run the following script. If all outputs show "OK", the environment setup is complete.

```python
"""
Course environment one-click verification script
Run: python verify_env.py
"""
import sys

checks = []

# 1. Python version
ver = sys.version_info
checks.append(("Python >= 3.10", ver >= (3, 10)))

# 2. Core packages
for pkg in ["numpy", "matplotlib", "scipy", "tqdm"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# 3. PyTorch
try:
    import torch
    checks.append(("PyTorch", True))
    checks.append(("CUDA GPU", torch.cuda.is_available()))
except ImportError:
    checks.append(("PyTorch", False))

# 4. RL environments
try:
    import gymnasium
    checks.append(("Gymnasium", True))
except ImportError:
    checks.append(("Gymnasium", False))

# 5. LLM frameworks
for pkg in ["transformers", "datasets", "peft", "accelerate", "trl"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# Output report
print("\n" + "=" * 45)
print("  Environment Verification Report")
print("=" * 45)
for name, ok in checks:
    status = "OK" if ok else "MISSING"
    print(f"  {name:<25} {status}")
print("=" * 45)
passed = sum(1 for _, ok in checks if ok)
print(f"  Passed: {passed}/{len(checks)}")
print("=" * 45 + "\n")
```

## Package Overview

| Package           | Recommended version | Purpose                                      | Relevant chapters |
| ----------------- | ------------------- | -------------------------------------------- | ----------------- |
| Python            | 3.10+               | Runtime                                      | Entire book       |
| PyTorch           | 2.1+                | Deep learning framework                      | Entire book       |
| gymnasium         | 0.29+               | RL environment interface                     | Ch1, Ch3-Ch6      |
| stable-baselines3 | 2.2+                | Pre-packaged RL algorithms                   | Ch1, Ch4-Ch6      |
| numpy             | 1.24+               | Numerical computing                          | Entire book       |
| matplotlib        | 3.7+                | Visualization and plotting                   | Entire book       |
| pybullet          | 3.2+                | Robotics simulation                          | Ch11              |
| mujoco            | 3.0+                | High-fidelity physics simulation             | Ch11              |
| isaacsim          | 4.0+                | GPU-parallel robotics simulation (Isaac Lab) | Ch11, Ch12        |
| mlagents          | 1.0+                | Unity 3D game RL environments                | Appendix          |
| ale-py            | 0.8+                | Atari emulator                               | Ch4               |
| transformers      | 4.45+               | LLM model loading                            | Ch7-Ch10          |
| trl               | 0.12+               | LLM reinforcement learning training          | Ch7-Ch8           |
| peft              | 0.13+               | Parameter-efficient fine-tuning (LoRA)       | Ch7-Ch10          |
| accelerate        | 1.0+                | Distributed training                         | Ch7-Ch10          |
| datasets          | 3.0+                | Dataset loading                              | Ch7-Ch10          |
| wandb             | 0.16+               | Experiment tracking                          | Entire book       |
| tensorboard       | 2.15+               | Training visualization                       | Entire book       |

::: tip No need to install everything at once
We recommend installing packages as you progress through the course: start with the basics (Python environment, PyTorch, core RL packages), then install simulator environments and LLM frameworks when you reach the corresponding chapters. This avoids having to resolve too many dependency conflicts at once.
:::
