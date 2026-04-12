# 附录 D：环境安装指南

> **本附录目标**：从零开始搭建本课程所需的完整开发环境，涵盖 Python、PyTorch、RL 工具链和 LLM 训练框架。跟着步骤走完，你就能跑通全书所有实验。

## D.1 Python 环境准备

推荐使用 **Python 3.10+**（3.10、3.11 或 3.12 均可）。我们建议用 conda 管理环境，方便切换 CUDA 版本。

```bash
# 方式一：使用 conda（推荐）
conda create -n rl-course python=3.10 -y
conda activate rl-course

# 方式二：使用 venv（轻量级）
python3.10 -m venv rl-course
source rl-course/bin/activate  # Linux/macOS
# rl-course\Scripts\activate   # Windows
```

::: tip 为什么推荐 Python 3.10？
PyTorch 2.x、transformers 4.x 和 gymnasium 都对 3.10 有最好的兼容性。3.12 也可以，但部分旧版库可能还没适配。
:::

## D.2 PyTorch 安装

PyTorch 是本课程所有深度学习实验的基础。根据你的硬件选择对应的安装命令。

**第一步：检查 GPU 驱动**

```bash
# 查看 NVIDIA 驱动和 CUDA 版本
nvidia-smi
```

输出右上角会显示 `CUDA Version: 12.x`，这就是你驱动支持的最高 CUDA 版本。

**第二步：安装 PyTorch**

```bash
# CUDA 12.4（推荐，适合大多数新显卡）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8（适合旧显卡）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU 版本（没有 GPU 的同学）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon（M1/M2/M3/M4 Mac）
pip install torch torchvision
```

**第三步：验证 PyTorch GPU**

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

## D.3 核心 RL 包

这些是贯穿课程前半部分（Ch1-Ch6）的基础包。

```bash
# Gymnasium（Gym 的继任者，所有环境的基础）
pip install gymnasium

# Stable-Baselines3（封装好的 RL 算法库）
pip install stable-baselines3[extra]

# 科学计算与可视化
pip install numpy scipy matplotlib seaborn pandas

# 进度条与日志
pip install tqdm tensorboard wandb
```

## D.4 仿真环境安装

不同章节用到不同的仿真环境，可以按需安装。

```bash
# 第9章：PyBullet 机器人仿真
pip install pybullet

# 第4章：Atari 游戏（需要 ale-py）
pip install "gymnasium[atari,accept-rom-license]"
pip install ale-py

# 第4章：ViZDoom 第一人称3D
pip install vizdoom

# 第4章：stable-retro（经典游戏）
pip install stable-retro
```

**MuJoCo 安装（可选，用于高精度机器人仿真）**

```bash
# MuJoCo 现已免费开源，直接 pip 安装
pip install mujoco
# 如果要在 gymnasium 中使用 MuJoCo 环境
pip install "gymnasium[mujoco]"
```

::: warning MuJoCo 深度学习版本
MuJoCo 需要 GPU 渲染支持。在无头服务器上，需要设置 `export MUJOCO_GL=egl` 或 `export MUJOCO_GL=osmesa`。
:::

**Isaac Lab 安装（可选，用于 GPU 并行机器人仿真）**

Isaac Lab 是 NVIDIA Isaac Gym 的继任者，支持 GPU 上万级机器人并行训练，适用于大规模机器人 RL 研究。需要 NVIDIA GPU + Linux。

```bash
# Isaac Lab 依赖 Isaac Sim，需要先安装 Isaac Sim
pip install isaacsim[all]

# 克隆 Isaac Lab 仓库
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 验证安装
python -c "import isaacsim; print('Isaac Lab ready')"
```

::: warning Isaac Lab 系统要求
Isaac Lab 需要 Linux 系统（Ubuntu 22.04 推荐）和 NVIDIA RTX 系列 GPU（至少 8GB 显存）。macOS 不支持。如果你使用 macOS，可以跳过 Isaac Lab，使用 PyBullet 或 MuJoCo 作为替代。
:::

**Unity ML-Agents 安装（可选，用于 3D 游戏 RL）**

Unity ML-Agents 让你在 Unity 引擎构建的 3D 环境中训练 RL 智能体，适合平台跳跃、躲避障碍等需要空间推理的任务。

```bash
# 安装 Python 端的 mlagents 包
pip install mlagents-envs

# 如果你需要训练功能（包含 PyTorch 依赖）
pip install mlagents

# 验证安装
python -c "from mlagents_envs.environment import UnityEnvironment; print('ML-Agents ready')"
```

使用 ML-Agents 需要下载或自行构建 Unity 环境（`.exe` / `.app` / Linux 可执行文件）。预构建环境可从 [ML-Agents GitHub Releases](https://github.com/Unity-Technologies/ml-agents/releases) 获取。详细使用方式参见[附录经典项目](../appendix_game_projects/intro)。

::: tip Unity ML-Agents 适用场景
ML-Agents 的独特价值在于**3D 空间推理**：Atari 是 2D 像素，CartPole 是低维向量，而 ML-Agents 提供完整的 3D 物理环境（重力、碰撞、遮挡）。如果你的研究涉及视觉导航、空间推理或多智能体 3D 协作，ML-Agents 是 Gymnasium/PyBullet 之外的有力补充。
:::

## D.5 LLM 训练框架

课程后半部分（Ch7-Ch12）涉及大模型对齐训练。

```bash
# Hugging Face 生态
pip install transformers datasets accelerate peft

# TRL（Transformer Reinforcement Learning，DPO/PPO 训练的核心）
pip install trl

# 量化推理（可选，节省显存）
pip install bitsandbytes

# 评估工具
pip install lm-eval
```

::: tip 版本兼容性
如果遇到版本冲突，可以用以下命令安装经过测试的版本组合：

```bash
pip install "transformers>=4.45.0" "trl>=0.12.0" "peft>=0.13.0" "accelerate>=1.0.0"
```

:::

## D.6 GPU 驱动检查清单

在开始训练之前，确认以下几项都正常。

| 检查项      | 命令                             | 期望结果                |
| ----------- | -------------------------------- | ----------------------- |
| NVIDIA 驱动 | `nvidia-smi`                     | 显示驱动版本和 GPU 信息 |
| CUDA 版本   | `nvcc --version`                 | 与 PyTorch 编译版本匹配 |
| cuDNN       | `torch.backends.cudnn.version()` | 返回版本号              |
| PyTorch GPU | `torch.cuda.is_available()`      | `True`                  |

```python
# 一键检查脚本
import torch
print("=" * 50)
print("环境检查报告")
print("=" * 50)
print(f"Python:       {__import__('sys').version}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA 可用:    {torch.cuda.is_available()}")
print(f"CUDA 版本:    {torch.version.cuda or 'CPU'}")
if torch.cuda.is_available():
    print(f"cuDNN:        {torch.backends.cudnn.version()}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"显存:         {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"Apple MPS:    {torch.backends.mps.is_available()}")
print("=" * 50)
```

## D.7 常见安装问题与修复

### 问题 1：CUDA 版本不匹配

**症状**：`RuntimeError: CUDA out of memory` 或 `Found no NVIDIA driver`

**原因**：PyTorch 编译时的 CUDA 版本与系统驱动不匹配。

```bash
# 查看当前 PyTorch 使用的 CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 如果不匹配，重新安装对应版本的 PyTorch
# 先卸载
pip uninstall torch torchvision -y
# 再安装匹配的版本（参见 D.2 节）
```

### 问题 2：包版本冲突

**症状**：`ImportError` 或 `AttributeError`，提示某个包版本不对。

```bash
# 查看所有已安装包的版本
pip list | grep -E "torch|gymnasium|transformers|trl"

# 强制重装解决冲突
pip install --force-reinstall <包名>
```

### 问题 3：Apple Silicon (M 系列芯片) 兼容性

```bash
# M 系列芯片使用 MPS 加速
# 确保安装了 arm64 版本的 PyTorch
python -c "import platform; print(platform.processor())"
# 应该输出 arm64

# 部分 RL 环境在 Mac 上不支持渲染，使用 headless 模式
# export PYOPENGL_PLATFORM=egl
```

### 问题 4：MuJoCo 渲染失败

```bash
# 无头服务器上设置环境变量
export MUJOCO_GL=egl
# 或者用 osmesa
export MUJOCO_GL=osmesa
```

## D.8 全局验证脚本

运行以下脚本，如果全部输出 "OK"，说明环境搭建完成。

```python
"""
课程环境一键验证脚本
运行: python verify_env.py
"""
import sys

checks = []

# 1. Python 版本
ver = sys.version_info
checks.append(("Python >= 3.10", ver >= (3, 10)))

# 2. 核心包
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

# 4. RL 环境
try:
    import gymnasium
    checks.append(("Gymnasium", True))
except ImportError:
    checks.append(("Gymnasium", False))

# 5. LLM 框架
for pkg in ["transformers", "datasets", "peft", "accelerate", "trl"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# 输出报告
print("\n" + "=" * 45)
print("  环境验证报告")
print("=" * 45)
for name, ok in checks:
    status = "OK" if ok else "MISSING"
    print(f"  {name:<25} {status}")
print("=" * 45)
passed = sum(1 for _, ok in checks if ok)
print(f"  通过: {passed}/{len(checks)}")
print("=" * 45 + "\n")
```

## D.9 环境包总览

| 包名              | 推荐版本 | 用途                            | 相关章节     |
| ----------------- | -------- | ------------------------------- | ------------ |
| Python            | 3.10+    | 运行时                          | 全书         |
| PyTorch           | 2.1+     | 深度学习框架                    | 全书         |
| gymnasium         | 0.29+    | RL 环境接口                     | Ch1, Ch3-Ch6 |
| stable-baselines3 | 2.2+     | 封装好的 RL 算法                | Ch1, Ch4-Ch6 |
| numpy             | 1.24+    | 数值计算                        | 全书         |
| matplotlib        | 3.7+     | 可视化绘图                      | 全书         |
| pybullet          | 3.2+     | 机器人仿真                      | Ch9          |
| mujoco            | 3.0+     | 高精度物理仿真                  | Ch9          |
| isaacsim          | 4.0+     | GPU 并行机器人仿真（Isaac Lab） | Ch9, Ch13    |
| mlagents          | 1.0+     | Unity 3D 游戏 RL 环境           | 附录         |
| ale-py            | 0.8+     | Atari 模拟器                    | Ch4          |
| transformers      | 4.45+    | LLM 模型加载                    | Ch7-Ch12     |
| trl               | 0.12+    | LLM 强化学习训练                | Ch7-Ch8      |
| peft              | 0.13+    | 参数高效微调 (LoRA)             | Ch7-Ch12     |
| accelerate        | 1.0+     | 分布式训练                      | Ch7-Ch12     |
| datasets          | 3.0+     | 数据集加载                      | Ch7-Ch12     |
| wandb             | 0.16+    | 实验追踪                        | 全书         |
| tensorboard       | 2.15+    | 训练可视化                      | 全书         |

::: tip 不用一次装完
建议按课程进度安装：先装 D.1-D.3 的基础包，等学到对应章节再装仿真环境和 LLM 框架，避免一次性解决太多依赖冲突。
:::
