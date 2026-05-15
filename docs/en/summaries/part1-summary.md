---
title: Part I Summary
---

# Part 1: Practical Foundations - Core Concepts Recap

## What Did We Learn in This Part?

In the first two chapters, we ran two complete reinforcement learning training experiments from scratch.

In Chapter 1, we trained a CartPole agent with Stable Baselines3. This is the "Hello World" of reinforcement learning.
In Chapter 2, we used the TRL library to run DPO preference alignment on the Qwen2.5-0.5B model, and directly observed the model learning to distinguish good answers from bad ones.

Together, these two chapters gave us the following takeaways:

- **The core loop**: RL is the loop "observe state -> choose action -> receive reward -> update policy." No matter how sophisticated an algorithm is, it is a refinement of this loop.
- **Policy**: a policy $\pi(a|s)$ is "given the current state, assign probabilities to each action." Training aims to make good actions increasingly likely.
- **Policy gradients**: the compact rule $\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a|s) \cdot G$ says that if the return $G$ of an action is positive, increase that action's probability.
- **DPO loss**: $\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$. The core idea is to make the implicit reward of the preferred response higher than that of the rejected one, thereby bypassing an explicit reward model.
- **Two toolchains**: traditional RL uses Gymnasium + Stable Baselines3; LLM alignment uses Transformers + TRL.

Now let us review these chapter by chapter.

## Chapter 1: CartPole - Your First RL Program

### The Agent-Environment Interaction Loop

Reinforcement learning is an iterative interaction process between an Agent and an Environment. Imagine playing a game: you see the screen (state), press a key (action), the game gives you points (reward), then the screen updates. Repeating this process, your score improves. This is RL.

In CartPole, the state $s$ observed by the agent is a 4D vector: cart position and velocity, pole angle and angular velocity. The agent chooses between two actions: push left or push right, i.e. $\mathcal{A} = \{0, 1\}$. The agent receives reward +1 for each time step it survives; when the pole falls, the episode ends.

The interaction can be summarized as a simple loop:

```python
obs, info = env.reset()
while True:
    action = model.predict(obs)  # policy: choose an action given the state
    obs, reward, done, truncated, info = env.step(action)  # environment returns next state and reward
    if done or truncated:
        break
```

This "observe -> decide -> act -> reward" loop is the core RL paradigm. Whether we later use DQN, PPO, or DPO, the essence is the same: how do we make the policy $\pi(a|s)$ better and better?

### Policy and Policy Gradient

A policy $\pi_\theta(a|s)$ is a mapping from states to an action-probability distribution. In CartPole, the policy is a two-layer fully-connected network with 64 hidden units per layer. It takes the 4 state values as input and outputs logits for 2 actions; after a softmax, these become probabilities.

We train this policy using **policy gradients**. The intuition is straightforward: if an action leads to a good outcome (large return $G_t$), increase that action's probability; otherwise decrease it.

Concretely, our goal is to maximize the expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$. The neural network parameters $\theta$ control only the policy $\pi_\theta(a|s)$, i.e., the probabilities of actions.

If we differentiate $J$ with respect to $\theta$, the first step produces a term like $\sum_a \nabla_\theta \pi_\theta(a|s) \cdot G_t$. But there is a problem: $\nabla \pi$ is not a probability distribution, so you cannot sample from it, while training is driven by sampling.

The standard fix is a mathematical trick: $\nabla \pi = \pi \cdot \nabla \log \pi$ (by the chain rule). Multiplying by $\pi$ turns the gradient back into an expectation:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

Now we can sample. Run one episode to obtain samples $(s_t, a_t, G_t)$, and use a **single-sample approximation**:

$$\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

Intuitively, $\nabla_\theta \log \pi_\theta(a_t|s_t)$ is the "knob" that changes the action probability, and $G_t$ tells you which direction is good: if the return is positive, increase the probability; if negative, decrease it.

The log has two practical benefits: it turns probability products into sums (more numerically stable), and since $\nabla \log \pi = \nabla \pi / \pi$, low-probability actions naturally receive larger gradients, which helps exploration. The full derivation is in [Chapter 5](../chapter05_policy_gradient/reinforce).

The corresponding loss is $\mathcal{L} = -\log \pi_\theta(a_t|s_t) \cdot G_t$. Note the minus sign: we perform gradient descent, so we minimize this loss to maximize expected return. This is the core of the REINFORCE algorithm.

During training, two key metrics typically show a "scissors crossing" pattern: episodic reward increases as the policy improves, while policy entropy (Entropy) $\mathcal{H} = -\sum_a \pi(a|s) \log \pi(a|s)$ decreases as the policy shifts from random exploration to more confident choices. This is a healthy sign.

### Training with Stable Baselines3

Thanks to the modern open-source ecosystem, the entire workflow above can be run in just a few lines:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment and agent
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

# Train - just this line
model.learn(total_timesteps=80000)

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

Here we used PPO, the most widely used policy-gradient method in practice. Its key idea is developed in detail in Chapter 6. For now, the only point you need is: PPO adds a **clipping mechanism** on top of policy gradients to prevent updates that are too large, which can cause training to collapse.

## Chapter 2: DPO - Teaching an LLM to "Speak Well"

### The Three Stages of Modern LLM Training

Modern LLM training usually has three stages.

The first stage is **pre-training**: the model learns "predict the next token" on massive text corpora, roughly equivalent to reading widely.

The second stage is **supervised fine-tuning (SFT)**: high-quality instruction-following data teaches the model to follow commands, roughly equivalent to learning etiquette.

The third stage is **reinforcement learning alignment (RL alignment)**: the model learns to distinguish good answers from bad ones, roughly equivalent to learning preferences and values.

DPO (Direct Preference Optimization) is a core method for the third stage.

### The DPO Loss

The key idea of DPO is elegant: instead of training a reward model and then optimizing with RL (traditional RLHF), we can directly use preference data to optimize the policy.

Suppose we have preference tuples $(x, y_w, y_l)$, where $x$ is the prompt, $y_w$ is the preferred answer, and $y_l$ is the rejected answer. DPO defines an **implicit reward**:

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

Here $\pi_\theta$ is the model being trained, and $\pi_{\text{ref}}$ is a frozen reference model (the pre-alignment model). $\beta$ is a hyperparameter controlling how far the model is allowed to deviate from the reference: larger $\beta$ yields more conservative updates.

Why is this an "implicit reward"? Traditional RLHF trains a separate reward model to score answers. DPO observes that a useful signal is already hidden in the probability ratio: if the model prefers an answer more than the reference does ($\pi_\theta > \pi_{\text{ref}}$), the implicit reward is positive; otherwise it is negative.

With implicit reward, DPO trains by forcing the preferred answer to have higher reward than the rejected answer:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

where $\sigma$ is the sigmoid function. Intuitively, this is a classification problem: given a pair, the model should identify which is better. When the preferred answer's implicit reward is much higher than the rejected answer's, the term inside $\sigma$ is large, $\log \sigma$ is close to 0, and the loss is small. If the model cannot yet tell them apart, the loss is large and the gradient pushes the model to improve.

### Training DPO with TRL

In practice, we used HuggingFace TRL, and the workflow is similarly concise:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# Load models: one trainable, one frozen reference
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Configure training
training_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=3,
    beta=0.1,  # KL penalty coefficient
)

# Train
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=preference_dataset,  # columns: prompt, chosen, rejected
    processing_class=tokenizer,
)
trainer.train()
```

Two metrics matter during training: **training loss** should decrease over time (the model learns to distinguish preference pairs), and **reward margin** (the implicit reward difference between preferred and rejected answers) should increase over time (the model's discrimination ability improves).

## Summary

These two chapters established two key points.

First, the core of RL is the Agent-Environment interaction loop: the policy chooses actions from states, the environment returns rewards and next states, and the policy improves itself from reward signals.

Second, RL can be used both for traditional control tasks (CartPole) and for LLM alignment (DPO). They share the same underlying logic: define an objective and optimize it via gradient descent.

In Part 2, we will build the mathematical foundation behind this logic - MDPs, Bellman equations, DQN, the policy gradient theorem, and PPO - to prepare for later LLM alignment chapters.

> **Next stop**: [Part 2: Theory and Methods](/chapter03_mdp/intro)
