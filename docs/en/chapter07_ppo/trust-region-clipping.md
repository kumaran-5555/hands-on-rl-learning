---
title: 7.3 Trust Region and Clipping
---

# 7.3 Trust Region and Clipping

In the previous sections, we already looked at PPO training curves on BipedalWalker, and we derived the mathematics of the clipped surrogate objective (review: [PPO Math Derivation](./ppo-math)). But one core question is still unanswered:

What exactly is the clipping mechanism protecting? Why can a simple policy-gradient method “collapse” in practice?

To answer that, we have to start from the risk of policy updates themselves.

::: tip Prerequisites for This Section

- [The policy-gradient update rule](../chapter05_policy_gradient/reinforce): clipping is designed to protect this update
- [The advantage function $A(s,a)$](../chapter06_actor_critic/advantage-function): the directional signal for policy updates
  :::

## The Instability of Naive Policy Gradients

Recall the policy-gradient update rule from Chapter 5:

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)$$

This rule says: if action $a$ has positive advantage $A(s,a) > 0$ (better than average), then we update parameters in the direction that increases $\pi(a|s)$. That sounds reasonable. The problem is that there is **no explicit bound on the update magnitude**.

Picture a concrete scenario. In some state, the policy assigns probability 0.6 to action $a_1$ and 0.4 to $a_2$. If a single update pushes $a_1$ all the way to 0.99, then $a_2$ drops to 0.01. But this change is driven by just the samples you happened to see in this batch. What if the high advantage was mostly luck? You have now “blocked off” an alternative that might actually be good.

Worse, once the policy changes sharply, previously collected data becomes less relevant, because that data was generated under the “old policy.”

This is the central dilemma of naive policy gradients:

**A single update has high variance, but the policy change is not easily reversible.**

There is no “undo” button if you update too aggressively.

## Importance Sampling

If we want to limit the update size, we first have to solve a basic mathematical mismatch: the training data is collected under the old policy $\pi_{\text{old}}$, but we want to optimize a new policy $\pi_\theta$. Can we reuse old data to evaluate how the new policy would behave?

Yes, using **importance sampling**. The key idea is that expectations under one distribution can be rewritten in terms of another distribution using a likelihood ratio:

$$\mathbb{E}_{a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \cdot f(a) \right] = \mathbb{E}_{a \sim \pi_\theta} [f(a)]$$

The ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

is called the **policy ratio**. It measures how the probability of the same state-action pair changes between the new and old policies:

- $r_t(\theta) = 1$: the new policy matches the old policy (no change)
- $r_t(\theta) > 1$: the new policy is more likely to take this action
- $r_t(\theta) < 1$: the new policy is less likely to take this action

With importance sampling, the policy-gradient objective can be rewritten as:

$$L^{\text{IS}}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} \cdot A_t \right] = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right]$$

This looks ideal: we can evaluate a new policy using old data. But it hides a dangerous trap: **if $r_t(\theta)$ becomes very large (say 10 or 100), then the gradient gets amplified by the same factor**. One optimization step can then cause a massive policy shift.

Importance sampling gives us the ability to reuse old data, but it does not guarantee that we use it safely.

## TRPO and a KL-Divergence Constraint

In 2015, Schulman et al. proposed TRPO (Trust Region Policy Optimization), which introduces a rigorous constraint on the policy update:

**After each update, the KL divergence between the old and new policies must not exceed a threshold $\delta$.**

$$\max_\theta \; \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right] \quad \text{s.t.} \quad \mathbb{E}_t \left[ D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta) \right] \leq \delta$$

The KL divergence $D_{\text{KL}}$ is a standard way to measure the “distance” between probability distributions. A typical value is $\delta = 0.01$, which means: after each update, the policy’s behavior distribution is only allowed to change slightly.

You can think of this as drawing a “trust region.” The policy is allowed to move safely inside the region, but it is not allowed to step outside.

TRPO is mathematically elegant, but it has a severe engineering drawback: **it requires (approximations of) second-order information, involving the Hessian**. For a neural network with millions of parameters, the Hessian dimension scales like the square of parameter count, which is far beyond what you can store or compute directly. TRPO uses techniques like conjugate gradient to approximate the solution, but it is still slow and complex.

More importantly, in LLM settings the policy network might be a 70B-parameter language model. Computing second-order updates there is simply not realistic.

## PPO’s Clipping Mechanism

In 2017, Schulman introduced PPO (Proximal Policy Optimization). The key insight of PPO is:

**Instead of precisely solving a trust-region constrained optimization problem (as TRPO does), we can directly clip “unsafe” updates.**

PPO’s clipped objective is:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \right) \right]$$

Do not let the formula intimidate you. We can unpack it step by step.

**Part 1: the unclipped objective** $r_t(\theta) \cdot A_t$.

This is just the importance-sampling form: if the new policy increases the probability of actions with positive advantage, the objective grows; if it decreases those probabilities, the objective shrinks.

**Part 2: the clipped objective** $\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t$.

Here we clamp the policy ratio into a safe interval:

$$r_t(\theta) \in [1-\varepsilon, \; 1+\varepsilon]$$

In other words, for a given sample $(s_t, a_t)$, we do not allow the new policy to assign a probability that is too much larger or smaller than the old policy.

**Part 3: take the minimum** $\min(\cdot)$.

This is the crucial detail. Why is it a minimum rather than a maximum?

- If $A_t > 0$, then making $r_t(\theta)$ larger increases $r_t(\theta)A_t$, and we would like to encourage it, but only up to a point. Once $r_t(\theta) > 1+\varepsilon$, the clipped term becomes $(1+\varepsilon)A_t$, a constant. Taking the minimum means the objective is capped there, so the gradient becomes zero beyond the cap.
- If $A_t < 0$, then making $r_t(\theta)$ smaller decreases $r_t(\theta)A_t$ (which is “better” since the objective is maximized). But again, we only want this up to a point. Once $r_t(\theta) < 1-\varepsilon$, the clipped term becomes $(1-\varepsilon)A_t$, and taking the minimum again prevents the objective from pushing $r_t$ further away.

You can summarize PPO clipping in one sentence:

When the policy ratio leaves the safe range $[1-\varepsilon, 1+\varepsilon]$, the objective stops providing incentive to move further in that direction.

## Why Take the Minimum?

It helps to examine the two cases explicitly.

**Case 1: $A_t > 0$ (a “good” action).**

- If $r_t(\theta)$ increases slightly (still within the safe interval), then the objective increases, so we keep learning.
- If $r_t(\theta)$ becomes too large, i.e. $r_t(\theta) > 1+\varepsilon$, then the clipped term is capped at $(1+\varepsilon)A_t$.
- Taking the minimum means the objective becomes that capped value, so the gradient becomes zero. The update stops pushing $r_t$ larger.

**Case 2: $A_t < 0$ (a “bad” action).**

- If $r_t(\theta)$ decreases slightly (still within the safe interval), then $r_t(\theta)A_t$ increases (becomes less negative), so we keep learning.
- If $r_t(\theta)$ becomes too small, i.e. $r_t(\theta) < 1-\varepsilon$, then the clipped term is capped at $(1-\varepsilon)A_t$.
- Taking the minimum again makes the objective “flat” beyond the threshold, which removes the gradient signal to keep decreasing $r_t$.

If we used $\max$ instead of $\min$, clipping would fail: even when $r_t$ is already outside the safe interval, the objective could still prefer the unclipped term and continue to drive $r_t$ further away. The minimum is what enforces “no extra reward for moving further out of bounds.”

## Understanding Clipping with Code

Let’s make the behavior of clipping tangible with a short piece of code:

```python
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Visualize the PPO Clip objective
# ==========================================
epsilon = 0.2
r = np.linspace(0.0, 2.0, 500)  # policy ratio r_t(θ)

def ppo_clip_objective(r, A, eps=0.2):
    """PPO clipped objective: L = min(r * A, clip(r, 1-eps, 1+eps) * A)"""
    r_clipped = np.clip(r, 1 - eps, 1 + eps)
    return np.minimum(r * A, r_clipped * A)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Case: A > 0
A_pos = 1.0
obj_pos = ppo_clip_objective(r, A_pos)
ax1.plot(r, r * A_pos, 'b--', alpha=0.5, label='Unclipped: r × A')
ax1.plot(r, obj_pos, 'r-', linewidth=2, label='PPO: min(r×A, clip(r)×A)')
ax1.axvline(x=1+epsilon, color='gray', linestyle=':', label=f'1+ε={1+epsilon}')
ax1.axvline(x=1-epsilon, color='gray', linestyle=':', label=f'1-ε={1-epsilon}')
ax1.set_title('A > 0 (good action)')
ax1.set_xlabel('Policy ratio r_t(θ)')
ax1.set_ylabel('Objective value')
ax1.legend()

# Case: A < 0
A_neg = -1.0
obj_neg = ppo_clip_objective(r, A_neg)
ax2.plot(r, r * A_neg, 'b--', alpha=0.5, label='Unclipped: r × A')
ax2.plot(r, obj_neg, 'r-', linewidth=2, label='PPO: min(r×A, clip(r)×A)')
ax2.axvline(x=1+epsilon, color='gray', linestyle=':', label=f'1+ε={1+epsilon}')
ax2.axvline(x=1-epsilon, color='gray', linestyle=':', label=f'1-ε={1-epsilon}')
ax2.set_title('A < 0 (bad action)')
ax2.set_xlabel('Policy ratio r_t(θ)')
ax2.legend()

plt.suptitle('Behavior of the PPO Clip objective (ε=0.2)', fontsize=14)
plt.tight_layout()
plt.savefig("ppo_clip_visualization.png", dpi=150)
print("Saved visualization of the clipped objective")
```

If you run this code, you will see:

- When $A > 0$, the objective becomes flat once $r_t > 1.2$ (the gradient goes to zero, so the update stops pushing further).
- When $A < 0$, the objective becomes flat once $r_t < 0.8$ (again, the gradient goes to zero beyond the boundary).

This is PPO’s “safety belt”: once the ratio leaves the safe interval, the learning signal automatically disappears in the direction that would increase the change further.

## Sensitivity to $\varepsilon$

The choice of $\varepsilon$ directly affects training dynamics. Here is a practical rule-of-thumb summary:

| ε value | Update size | Training speed           | Stability        | Typical use case                         |
| ------: | ----------- | ------------------------ | ---------------- | ---------------------------------------- |
|    0.05 | very small  | very slow                | extremely stable | fine-tuning an already trained policy    |
|     0.1 | small       | slower                   | stable           | LLM alignment (large models are fragile) |
|     0.2 | medium      | moderate                 | moderate         | games/control tasks (a common default)   |
|     0.3 | larger      | faster                   | unstable         | quick experiments/simple tasks           |
|     0.5 | very large  | fast but often collapses | very unstable    | not recommended                          |

In LLM alignment settings, practitioners often use a smaller $\varepsilon$ (around 0.1 or even less), because the policy space of a language model is larger and more brittle: a single poorly controlled update can degrade the model’s general language ability (for instance, it can “forget” how to speak Chinese).

<details>
<summary>Question: If PPO clipping makes training “too conservative,” can we speed it up without sacrificing stability?</summary>

Several common strategies are used in practice:

1. **Adaptive ε**: PPO-PPG (Phasic Policy Gradient) suggests using a larger ε early in training and gradually shrinking it later, like “take big steps to explore first, then take small steps to refine.”
2. **More update epochs per batch**: PPO commonly performs multiple epochs (often 10) over the same batch of data. If clipping makes each step small, increasing the number of epochs can accumulate a meaningful overall update.
3. **Early stopping based on KL divergence**: monitor the KL divergence during optimization, and stop early if KL exceeds a threshold within an epoch. This effectively combines the TRPO idea (a KL constraint) with PPO’s clipping.

In practice, the second approach is the most common. PPO’s default `n_epochs=10` is already motivated by the idea that clipping limits per-step movement, so multiple passes are used to accumulate sufficient progress.

</details>

<details>
<summary>Question: TRPO is more theoretically principled. Why does industry almost always choose PPO?</summary>

Because in engineering practice, “simple and reliable” usually beats “theoretically perfect.”

TRPO requires second-order machinery (Hessian-vector products). On large models this is slow, complex to implement, and easy to get wrong. PPO, in contrast, can be implemented with a simple `torch.clamp` and a `min`, often in under ten lines of code.

The PPO paper (2017) also shows empirically that PPO matches TRPO and often performs as well or better across many tasks. One reason is that TRPO’s own second-order approximations introduce error; solving the constrained problem precisely does not necessarily outperform PPO’s simple heuristic clipping.

This tradeoff becomes even clearer in the LLM era. For a 70B-parameter policy, second-order optimization is simply not practical. OpenAI’s alignment training in InstructGPT and GPT-4 uses PPO rather than TRPO for exactly these reasons.

</details>

At this point, you should understand the full story behind PPO clipping: the collapse risk in naive policy gradients, the data reuse enabled by importance sampling, the KL-constrained trust region in TRPO, and PPO’s clipped approximation of that idea.

But PPO still has another key component we have not expanded here: GAE (Generalized Advantage Estimation). In LLM alignment, GAE also leads us to the largest practical burden: the Reward Model. Let’s continue with [GAE, Reward Models, and LLM Alignment](./gae-reward-model).
