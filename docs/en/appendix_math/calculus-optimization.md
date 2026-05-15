---
title: E.3 Calculus and Optimization
---

# E.3 Calculus and Optimization

Training a reinforcement learning agent is, at its core, a matter of adjusting parameters: making the average return higher and higher, or making prediction error smaller and smaller. The underlying language for this process is calculus. Derivatives tell us "which way to move", gradients tell us "how each parameter should move", and the chain rule lets that signal travel backward through the entire computation graph.

This section follows that thread. We start from functions and rates of change, move step by step to derivatives, gradients, and the chain rule, then see how these tools appear in policy gradients, Taylor approximations, PPO clipping, and GRPO normalization.

![Gradient update diagram](../../appendix_math/images/rl-gradient-update.svg)

## Roadmap

| Article                                                                 | Mathematical pace                                                  | Role in reinforcement learning                                                |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| [E.3.1 Derivatives, Gradients, and the Chain Rule](./calculus-basics)   | Function -> derivative -> gradient -> chain rule                   | Understand how parameters affect the objective function                       |
| [E.3.2 From Gradients to Policy Gradients](./calculus-policy-gradient)  | Log-probability gradient -> return weighting -> advantage function | Derive the update direction behind "increase the probability of good actions" |
| [E.3.3 Optimization Stability: PPO and Adam](./calculus-ppo)            | Probability ratio -> clipping -> adaptive step size                | Control policy update size and gradient noise                                 |
| [E.3.4 Derivation Tools: Log Trick and Taylor](./calculus-derivations)  | Log-derivative trick -> Taylor expansion -> second-order intuition | Understand the derivation skeleton of policy gradients and PPO                |
| [E.3.5 Complete Optimization Formulas](./calculus-advanced-formulas)    | Full expressions for PG, DQN, GAE, PPO, GRPO                       | Connect modern RL training objectives                                         |
| [E.3.6 Summary, Formulas, and Exercises](./calculus-formulas-exercises) | Formula review -> pitfalls -> exercises                            | Review and check understanding                                                |
