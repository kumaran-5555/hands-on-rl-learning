---
title: E.1 Mathematical Objects and Linear Algebra
---

# E.1 Mathematical Objects and Linear Algebra

Chapter 3 introduced the Bellman equation $V(s) = R(s) + \gamma\sum_{s'}P(s'|s,a)V(s')$, which describes the value of a single state. In actual computation, however, three problems appear in sequence: how to express the equations for all states at once, how to approximate values when the state space is too large, and how to keep the iterative process stable. Module E.1 shows the linear algebra tools that answer each problem and how those tools build on one another.

![Two-state Bellman equation diagram](../../appendix_math/images/rl-two-state-bellman.svg)

## Content Overview

| Problem               | Difficulty                              | Mathematical tool introduced                | Key formula                                                     | Link to Chapter 3        |
| --------------------- | --------------------------------------- | ------------------------------------------- | --------------------------------------------------------------- | ------------------------ |
| Too many equations    | 1000 states = 1000 equations            | Vectors, matrices, linear systems           | **v** = (I - gamma P)^-1 **r**                                  | Mathematical core of DP  |
| State space too large | Too many states for a value table       | Dot products, norms, function approximation | v_hat(s) = **w**^T **x**(s)                                     | Mathematical core of DQN |
| Training stability    | Training may diverge, explode, or drift | Eigenvalues, weighted norms, trust regions  | rho(gamma P) <= gamma < 1, Delta theta^T F Delta theta <= delta | Mathematical core of PPO |

## Reading Path

| Article                                                                                   | Question it answers                                                        | Corresponding problem      |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------- |
| [E.1.1 Scalars, Vectors, and Matrices](./linear-algebra-basics)                           | How do we represent states, values, and transitions?                       | Too many equations, basics |
| [E.1.2 Matrix Form of the Bellman Equation](./linear-algebra-bellman)                     | Can 1000 Bellman equations be compressed into one?                         | Too many equations         |
| [E.1.3 Dot Products, Norms, and Function Approximation](./linear-algebra-function-approx) | What if there are too many states to store? How do we measure update size? | State space too large      |
| [E.1.4 Convergence, Eigenvalues, and Trust Regions](./linear-algebra-advanced)            | Will training explode? How can parameters be updated safely?               | Training stability         |
| [E.1.5 Formula Review and Exercises](./linear-algebra-formulas-exercises)                 | Revisit Chapter 3 from this perspective                                    | Full review                |

Read E.1.1 through E.1.4 in order, then use E.1.5 for review and practice. If a concept is already familiar, you can jump directly to the corresponding article.
