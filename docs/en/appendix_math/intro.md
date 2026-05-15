---
title: E. Mathematical Foundations for RL
---

# E. Mathematical Foundations for RL

If you opened this appendix, it is probably because a formula in the main text slowed you down. Maybe it was an expectation symbol inside a Bellman equation, a KL divergence term inside PPO, or the gradient operator that suddenly appears in the policy gradient theorem.

These formulas can look intimidating, but the underlying tools are not that many: scalars, vectors, matrices, probability, expectation, derivatives, gradients, entropy, and KL divergence. Once you understand what each word means, and how they combine inside reinforcement learning formulas, the notation becomes far less mysterious.

This appendix does not organize math by algorithm. It follows a more natural learning order:

mathematical objects -> linear operations -> probability and expectation -> stochastic estimation -> recursive equations -> optimization and gradients -> distribution distance -> full RL formulas.

## Learning Route

The whole appendix can be summarized as one line:

**mathematical objects -> linear algebra -> probability and expectation -> stochastic estimation -> Bellman recursion -> optimization and gradients -> distribution distance -> RL derivations.**

## Appendix Map

| Section                                                                             | Topic                                                                                   | Main question                                                         |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [E.1 Mathematical objects and linear algebra](./linear-algebra)                     | scalars, vectors, matrices, dot products, norms, linear equations                       | How do we write states, values, and parameters as computable objects? |
| [E.2 Probability, expectation, and stochastic estimation](./probability-statistics) | probability, conditional probability, random variables, expectation, variance, sampling | How do random trajectories become average value?                      |
| [E.3 Calculus and optimization](./calculus-optimization)                            | derivatives, gradients, chain rule, Taylor expansion, optimization algorithms           | Which direction should parameters move?                               |
| [E.4 Information theory and distribution distance](./information-theory)            | self-information, entropy, cross-entropy, KL, mutual information                        | How do we measure policy randomness and policy change?                |

## Suggested Reading Order

If you want to rebuild the mathematical foundations systematically, read in this order:

1. [E.1.1 Vector and matrix basics](./linear-algebra-basics): understand scalars, vectors, matrices, and matrix multiplication.
2. [E.1.2 Bellman equations in matrix form](./linear-algebra-bellman): see how value recursion becomes a linear system.
3. [E.2.1 Probability, conditional probability, and expectation](./probability-basics): learn random variables and expectation.
4. [E.2.2 Random trajectories and state value](./probability-value): connect expectation to returns and value functions.
5. [E.2.3 Monte Carlo and importance sampling](./probability-sampling): estimate values from samples when the environment model is unknown.
6. [E.3.1 Derivatives, gradients, and the chain rule](./calculus-basics): understand parameter changes and backpropagation.
7. [E.3.2 Policy gradients and advantage functions](./calculus-policy-gradient): apply gradients to policy optimization.
8. [E.4.1 Self-information, entropy, and exploration](./information-basics): understand policy randomness.
9. [E.4.2 Cross-entropy and KL divergence](./information-cross-entropy-kl): understand distances between distributions.
10. Finally, return to each module's formula summary and exercises.

If you only need one concept, it is completely fine to jump directly to the relevant page.

## A Running Example

Several sections reuse the same tiny two-state environment. There are two states, $s_1$ and $s_2$:

- in $s_1$, the agent receives reward $2$, then moves to $s_2$
- in $s_2$, the agent receives reward $1$, then moves back to $s_1$
- the discount factor is $\gamma = 0.5$

Let the state values be $v_1$ and $v_2$. Intuitively:

$$
\begin{aligned}
v_1 &= 2 + 0.5v_2, \\
v_2 &= 1 + 0.5v_1.
\end{aligned}
$$

This same example plays different roles in different modules:

- in linear algebra, it is a two-variable linear system
- in probability, it is "immediate reward + expected next-state value"
- in stochastic estimation, it can be approximated from sampled trajectories
- in optimization, it becomes a target for a value network or policy network
- in information theory, it connects to policy distributions, exploration, and update constraints

If you can translate a complicated formula back into this two-state example, math stops being a wall and becomes a tool.

## How to Use This Appendix

The sidebar divides the content into four math modules. You do not need to read everything at once. There are three useful modes:

1. **Systematic catch-up**: start from E.1.1 and read in order.
2. **Just-in-time lookup**: if Bellman matrix form is confusing, read E.1.2; if GAE is confusing, read the probability and calculus sections; if KL constraints are confusing, read E.4.2.
3. **Quick review**: use each module's formula summary and exercises after finishing the corresponding topic.

If the sidebar feels too large, read only the first page of each module first:

- [E.1.1 Scalars, vectors, and matrices](./linear-algebra-basics)
- [E.2.1 Probability, conditional probability, and expectation](./probability-basics)
- [E.3.1 Derivatives, gradients, and the chain rule](./calculus-basics)
- [E.4.1 Self-information, entropy, and exploration](./information-basics)

After these four pages, return to the detailed topics whenever a formula in the main text needs support.
