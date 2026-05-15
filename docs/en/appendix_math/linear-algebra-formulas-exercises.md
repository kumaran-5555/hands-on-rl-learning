---
title: E.1.5 Formula Review and Exercises
---

# E.1.5 Formula Review and Exercises

> **Prerequisites**: This page summarizes all formulas in module E.1. It is best read after [E.1.1](./linear-algebra-basics) through [E.1.4](./linear-algebra-advanced). If this is your first pass, read the main articles first.

---

## Revisiting Chapter 3

You now have the tools of vectors, matrices, dot products, norms, and eigenvalues. Looking back at Chapter 3, many formulas that may have seemed acceptable by definition have cleaner matrix expressions underneath.

### Bellman Equation: From State-by-State Writing to One Matrix Line

In Chapter 3, the Bellman expectation equation was written as:

$$
V^\pi(s) = \sum_{a} \pi(a|s)\left[R(s,a) + \gamma\sum_{s'} P(s'|s,a)V^\pi(s')\right].
$$

With matrices, $n$ such equations become one line:

$$
\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}.
$$

The closed-form solution is $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$. The DP methods in Chapter 3 repeatedly apply $v_{k+1} = r + \gamma Pv_k$; in essence, they iteratively approximate this closed-form solution.

### TD Error: Incremental Updates and the Matrix Equation

The TD update in Chapter 3 is:

$$
V(s) \leftarrow V(s) + \alpha\left[r + \gamma V(s') - V(s)\right].
$$

In vector language, this can be written as:

$$
\boldsymbol{v} \leftarrow \boldsymbol{v} + \alpha \cdot \boldsymbol{e}_s \cdot \delta,
$$

where $\boldsymbol{e}_s$ is the one-hot vector for state $s$, with only the $s$-th position equal to $1$, and $\delta = r + \gamma V(s') - V(s)$ is the TD Error. This and the Bellman matrix form $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ from E.1.2 are two expressions of the same object: the former is an incremental update that changes one component at a time, while the latter is a global equation that all components satisfy simultaneously.

### Tabular Q-Learning: A Special Case of One-Hot plus Dot Product

Chapter 3's Q-Learning stores one number for every state-action pair. From the linear-algebra perspective, this is linear approximation with one-hot features: $Q(s,a) = \boldsymbol{w}^\top \boldsymbol{x}(s,a)$, where $\boldsymbol{x}(s,a)$ is a one-hot vector. Tabular lookup is a special case of linear approximation.

### Policy Gradient Updates: The Stage for Norms and Trust Regions

The second route in Chapter 3 defined a policy objective $J(\theta)$. When updating parameters, gradient clipping limits $\|\boldsymbol{g}\|_2 \leq c$, and the trust-region constraint $\Delta\theta^\top F\,\Delta\theta \leq \delta$ keeps training safe. These are all uses of linear algebra tools for stability.

---

## Concept Map

The table below organizes all concepts in module E.1 by dependency. You can read down the dependency chain or use it as an index to jump to a concept.

| Concept                         | Core formula                                         | Depends on            | Role in RL                            |
| ------------------------------- | ---------------------------------------------------- | --------------------- | ------------------------------------- |
| Scalar                          | r=2, gamma=0.9                                       | None                  | Rewards and hyperparameters           |
| Vector                          | **v** in R^n                                         | Scalar                | Stores values of all states           |
| Matrix                          | P in R^(n x n)                                       | Vector                | Stores transitions among all states   |
| Matrix multiplication           | (Pv)\_i = sum_j P_ij v_j                             | Matrix + vector       | Probability-weighted future value     |
| Bellman matrix form             | **v** = **r** + gamma P **v**                        | Matrix multiplication | Value recursion equation              |
| Linear-system solution          | **v** = (I - gamma P)^-1 **r**                       | Bellman matrix form   | Theoretical closed-form solution      |
| Dot product                     | **w**^T **x** = sum_i w_i x_i                        | Vector                | Linear value or policy approximation  |
| L2 norm                         | $\lVert\boldsymbol{x}\rVert_2 = \sqrt{\sum_i x_i^2}$ | Vector                | Gradient clipping and regularization  |
| Eigenvalue                      | A **u** = lambda **u**                               | Matrix                | Analyzes matrix scaling behavior      |
| Spectral radius and convergence | rho(gamma P) <= gamma < 1                            | Eigenvalues           | Value-iteration convergence guarantee |
| Trust-region constraint         | (theta-theta_old)^T F (theta-theta_old) <= delta     | Norms + eigenvalues   | Safe update in TRPO                   |

Reading from top to bottom, each concept adds a new capability on top of the previous one. If a concept feels unclear, return to the row named in its "Depends on" column.

---

## From Difficulties to Tools

The central thread of module E.1 is moving from "cannot compute it" to "can compute it safely."

| Stage        | Difficulty                              | Mathematical tool                   | Key formula                                                        |
| ------------ | --------------------------------------- | ----------------------------------- | ------------------------------------------------------------------ |
| Difficulty 1 | 1000 states = 1000 equations            | Vectors, matrices, equation systems | **v** = (I - gamma P)^-1 **r**                                     |
| Difficulty 2 | Too many states for the value table     | Dot products, norms                 | v_hat(s) = **w**^T **x**(s), $\lVert\boldsymbol{g}\rVert_2 \leq c$ |
| Difficulty 3 | Training may diverge, explode, or drift | Eigenvalues, weighted norms         | rho(gamma P) <= gamma < 1, Delta theta^T F Delta theta <= delta    |

---

## Common Pitfalls

1. **Treating matrices as abstract symbols.** In the Bellman equation, every row of transition matrix $P$ is a probability table describing where the next state goes from the current state.
2. **Thinking inversion is the practical algorithm.** $\boldsymbol{v}=(I-\gamma P)^{-1}\boldsymbol{r}$ is a theoretical closed-form solution. Real large-scale problems usually use iteration or function approximation, which is exactly the DP -> MC -> TD progression in Chapter 3.
3. **Ignoring vector direction.** Gradients, advantage updates, and trust-region constraints care not only about numerical magnitude, but also about direction in parameter space.
4. **Confusing norms and regularization.** A norm is a measurement tool, answering "how large." Regularization adds a norm to the training objective to constrain parameters. Gradient clipping limits the magnitude of a single update. These are three different uses.
5. **Thinking the L2 norm is the only norm.** The L1 norm encourages sparse solutions, the Frobenius norm measures matrix size, and weighted norms, or quadratic forms, account for direction sensitivity. Different settings use different norms.

---

## Exercises

### Basic

1. **Bellman equation by hand.** If $\gamma=0.9$, $v_1=1+0.9v_2$, and $v_2=2+0.9v_1$, compute $v_1$ and $v_2$ by hand.

2. **Write the matrix.** Write the corresponding $\boldsymbol{r}$ and $P$ for the previous problem, then verify whether $\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}$ equals your hand calculation.

3. **Gradient clipping.** If the gradient is $[6,8]^\top$ and the maximum norm is $5$, what is the clipped gradient?

### Intermediate

4. **Eigenvalue calculation.** What are the eigenvalues of $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$? It is an upper-triangular matrix; what pattern do its eigenvalues follow?

5. **Three-state Bellman equation.** There are three states, transition matrix $P = \begin{bmatrix} 0.2 & 0.5 & 0.3 \\ 0.0 & 0.4 & 0.6 \\ 0.7 & 0.3 & 0.0 \end{bmatrix}$, reward vector $\boldsymbol{r}=[1, 2, 3]^\top$, and $\gamma=0.5$. Write the Bellman matrix form without solving it.

6. **Dot-product approximation.** State $s$ has features $\boldsymbol{x}(s)=[0.3, -0.5, 0.8]^\top$, and weights are $\boldsymbol{w}=[2, -1, 3]^\top$. Compute $Q(s,a) = \boldsymbol{w}^\top\boldsymbol{x}(s)$. If the true value of $Q(s,a)$ is $5$, what is the error?

### Challenge

7. **Convergence-speed estimate.** Suppose $\gamma=0.95$ and the initial error is $\|\boldsymbol{e}_0\|=100$. How many Bellman updates are needed at minimum to make the error less than $0.01$? Hint: $\|\boldsymbol{e}_k\| \leq \gamma^k \|\boldsymbol{e}_0\|$.

8. **Trust-region check.** Let $F = \begin{bmatrix} 2 & 0 \\ 0 & 8 \end{bmatrix}$, $\Delta\theta = [0.3, 0.1]^\top$, and $\delta = 0.5$. Is this update inside the trust region? What about $\Delta\theta = [0.3, 0.2]^\top$? Explain why the second direction is more expensive.
