---
title: E.1.4 Convergence, Eigenvalues, and Trust Regions
---

# E.1.4 Convergence, Eigenvalues, and Trust Regions

> **Prerequisites**: [E.1.2 Bellman Matrix Form](./linear-algebra-bellman), especially $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$. [E.1.3 Dot Products and Norms](./linear-algebra-function-approx), especially the definition of the L2 norm.

---

## Overview

The previous two articles derived the matrix form of the Bellman equation, $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$, and discussed function approximation. Both share one premise: training repeatedly applies an update rule. Whether it is value iteration $v_{k+1} = r + \gamma P v_k$ or SGD updating weights $\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \boldsymbol{g}$, the process is iterative.

The previous article ended with two unresolved questions:

1. When we repeatedly apply $v_{k+1} = r + \gamma P v_k$, **how do we guarantee that the sequence converges instead of oscillating forever or diverging?**
2. When policy parameters are updated, gradient clipping limits the maximum step size, but **the same step size in different parameter directions can have very different effects on the policy distribution**. This requires a more refined constraint.

The answers involve three mathematical tools. They protect training stability from long-term, short-term, and fine-grained perspectives. This article establishes three core conclusions:

**Eigenvalues guarantee convergence**

$$
\boxed{\rho(\gamma P) \leq \gamma < 1 \quad\Longrightarrow\quad \|\boldsymbol{e}_{k}\| \leq \gamma^k \|\boldsymbol{e}_0\| \to 0}
$$

**Norms limit single-step magnitude**

$$
\boxed{\|\boldsymbol{g}_{clipped}\|_2 = \min\!\left(\|\boldsymbol{g}\|_2,\; c\right)}
$$

**Weighted norms give direction-sensitive safe updates**

$$
\boxed{\Delta\theta^\top F\,\Delta\theta \leq \delta}
$$

We now develop these ideas one by one.

---

## Convergence: Conditions and Guarantees for Bellman Updates

### Intuition for Convergence: From One Dimension to Two

Start with a simple numerical transformation:

$$
x_{k+1} = 0.5\,x_k.
$$

If $x_0=8$, then $8 \to 4 \to 2 \to 1 \to 0.5 \to \cdots$. Since each step multiplies by $0.5$, whose absolute value is less than $1$, the sequence converges to $0$.

Now move to two dimensions. Suppose a transformation multiplies the two components of a vector by $0.5$ and $0.3$:

$$
A = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.3 \end{bmatrix}.
$$

Then $A^k$ multiplies the first component by $0.5^k$ and the second by $0.3^k$. As $k$ grows, both components approach $0$.

The key observation is: **this transformation scales two directions by $0.5$ and $0.3$**. Those numbers are the **eigenvalues** of the matrix.

### Eigenvalues and Eigenvectors

More generally, if there exists a nonzero vector $\boldsymbol{u}$ and a scalar $\lambda$ such that:

$$
A\boldsymbol{u} = \lambda \boldsymbol{u},
$$

then $\boldsymbol{u}$ is an **eigenvector** of matrix $A$, and $\lambda$ is the corresponding **eigenvalue**. In other words, when $A$ acts on $\boldsymbol{u}$, it does not change the vector's direction; it only scales its length by $\lambda$.

### A Concrete Calculation

We first compute eigenvalues for an ordinary matrix, then apply the conclusion to Bellman updates and explain why the eigenvalues of $\gamma P$ must be less than $1$ in absolute value.

Consider a $2\times2$ matrix:

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}.
$$

The eigenvalues satisfy the characteristic equation $\det(A - \lambda I) = 0$:

$$
\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = 0.
$$

Expanding:

$$
(4-\lambda)(3-\lambda) - 2 = 0 \quad\Longrightarrow\quad \lambda^2 - 7\lambda + 10 = 0.
$$

Solving the quadratic:

$$
\lambda = \frac{7 \pm \sqrt{49-40}}{2} = \frac{7 \pm 3}{2}.
$$

Therefore $\lambda_1 = 5$ and $\lambda_2 = 2$. Geometrically, this matrix scales some directions by $5$ and other directions by $2$. When repeatedly applying $A$, the direction with $\lambda_1 = 5$ will dominate growth.

This gives the direct conclusion: **if the largest eigenvalue magnitude of a matrix is greater than $1$, repeatedly applying it can make vectors grow without bound; if all eigenvalue magnitudes are less than $1$, vectors shrink and the iteration converges**.

### Eigenvalue Argument for Bellman Update Convergence

Multiplying by $0.5$ converges in one dimension, and in two dimensions the same is true when all eigenvalue magnitudes are less than $1$. This conclusion applies directly to Bellman updates:

$$
\boldsymbol{v}_{k+1} = \boldsymbol{r} + \gamma P\boldsymbol{v}_k.
$$

Assume the true solution is $\boldsymbol{v}^*$ and satisfies $\boldsymbol{v}^* = \boldsymbol{r} + \gamma P\boldsymbol{v}^*$. Subtract the two equations:

$$
\boldsymbol{v}_{k+1} - \boldsymbol{v}^* = \gamma P(\boldsymbol{v}_k - \boldsymbol{v}^*).
$$

Let the error be $\boldsymbol{e}_k = \boldsymbol{v}_k - \boldsymbol{v}^*$. Then:

$$
\boldsymbol{e}_{k+1} = \gamma P\,\boldsymbol{e}_k.
$$

This recurrence has the same structure as the original one-dimensional example $x_{k+1} = 0.5\,x_k$; the only difference is that scalar multiplication has been replaced by matrix multiplication with $\gamma P$.

$P$ is a transition probability matrix, and each row sums to $1$, so its spectral radius, the largest absolute eigenvalue, satisfies $\rho(P) \leq 1$. After multiplying by the discount factor:

$$
\rho(\gamma P) \leq \gamma < 1.
$$

Thus every eigenvalue of $\gamma P$ has absolute value less than $1$. The error shrinks in every direction, so **Bellman updates must converge**.

This is the mathematical guarantee behind the Chapter 3 DP instruction "iterate until convergence." $\gamma < 1$ is not merely an engineering choice; it is the mathematical condition for convergence.

### How the Size of $\gamma$ Affects Convergence Speed

$\gamma$ affects not only whether the iteration converges, but also how fast it converges.

| gamma | rho(gamma P) | Convergence speed | Meaning                                   |
| ----- | ------------ | ----------------- | ----------------------------------------- |
| 0.1   | <= 0.1       | Very fast         | Focuses mostly on the immediate future    |
| 0.5   | <= 0.5       | Fairly fast       | Balances near and distant returns         |
| 0.9   | <= 0.9       | Slow              | Looks far ahead and needs more iterations |
| 0.99  | <= 0.99      | Very slow         | Strongly emphasizes long-term return      |

Numerical demonstration: suppose the initial error is $\|\boldsymbol{e}_0\|=10$. After $k$ iterations, the error bound is:

| Iterations k | Error with gamma=0.5 | Error with gamma=0.9 | Error with gamma=0.99 |
| ------------ | -------------------- | -------------------- | --------------------- |
| 1            | 5                    | 9                    | 9.9                   |
| 5            | 0.31                 | 5.9                  | 9.51                  |
| 10           | 0.01                 | 3.49                 | 9.04                  |
| 50           | approx 0             | 0.005                | 6.05                  |
| 100          | approx 0             | approx 0             | 3.66                  |

The closer $\gamma$ is to $1$, the slower convergence becomes, but the resulting value looks farther into the future. This is a common accuracy-efficiency tradeoff in RL.

### Contraction Mapping: A Formal Guarantee

Eigenvalues explain that error shrinks in every direction. Mathematically, there is a more unified way to express the same conclusion: a **contraction mapping**. It can be shown that the Bellman operator $\mathcal{T}\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ satisfies:

$$
\|\mathcal{T}\boldsymbol{u} - \mathcal{T}\boldsymbol{v}\|_\infty \leq \gamma \|\boldsymbol{u}-\boldsymbol{v}\|_\infty.
$$

This inequality says: **after one Bellman update, the distance between two estimates is at most $\gamma$ times what it was before**. Since $\gamma < 1$, the distance strictly decreases. This means the iteration has exactly one fixed point, the target value function $\boldsymbol{v}^*$.

<details>
<summary>Expand: Banach Fixed-Point Theorem</summary>

A contraction mapping has an elegant property, known as the Banach fixed-point theorem: in a complete metric space, a contraction mapping has exactly one fixed point, and repeated iteration converges to it. $\boldsymbol{v}^*$ is that fixed point.

The full theorem statement is not necessary here. The core conclusion is: **$\gamma < 1$ is the mathematical guarantee for Bellman update convergence**. If $\gamma = 1$, meaning no discounting, the update may fail to converge in some cases.

</details>

---

Eigenvalues guarantee convergence from a long-term perspective, but convergence only says that repeated iteration eventually reaches a fixed point. It does not guarantee that each individual update has controlled magnitude. The gradient clipping introduced in E.1.3 uses the L2 norm to limit the maximum step length and already prevents many training instabilities. But it has a blind spot: **parameter changes in different directions can affect the policy distribution very differently**, while the L2 norm treats all directions equally. To address this, we need to upgrade "uniform length" into "weighted length."

---

## Direction Sensitivity: From Ordinary Norms to Trust-Region Constraints

### The Limitation of Ordinary Distance

The L2 norm is defined as:

$$
\|\boldsymbol{x}\|_2^2 = \boldsymbol{x}^\top \boldsymbol{x}.
$$

For example, if $\boldsymbol{x} = [3, 4]^\top$, then $\boldsymbol{x}^\top\boldsymbol{x} = 9 + 16 = 25$. The L2 norm treats every direction equally: moving one step to the right and one step upward have the same "length."

In parameter space, however, different directions have different safety risks. Consider two parameter update directions:

**Direction A**: $\Delta\theta = [1, 0]^\top$

This direction mainly changes the first parameter. Suppose that parameter controls the logit for choosing action left. Changing it by $1$ may only shift the probability from $0.5$ to about $0.73$.

**Direction B**: $\Delta\theta = [0, 1]^\top$

This direction changes the second parameter. Suppose that parameter controls output scale. Changing it by $1$ may make all action probabilities change drastically.

Both directions have L2 norm $1$, but their effects on the policy distribution are completely different. A plain L2 constraint $\|\Delta\theta\|_2 \leq \delta$ cannot distinguish this difference.

### Weighted Norms: Direction-Aware Measurement

To distinguish risk across directions, introduce a matrix $F$ and define a **weighted length** in place of the ordinary L2 norm:

$$
\|\boldsymbol{x}\|_F^2 = \boldsymbol{x}^\top F\,\boldsymbol{x}.
$$

Consider a concrete example:

$$
F = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}, \qquad \boldsymbol{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}.
$$

Then:

$$
\boldsymbol{x}^\top F\,\boldsymbol{x}
= \begin{bmatrix} 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}
\begin{bmatrix} 1 \\ 1 \end{bmatrix}
= 1 \times 1 + 4 \times 1 = 5.
$$

The second direction is amplified by weight $4$, so the same step length costs more in that direction. The larger the diagonal element of $F$, the more tightly movement in that direction is constrained.

### Trust-Region Constraint

TRPO uses a trust-region constraint of the following form:

$$
(\theta - \theta_{old})^\top F(\theta - \theta_{old}) \leq \delta.
$$

Here $F$ is usually the **Fisher information matrix**. The intuition is: **a parameter update should not be judged only by Euclidean distance; it should also account for how much the step changes the policy distribution**. If a direction would sharply change the policy distribution, $F$ shrinks the allowed step in that direction. If a direction barely affects the policy distribution, $F$ allows a larger step.

The $(i,j)$ entry of the Fisher information matrix is:

$$
F_{ij} = \mathbb{E}_\pi\left[\frac{\partial \log \pi_\theta(a\mid s)}{\partial \theta_i}\frac{\partial \log \pi_\theta(a\mid s)}{\partial \theta_j}\right].
$$

The core role of $F$ is to map distances in parameter space to distances in policy-distribution space. The former is Euclidean geometry; the latter is information geometry.

### From TRPO to PPO: Exact Constraints and Approximate Constraints

TRPO directly solves an optimization problem with a quadratic constraint, which is computationally expensive because it requires computing the Fisher matrix and its inverse. PPO, the mainstream industrial RL algorithm, uses a simplified strategy:

- It does not explicitly compute $F$. Instead, it uses the probability ratio $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{old}(a_t\mid s_t)$ to indirectly measure policy change.
- It does not solve constrained optimization. Instead, it clips `clip(r_t, 1-\epsilon, 1+\epsilon)` to hard-limit how far the probability ratio can deviate.

The progression from ordinary norms to trust regions is:

$$
\|\Delta\theta\|_2^2 = \Delta\theta^\top \Delta\theta \quad\longrightarrow\quad \Delta\theta^\top F\,\Delta\theta \leq \delta \quad\longrightarrow\quad \text{clip}(r_t,\; 1-\epsilon,\; 1+\epsilon)
$$

Each step has the same goal: **limit the magnitude of policy change**. They differ in the tradeoff between precision and computational cost.

::: warning Common Pitfall
TRPO's trust-region constraint does not limit Euclidean distance in parameter space. It limits the effect of parameter changes on the policy distribution. The same $\|\Delta\theta\|_2 = 0.1$ can produce completely different policy changes in different directions.
:::

---

## A Three-Level View

Putting the three tools together, they answer the same core question from three levels: how do we keep training stable?

| Level        | Problem               | Tool                         | Formula                                         |
| ------------ | --------------------- | ---------------------------- | ----------------------------------------------- |
| Long term    | Iteration convergence | Eigenvalues, spectral radius | rho(gamma P) <= gamma < 1                       |
| Short term   | Single-step magnitude | L2 norm, gradient clipping   | $\lVert\boldsymbol{g}_{clipped}\rVert_2 \leq c$ |
| Fine-grained | Direction sensitivity | Weighted norm, trust region  | Delta theta^T F Delta theta <= delta            |

The progression is: eigenvalues guarantee convergence from the **long-term** perspective, norms limit single-step update size from the **short-term** perspective, and weighted norms distinguish risk across directions from the **fine-grained** perspective. The constraints become more precise, and the computational cost also rises.

---

## E.1 Module Overview: The Logical Chain Across Four Articles

This is the last main article in module E.1. The four articles revolve around one core question: **Can the Bellman equation $V(s) = R(s) + \gamma\sum P V(s')$ actually be solved?** Solving it faces three obstacles in sequence, and each obstacle introduces a new group of mathematical tools:

```text
Bellman equation in Chapter 3
  |
  v  Obstacle 1: too many equations to write one by one
E.1.1 + E.1.2  vectors, matrices, linear systems
  |  -> v = r + gamma P v  writes all equations at once
  |  -> v = (I-gamma P)^-1 r  solves in one step
  |  But when there are too many states, the matrix cannot be stored
  v  Obstacle 2: too many states, matrix cannot be stored
E.1.3  dot products, norms, function approximation
  |  -> v_hat(s) = w^T x(s)  approximate with a dot product
  |  -> ||g||_2 <= c  use a norm to limit updates
  |  But with approximation plus iteration, is training stable?
  v  Obstacle 3: training stability under approximation and iteration
E.1.4  eigenvalues, weighted norms, trust regions
     -> three levels: convergence guarantee + gradient clipping + trust region
```

Mapped to the core problems in RL:

| Question                                            | Mathematical tool                 | RL meaning                                 | Article |
| --------------------------------------------------- | --------------------------------- | ------------------------------------------ | ------- |
| How do we represent states and values?              | Vectors, matrices                 | Basis of tabular methods                   | E.1.1   |
| How do we compress Bellman equations?               | Linear systems                    | Theoretical basis of policy evaluation     | E.1.2   |
| How do we approximate values that cannot be tabled? | Dot products, norms               | Function approximation and neural networks | E.1.3   |
| Why can training be stable?                         | Eigenvalues, contraction mappings | Convergence guarantee                      | E.1.4   |
| How can policies be updated safely?                 | Weighted norms, trust regions     | Mathematical basis of TRPO/PPO             | E.1.4   |

> **Next**: [E.1.5 Formula Review and Exercises](./linear-algebra-formulas-exercises) revisits the RL concepts from Chapter 3 through the lens of linear algebra.
