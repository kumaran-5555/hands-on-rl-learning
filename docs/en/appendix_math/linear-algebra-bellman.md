---
title: E.1.2 Matrix Form of the Bellman Equation
---

# E.1.2 Matrix Form of the Bellman Equation

> **Prerequisites**: [E.1.1 Vectors and Matrices](./linear-algebra-basics), especially vectors, matrices, and matrix multiplication. It is also helpful to first read the Chapter 3 discussion of the [Bellman equation](../chapter03_mdp/value-bellman), so the single-state form is familiar.

---

## From One Equation to a Matrix Equation

Chapter 3 gave the Bellman equation for a single state:

$$
V^\pi(s) = \sum_{a} \pi(a|s)\left[R(s,a) + \gamma\sum_{s'} P(s'|s,a)V^\pi(s')\right].
$$

This formula works well for one state. But when an environment contains $n$ states, we need $n$ such equations, one for each state. Combining these $n$ equations gives a single matrix expression:

$$
\boxed{\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}}
$$

Here $\boldsymbol{v}$ is the value vector for all states, $\boldsymbol{r}$ is the immediate reward vector for all states, and $P$ is the state transition matrix. Moving terms also gives a closed-form solution:

$$
\boxed{\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}}
$$

We will derive these formulas from the single-state equation in three steps: first place all values into a vector, then place transitions into a matrix, and finally assemble and solve the system of equations.

---

## Step 1: Put All State Values into a Vector

Use the two-state example from the appendix introduction:

- In $s_1$, the reward is $2$, and the next state is always $s_2$.
- In $s_2$, the reward is $1$, and the next state is always $s_1$.
- The discount factor is $\gamma = 0.5$.

The two state values are $v_1$ and $v_2$. Put all state values into one column vector:

$$
\boldsymbol{v} =
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix},
\qquad
\boldsymbol{r} =
\begin{bmatrix}
2 \\
1
\end{bmatrix}.
$$

$\boldsymbol{v}$ is the unknown quantity, and $\boldsymbol{r}$ is the known immediate reward. With two states, the vector has two components. With $n$ states, it has $n$ components, but the notation stays the same.

---

## Step 2: Put the Transition Relationship into a Matrix

Starting from $s_1$ always leads to $s_2$, and starting from $s_2$ always leads to $s_1$. The transition matrix is:

$$
P =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}.
$$

Rows correspond to "which state we start from," and columns correspond to "which state we go to next." The meaning of matrix-vector multiplication is that row $i$ of $P\boldsymbol{v}$ gives the expected next-state value when starting from state $s_i$. Check:

$$
P\boldsymbol{v} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
v_2 \\
v_1
\end{bmatrix}.
$$

This says exactly that from $s_1$ the next state is $s_2$, so the future value is $v_2$; from $s_2$ the next state is $s_1$, so the future value is $v_1$. Matrix-vector multiplication has automatically performed the probability-weighted sum.

---

## Step 3: Assemble the System

We now have three objects: value vector $\boldsymbol{v}$, reward vector $\boldsymbol{r}$, and transition matrix $P$. Writing the Bellman equation state by state gives:

$$
\begin{aligned}
v_1 &= 2 + 0.5v_2, \\
v_2 &= 1 + 0.5v_1.
\end{aligned}
$$

In matrix language, the immediate reward is $\boldsymbol{r}$, and the discounted future value is $\gamma P\boldsymbol{v}$. Adding them gives:

$$
\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}.
$$

Verify the first row on the right:

$$
2 + 0.5 \times (0 \cdot v_1 + 1 \cdot v_2) = 2 + 0.5v_2.
$$

The second row:

$$
1 + 0.5 \times (1 \cdot v_1 + 0 \cdot v_2) = 1 + 0.5v_1.
$$

This is exactly the same as the two equations written separately. The matrix form introduces no new content; it only compresses many equations with the same structure into one equation.

### Why the Compression Works

The key is the term $P\boldsymbol{v}$. Each row of $P$ is exactly a set of transition probabilities whose sum is $1$, and matrix multiplication is exactly weighted averaging: probability times value. The Bellman equation says "value = immediate reward + discounted future value," and the matrix equation states the same relationship for **all states at once**.

| Symbol        | Meaning                                      | Dimension |
| ------------- | -------------------------------------------- | --------- |
| **v**         | Values of all states, the unknown            | n x 1     |
| **r**         | Immediate rewards of all states              | n x 1     |
| gamma P **v** | Discounted probability-weighted future value | n x 1     |

All three quantities have dimension $n \times 1$, so both sides of the equation have the same shape. This is the operation behind DP in Chapter 3: repeatedly apply $v_{k+1} = r + \gamma P v_k$ until convergence.

---

## Closed-Form Solution and Fixed Point

$\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ is a linear system, so it can be solved directly. Move the terms containing $\boldsymbol{v}$ to the left:

$$
\boldsymbol{v} - \gamma P\boldsymbol{v} = \boldsymbol{r}.
$$

Factor out the common term. Here $I$ is the identity matrix, with $1$ on the diagonal and $0$ elsewhere:

$$
(I - \gamma P)\boldsymbol{v} = \boldsymbol{r}.
$$

If $I - \gamma P$ is invertible, the closed-form solution is:

$$
\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}.
$$

Substitute the two-state numbers:

$$
I - \gamma P =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
- 0.5
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}.
$$

Solving the system:

$$
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
2 \\
1
\end{bmatrix}
\quad\Longrightarrow\quad
v_1 = 3.33,\quad v_2 = 2.67.
$$

### Geometric Intuition: The Intersection Is the Fixed Point

This system corresponds to two lines in the $(v_1, v_2)$ plane:

- The first equation is $v_1 - 0.5v_2 = 2$.
- The second equation is $-0.5v_1 + v_2 = 1$.

Their intersection is $(3.33, 2.67)$. This point is also the **fixed point** of the Bellman update: substituting $(3.33, 2.67)$ into $v_{new} = r + 0.5Pv$ returns the same $(3.33, 2.67)$. The value no longer changes, which means this is the true value.

---

## From Two States to $n$ States

The closed-form solution $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$ was derived for two states. If we generalize from 2 states to any $n$, the equation keeps the same form:

$$
\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma P_\pi \boldsymbol{v}_\pi.
$$

Here:

- $\boldsymbol{v}_\pi \in \mathbb{R}^n$: values of all states under policy $\pi$
- $\boldsymbol{r}_\pi \in\mathbb{R}^n$: expected immediate reward for each state
- $P_\pi \in\mathbb{R}^{n\times n}$: the transition matrix induced by the policy, with $P_\pi[i,j] = \sum_a \pi(a\mid s_i) p(s_j\mid s_i, a)$

With three states, $P$ is $3\times3$, and $\boldsymbol{v}$ and $\boldsymbol{r}$ are $3\times1$. The equation $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ still holds. This is the central advantage of matrix notation: **no matter how many states there are, the equation keeps the same form**.

### When Is $I - \gamma P$ Invertible?

The key requirement for solving is that $I - \gamma P$ must be invertible. Intuitively, this means the Bellman update must not diverge. When $0 < \gamma < 1$ and $P$ is a valid transition matrix whose rows sum to $1$, $I - \gamma P$ is essentially always invertible.

More precisely, the spectral radius of $\gamma P$, meaning the largest absolute eigenvalue, satisfies $\rho(\gamma P) \leq \gamma < 1$. Therefore all eigenvalues of $I - \gamma P$ stay away from $0$, so the matrix is invertible. E.1.4 explains this in detail.

---

## Why We Do Not Directly Invert the Matrix

The derivation makes the closed-form solution look simple: write the matrix equation, invert the matrix, and obtain the answer. In practice, this path is usually infeasible for three reasons:

1. **The scale is too large.** If the number of states is $n=10^6$, then $I-\gamma P$ is a $10^6 \times 10^6$ matrix. Matrix inversion costs $O(n^3)$ and is essentially impossible.
2. **The matrix may not be explicitly available.** In many practical problems, the entries of $P$ are unknown; we only observe sampled transitions. The MC and TD methods in Chapter 3 operate exactly under this condition.
3. **The state may be continuous.** If the state is an image or text, there may be no finite matrix at all, so $P$ cannot be constructed.

Practical algorithms approximate the solution with iterative methods:

- **Value iteration**: repeatedly execute $v_{k+1} = r + \gamma P v_k$ until convergence.
- **Policy evaluation**: repeatedly apply Bellman updates inside policy iteration.
- **TD learning**: use sampled data for incremental updates.

All of these methods approximate the solution $(I-\gamma P)^{-1}\boldsymbol{r}$ in a more scalable way, without actually computing the inverse. The evolution from DP to MC to TD in Chapter 3 corresponds to this path: direct iteration with a known model, sampling without a known model, and one-step sampling updates.

::: warning Common Pitfall
When you see $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$, do not assume a practical algorithm is really computing a matrix inverse. This formula is a theoretical closed-form solution used to show existence and uniqueness. Practical algorithms are iterative.
:::

---

## Matrix Form of the Q Function

So far we have handled only the state value $V$. The action value $Q(s,a)$ can also be written in matrix form, and it has a clear algebraic relationship with the matrix form of $V$.

### Notation

Put the Q values of all $(s,a)$ pairs into one long vector:

$$
\boldsymbol{q} =
\begin{bmatrix}
Q(s_1, a_1) \\
Q(s_1, a_2) \\
\vdots \\
Q(s_2, a_1) \\
\vdots
\end{bmatrix}
\in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}.
$$

Similarly, $\boldsymbol{r} \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ stores the immediate reward of each $(s,a)$ pair.

The transition matrix becomes $P \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}| \times |\mathcal{S}|}$. Each row corresponds to one $(s,a)$ pair, and each column corresponds to a next state $s'$:

$$
P[(s,a),\, s'] = P(s' \mid s, a).
$$

The policy matrix $\Pi_\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}||\mathcal{A}|}$ compresses the Q vector back into a V vector:

$$
\Pi_\pi[\,s,\, (s,a)\,] = \pi(a \mid s).
$$

### V-Q Relationship

$$
\boldsymbol{v}_\pi = \Pi_\pi \boldsymbol{q}_\pi
$$

Check row $i$: $\sum_a \pi(a|s_i) Q(s_i, a) = V(s_i)$. This is the matrix form of $V(s) = \sum_a \pi(a|s) Q(s,a)$.

### Bellman Expectation Equation for Q

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')
$$

In matrix form:

$$
\boldsymbol{q}_\pi = \boldsymbol{r} + \gamma P \boldsymbol{v}_\pi.
$$

Substitute $\boldsymbol{v}_\pi = \Pi_\pi \boldsymbol{q}_\pi$ to obtain the pure-Q recursion:

$$
\boldsymbol{q}_\pi = \boldsymbol{r} + \gamma P \Pi_\pi \boldsymbol{q}_\pi.
$$

The closed-form solution is $\boldsymbol{q}_\pi = (I - \gamma P \Pi_\pi)^{-1} \boldsymbol{r}$.

### Bellman Optimality Equation for Q

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')
$$

Matrix form:

$$
\boldsymbol{q}_* = \boldsymbol{r} + \gamma P \cdot \mathrm{rowmax}(\boldsymbol{q}_*)
$$

Here $\mathrm{rowmax}(\boldsymbol{q}) \in \mathbb{R}^{|\mathcal{S}|}$ extracts, for each state, the maximum Q value among all actions of that state. Because max is not a linear operation, the optimality equation has no closed-form linear solution and must be approximated by iteration, such as Q-Learning.

### Deriving the Matrix Form of V from Q

Substitute $\boldsymbol{v}_\pi = \Pi_\pi \boldsymbol{q}_\pi$ into $\boldsymbol{q}_\pi = \boldsymbol{r} + \gamma P \boldsymbol{v}_\pi$, then left-multiply both sides by $\Pi_\pi$:

$$
\Pi_\pi \boldsymbol{q}_\pi = \Pi_\pi \boldsymbol{r} + \gamma \Pi_\pi P \boldsymbol{v}_\pi
\quad\Longrightarrow\quad
\boldsymbol{v}_\pi = \underbrace{\Pi_\pi \boldsymbol{r}}_{\boldsymbol{r}_\pi} + \gamma \underbrace{\Pi_\pi P}_{P_\pi} \boldsymbol{v}_\pi.
$$

This is the same $\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma P_\pi \boldsymbol{v}_\pi$ derived earlier. In the matrix form of V, $\boldsymbol{r}_\pi$ and $P_\pi$ already average over the policy. In the matrix form of Q, the action dimension is preserved, and policy averaging is handled separately by $\Pi_\pi$. This is the matrix-language expression of the idea that $Q$ carries finer-grained information than $V$.

---

## Matrix Form of DP Iteration

Chapter 3 introduced the state-by-state update for DP policy evaluation:

$$
V(s) \leftarrow \sum_a \pi(a|s)\left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right].
$$

The matrix form splits the Bellman expectation equation into an iteration:

$$
\boldsymbol{v}_{k+1} = \boldsymbol{r}_\pi + \gamma P_\pi \boldsymbol{v}_k.
$$

Each round performs one Bellman update for all states simultaneously. In the two-state example above, starting from $\boldsymbol{v}_0 = \boldsymbol{0}$ and iterating repeatedly converges to the closed-form solution $\boldsymbol{v} = (I - \gamma P_\pi)^{-1}\boldsymbol{r}_\pi$.

Policy improvement, $\pi'(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a)V^\pi(s')]$, becomes the following matrix operation: for each state $s$, compare the rows of $\boldsymbol{r} + \gamma P\boldsymbol{v}_\pi$ that belong to that state, where each row corresponds to one action, and choose the action with the largest value.

---

## Comparison Table

| Concept                      | State-by-state form, Chapter 3                                                         | Matrix form                                                                           |
| ---------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Bellman expectation equation | $V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]$ | $\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma P_\pi \boldsymbol{v}_\pi$           |
| Bellman optimality equation  | $V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right]$                 | $\boldsymbol{v}_* = \boldsymbol{r}_* + \gamma P_* \boldsymbol{v}_*$, with rowwise max |
| Closed-form solution         | -                                                                                      | $\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}$                                  |
| V-Q relationship             | $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$                                                | $\boldsymbol{v}_\pi = \Pi_\pi \boldsymbol{q}_\pi$                                     |
| Q Bellman expectation        | $Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')$    | $\boldsymbol{q}_\pi = \boldsymbol{r} + \gamma P \Pi_\pi \boldsymbol{q}_\pi$           |
| Q Bellman optimality         | $Q^*(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a')$                      | $\boldsymbol{q}_* = \boldsymbol{r} + \gamma P \cdot\mathrm{rowmax}(\boldsymbol{q}_*)$ |
| DP policy evaluation         | $V(s) \leftarrow \sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')]$         | $\boldsymbol{v}_{k+1} = \boldsymbol{r}_\pi + \gamma P_\pi \boldsymbol{v}_k$           |

MC and TD methods update individual states from samples, so they do not have a corresponding full-matrix form in the same sense.

---

## Limitations of the Matrix Form

This article compressed the Bellman equation into the matrix form $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ and gave the closed-form solution $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$. No matter how many states there are, the equation is one line.

But the matrix form assumes one thing: **each state has an independent value $v(s)$ that can be stored**.

| Environment                                  | State-space size | Can store? |
| -------------------------------------------- | ---------------- | ---------- |
| Two-state toy environment                    | 2                | Yes        |
| GridWorld 10 x 10                            | 100              | Yes        |
| Go board                                     | ~10^170          | No         |
| Continuous state, such as robot joint angles | Infinite         | No         |

Once the number of states becomes large, not only is matrix inversion infeasible, but even the value table itself exceeds storage capacity. The closed-form solution is compact, but for Go's $10^{170}$ states, the required storage is unrealistic.

The solution is to avoid storing one value for every state. Instead, use a function to **approximate** value: extract features from the state, then compute value from the dot product of features and weights. The next article develops this idea.

> **Next**: [E.1.3 Dot Products, Norms, and Function Approximation](./linear-algebra-function-approx) explains how feature vectors and dot products approximate value when there are too many states to store.
