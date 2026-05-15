---
title: E.1.3 Dot Products, Norms, and Function Approximation
---

# E.1.3 Dot Products, Norms, and Function Approximation

> **Prerequisites**: [E.1.1 Vectors and Matrices](./linear-algebra-basics), especially vectors and vector operations. [E.1.2 Bellman Matrix Form](./linear-algebra-bellman), especially the matrix form $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$.

---

## Overview

The previous article derived the matrix form of the Bellman equation, $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$, and the closed-form solution $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$. These expressions are mathematically compact and rigorous, but they leave a practical difficulty: **when the state space is enormous, the value table cannot be fully stored**. Go has roughly $10^{170}$ states; the matrix $P$ cannot be explicitly constructed, and the value vector $\boldsymbol{v}$ cannot be written component by component.

The solution is to approximate value with a function: extract features from a state, then compute the value estimate as the dot product of features and weights. The core formula in this article is:

$$
\boxed{\hat{v}(s) = \boldsymbol{w}^\top \boldsymbol{x}(s)}
$$

Here $\boldsymbol{x}(s)$ is the feature vector of state $s$, and $\boldsymbol{w}$ is the weight vector to be learned. The result of the dot product is the approximate value of that state.

The first half of this article uses dot products to build the function-approximation framework. Once function approximation is introduced, training must repeatedly update the weights $\boldsymbol{w}$. How should the update step size be chosen? How do we measure the magnitude of a change? These questions naturally lead to **norms**, the main tool in the second half of the article. We begin with the limitation of tabular methods.

---

## The Limitation of Tabular Methods

Start with a concrete comparison:

| Environment                            | State-space size | Can store? |
| -------------------------------------- | ---------------- | ---------- |
| Two-state toy environment              | 2                | Yes        |
| GridWorld 10x10                        | 100              | Yes        |
| Go board                               | ~10^170          | No         |
| Continuous state, such as robot joints | Infinite         | No         |

The DP, MC, and TD methods introduced in Chapter 3 all share one premise, whether they solve the Bellman equation directly, estimate returns from complete trajectories, or update immediately at each step: **store one $v(s)$ for every state**. Q-Learning in Chapter 4 follows the same idea, but stores $Q(s,a)$ for every state-action pair instead.

Once the number of states grows beyond a certain scale, storing one independent number for every state is no longer feasible. The solution path is: **stop storing each state's value independently, and instead use one function to approximate them all**. The input to the function is a state's features, and the output is the estimated value of that state.

---

## Dot Product: Combining Features and Weights into a Prediction

The simplest function-approximation scheme extracts a set of features from the state and computes value through the **dot product** of features and weights.

Suppose a state has three features:

$$
\boldsymbol{x}(s) =
\begin{bmatrix}
1 \\
0.5 \\
2
\end{bmatrix}.
$$

These features might represent "distance to the goal is $1$," "velocity is $0.5$," and "number of obstacles is $2$." A linear value function uses three weights:

$$
\boldsymbol{w} =
\begin{bmatrix}
0.2 \\
1.0 \\
-0.1
\end{bmatrix}.
$$

The estimated state value is given by the dot product:

$$
\hat{v}(s) = \boldsymbol{w}^\top \boldsymbol{x}(s)
= 0.2 \times 1 + 1.0 \times 0.5 + (-0.1) \times 2 = 0.5.
$$

The essence of the dot product is this: **each feature's contribution to the final prediction is determined by its corresponding weight, and all contributions are summed to obtain the prediction**. $\boldsymbol{w}$ specifies the relative importance of each feature, $\boldsymbol{x}(s)$ gives the current state's feature values, and the dot product combines them into one scalar prediction.

The $Q(s,a)$ discussed in Chapter 3 can be approximated in the same way:

$$
Q(s,a) = \boldsymbol{w}^\top\phi(s,a),
$$

where $\phi(s,a)$ is the feature vector of the state-action pair. DQN in Chapter 4 generalizes linear approximation to a neural network, but whether the model is linear or deep, the underlying operations are vector operations.

### How Are Features Constructed?

Feature construction depends on the problem. Here are several common approaches.

**One-hot encoding** for discrete states:

If the state set is $\{s_1, s_2, s_3\}$, then:

| State | Feature vector |
| ----- | -------------- |
| s1    | [1, 0, 0]^T    |
| s2    | [0, 1, 0]^T    |
| s3    | [0, 0, 1]^T    |

With one-hot features, $\boldsymbol{w}^\top \boldsymbol{x}(s)$ simply extracts the $i$-th component of $\boldsymbol{w}$, exactly like a table lookup. In other words, **tabular methods are a special case of linear approximation**. The tabular Q-Learning method in Chapter 3 is essentially linear approximation with one-hot features.

**Hand-crafted features** in GridWorld:

In GridWorld, the feature vector of state $(r, c)$ can be designed as:

$$
\boldsymbol{x}(s) = [r/c_{max},\; c/r_{max},\; \text{dist\_to\_goal}]^\top.
$$

These features include normalized position information and distance to the goal.

**Features learned by a neural network**:

In deep RL, features are not designed manually. They are learned automatically by a neural network. An input image passes through several convolutional layers to produce a vector $\boldsymbol{x}(s)$, and a linear layer then computes $\boldsymbol{w}^\top \boldsymbol{x}(s)$. DQN in Chapter 4 follows this idea.

The path from tables to deep networks is a gradual increase in feature automation:

$$
\text{tabular, one-hot} \;\to\; \text{linear approximation, hand-crafted features} \;\to\; \text{deep networks, learned features}
$$

No matter which method is used, vectors are the core objects underneath. Each layer of a neural network performs matrix multiplication, a batched extension of dot products, with nonlinear activations such as ReLU inserted between layers so the overall model can fit complex functions.

### Geometric Meaning of the Dot Product

The dot product is not only "multiply elements and sum them." Geometrically:

$$
\boldsymbol{w}^\top \boldsymbol{x} = \|\boldsymbol{w}\| \|\boldsymbol{x}\| \cos\theta.
$$

Here $\theta$ is the angle between the two vectors. Therefore:

- If $\boldsymbol{w}$ and $\boldsymbol{x}$ point in similar directions, so $\cos\theta > 0$, the dot product is positive.
- If they point in opposite directions, so $\cos\theta < 0$, the dot product is negative.
- If they are orthogonal, so $\cos\theta = 0$, the dot product is zero.

In RL value estimation, a positive dot product means the state is evaluated positively under the current weights, while a negative dot product means it is evaluated negatively. The direction of $\boldsymbol{w}$ encodes the relative importance of features, and the direction of $\boldsymbol{x}(s)$ encodes the state's feature profile. The smaller the angle between them, the higher the estimate.

---

## From Prediction to Learning: Where Do the Weights Come From?

So far, we know that the dot product $\hat{v}(s) = \boldsymbol{w}^\top \boldsymbol{x}(s)$ can approximate value. But where does $\boldsymbol{w}$ come from?

The answer is: it is learned through training. In Chapter 3, TD methods update value estimates using TD Error:

$$
V(s) \leftarrow V(s) + \alpha\left[r + \gamma V(s') - V(s)\right].
$$

Training under function approximation follows the same logic: compute the gap between prediction and target, called the loss, and then update weights along the gradient direction:

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \boldsymbol{g},
$$

where $\boldsymbol{g}$ is the gradient vector and $\alpha$ is the learning rate.

This formula raises a new question: **as a vector, how large is the gradient $\boldsymbol{g}$?** If the gradient is too large, one parameter update may move too far and destabilize training. If it is too small, learning becomes slow. To answer "how large is a vector" quantitatively, we need a **norm**, the mathematical tool for measuring vector length. The target formula in this section is:

$$
\boxed{\|\boldsymbol{g}\|_2 = \sqrt{\sum_i g_i^2} \quad\longrightarrow\quad \text{if }\|\boldsymbol{g}\|_2 > c\text{, scale it down to }c}
$$

Norms measure "how large," and scaling controls "how far each step can move." We begin with the most common norm.

---

## Norms: Measuring the Size of a Vector

The most common norm is the **L2 norm**, defined as the Pythagorean theorem generalized to any number of dimensions:

$$
\|\boldsymbol{x}\|_2 = \sqrt{\sum_i x_i^2}.
$$

For vector $[3, 4]^\top$, the L2 norm is:

$$
\|[3, 4]^\top\|_2 = \sqrt{3^2 + 4^2} = 5.
$$

This is exactly the hypotenuse length of a right triangle. In two dimensions, the L2 norm reduces to Euclidean distance.

### L1 Norm

For the same vector $[3, 4]^\top$, the **L1 norm** is:

$$
\|[3, 4]^\top\|_1 = |3| + |4| = 7.
$$

The L1 norm is the sum of the absolute values of all components. Compared with L2, L1 penalizes large components less heavily because it does not square them, and it is more sensitive to small components.

### L1 and L2: How to Choose

| Property                 | L2 norm                                     | L1 norm                                  |
| ------------------------ | ------------------------------------------- | ---------------------------------------- |
| Formula                  | $\sqrt{\sum_i x_i^2}$                       | $\sum_i \lvert x_i\rvert$                |
| Response to large values | Sensitive, squares amplify large components | Relatively less sensitive                |
| In RL                    | Gradient clipping, weight decay             | Sparse regularization, feature selection |
| Typical use              | `max_grad_norm`                             | L1 regularization                        |

### Frobenius Norm, a Matrix Norm

The idea of a vector norm can be generalized to matrices. The **Frobenius norm** is the most common matrix norm:

$$
\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}.
$$

For example:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad \|A\|_F = \sqrt{1+4+9+16} = \sqrt{30} \approx 5.48.
$$

In RL, the Frobenius norm is often used for regularizing weight matrices and measuring differences between two matrices.

---

## Gradient Clipping: A Direct Use of Norms in Training

Once we have norms, we can quantify "the size of a gradient." When training neural networks, if the gradient norm is too large, the parameter update can be extremely aggressive and destabilize training. **Gradient clipping** limits this length.

### Numerical Example

Suppose after one backward pass, the gradients of four parameters are:

$$
\boldsymbol{g} =
\begin{bmatrix}
12 \\
5 \\
-8 \\
3
\end{bmatrix}.
$$

Its L2 norm is:

$$
\|\boldsymbol{g}\|_2 = \sqrt{144 + 25 + 64 + 9} = \sqrt{242} \approx 15.56.
$$

If the maximum norm is set to $5$, scale the gradient by $5/15.56 \approx 0.321$:

$$
\boldsymbol{g}_{clipped} = 0.321
\begin{bmatrix}
12 \\
5 \\
-8 \\
3
\end{bmatrix}
\approx
\begin{bmatrix}
3.85 \\
1.61 \\
-2.57 \\
0.96
\end{bmatrix}.
$$

The clipped gradient norm is exactly $5$. Notice that clipping **changes only the vector length, not its direction**; every component is scaled by the same ratio.

This is the mathematical meaning of the `max_grad_norm` parameter in RL code. In PyTorch:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### Why Gradient Clipping Is Needed in RL

RL gradients come from policy gradients and temporal-difference estimates, and their noise level is naturally higher than in supervised learning. The reason is that data comes from a continually changing sampling policy rather than a fixed training set. A single high-return trajectory may produce a very large gradient, causing a drastic parameter change and making the next policy very different from the previous one. The REINFORCE algorithm in Chapter 5 faces exactly this issue: its gradient estimates have very high variance. Gradient clipping places a hard upper bound on this fluctuation and prevents a single update from moving too far.

---

## Summary: From Tables to Function Approximation

The core transformation in this article is: **when the state space is too large to store item by item, use features and dot products to approximate value**.

| Scenario                       | Method                                     | Essence                                |
| ------------------------------ | ------------------------------------------ | -------------------------------------- |
| Few states, such as n < 1000   | Table: store v(s) for each state           | one-hot plus dot product               |
| Many states, known features    | Linear approximation: v(s) approx w^T x(s) | hand-crafted features plus dot product |
| Complex states, images or text | Deep network: v(s) approx f_theta(s)       | learned features plus nonlinearity     |

No matter which method is used, the underlying operations are based on vectors and dot products. Each neural-network layer performs matrix multiplication, and norms provide a unified tool for measuring size. They are used in both gradient clipping and regularization.

---

## After Function Approximation: Two Unanswered Questions

This article solved the problem "how do we store value when the state space is too large": approximate value with dot products, and quantify update magnitude with norms.

But after introducing function approximation, two questions remain:

1. Whether we use tables or function approximation, if we repeatedly apply the Bellman update $v_{k+1} = r + \gamma Pv_k$, will it converge stably? In Chapter 3, DP methods "iterate until convergence." Is this convergence theoretically guaranteed, or can the process oscillate or diverge?
2. In the policy-gradient update from the second route in Chapter 3, $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$, can a single update be so large that the policy becomes worse? Gradient clipping limits step size, but it treats all directions equally. Moving $0.1$ in one direction may barely matter, while moving $0.1$ in another direction may drastically change the policy.

The common essence of these two questions is that we must analyze how "amount of change" behaves differently in different directions. This leads to eigenvalues, spectral radius, and weighted norms, the topic of the next article.

> **Next**: [E.1.4 Convergence, Eigenvalues, and Trust Regions](./linear-algebra-advanced) explains why value iteration is stable and how parameter updates can account for direction sensitivity.
