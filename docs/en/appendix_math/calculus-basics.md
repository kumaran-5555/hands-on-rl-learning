---
title: E.3.1 Derivatives and Gradients
---

# E.3.1 Calculus Basics: Derivatives, Gradients, and the Chain Rule

> **Prerequisite**: This page does not require prior calculus knowledge, but you should first read the "two-state running example" in the [appendix introduction](./intro) and [E.1.1 Vectors and Matrices](./linear-algebra-basics).

---

## Functions, Objectives, and Rates of Change

The central task in reinforcement learning is optimization: adjusting policy parameters so that the average return becomes higher and higher. The vectors, matrices, and Bellman equations introduced in earlier appendix pages tell us "how good the current policy is", but they do not tell us "how to make it better". To answer that question, we need to know how much the objective function changes when the parameters change a little. This is exactly the problem calculus solves.

Imagine turning the knob on a radio to find a station. At different knob positions, the received signal has different clarity. Calculus studies a similar question: when one quantity changes, how does another quantity change with it? In reinforcement learning, the knob is the policy parameter $\theta$, and the signal clarity is the objective function $J(\theta)$.

An objective function can be understood as an input-output relationship:

$$
\theta \longmapsto J(\theta).
$$

If $J(\theta)$ represents average return, then training means searching for parameter values that make $J(\theta)$ larger. Derivatives and gradients answer a very plain question: **standing here right now, which direction makes the objective rise fastest?**

Once this point is clear, calculus no longer looks like a collection of isolated formulas. It becomes the basic language that optimization algorithms rely on.

---

## Derivatives Through a One-Dimensional Function

The phrase "which direction to move" needs a more precise tool. That tool is the derivative. We first build intuition with a simple function that has only one parameter.

$$
J(\theta)=-(\theta-0.8)^2+1.
$$

You can interpret $\theta$ as a highly simplified policy parameter. For example, $\theta=0.2$ means the probability of choosing right is small, while $\theta=0.8$ means the probability of choosing right is larger.

This function reaches its maximum at $\theta=0.8$. Without drawing the graph, look at a few values:

| $\theta$ | $J(\theta)$ |
| -------- | ----------- |
| $0.2$    | $0.64$      |
| $0.5$    | $0.91$      |
| $0.8$    | $1.00$      |
| $1.0$    | $0.96$      |

From $0.2$ to $0.5$, the objective increases. From $0.8$ to $1.0$, the objective decreases. A derivative describes this phenomenon: near the current position, how fast and in which direction the function value changes as the parameter changes.

Taking the derivative of this function gives $J'(\theta)$, read as "J prime of theta". The prime mark denotes a derivative:

$$
J'(\theta)=-2(\theta-0.8).
$$

At $\theta=0.2$:

$$
J'(0.2)=1.2.
$$

The derivative is positive, which means moving to the right increases the objective. At $\theta=1.0$:

$$
J'(1.0)=-0.4.
$$

The derivative is negative, which means moving to the left increases the objective.

---

## Gradient Ascent and Gradient Descent

Once we know the direction, the natural next step is to move along it. If the goal is to maximize return, we use gradient ascent:

$$
\theta \leftarrow \theta + \alpha J'(\theta).
$$

Start from $\theta=0.2$ and choose learning rate $\alpha=0.1$:

$$
\theta \leftarrow 0.2 + 0.1\times1.2 = 0.32.
$$

Compute the derivative again:

$$
J'(0.32)=-2(0.32-0.8)=0.96.
$$

Continue the update:

$$
\theta \leftarrow 0.32 + 0.1\times0.96 = 0.416.
$$

The parameter moves step by step toward $0.8$. Each step moves a small distance in the direction indicated by the derivative. This is the core logic of gradient ascent.

Conversely, if the goal is to minimize a loss function $L(\theta)$, we change the plus sign to a minus sign. This is gradient descent:

$$
\theta \leftarrow \theta - \alpha L'(\theta).
$$

The phrase "backpropagation plus optimizer" in deep learning is essentially this loop repeated over and over: compute gradients, then update parameters.

---

## From Derivatives to Gradients: What If There Is More Than One Parameter?

One-dimensional derivatives are easy to understand, but real models often have hundreds of thousands or even hundreds of millions of parameters. Fortunately, the idea generalizes directly. Suppose the objective function has two parameters:

$$
J(\theta_1,\theta_2)=-(\theta_1-1)^2-(\theta_2-2)^2+5.
$$

It is maximized at $(1,2)$. If we take the derivative with respect to each parameter and arrange the results into a vector, we get the gradient. The symbol $\nabla$, read as "nabla", means "collect all partial derivatives together":

$$
\nabla J(\theta_1,\theta_2)=
\begin{bmatrix}
-2(\theta_1-1) \\
-2(\theta_2-2)
\end{bmatrix}.
$$

If the current parameters are $(0,0)$, the gradient is

$$
\nabla J(0,0)=
\begin{bmatrix}
2 \\
4
\end{bmatrix}.
$$

This means that $\theta_1$ should increase and $\theta_2$ should also increase, with $\theta_2$ increasing more strongly. The gradient is a kind of steering wheel: it tells every parameter where to move at the same time. With learning rate $0.1$, one gradient-ascent update gives

$$
\begin{bmatrix}
\theta_1 \\
\theta_2
\end{bmatrix}
\leftarrow
\begin{bmatrix}
0 \\
0
\end{bmatrix}
+0.1
\begin{bmatrix}
2 \\
4
\end{bmatrix}
=
\begin{bmatrix}
0.2 \\
0.4
\end{bmatrix}.
$$

The concept of a gradient is not limited to two parameters. A neural network may have thousands or millions of parameters; its gradient is a vector of the same dimension, telling each parameter how it should move.

---

## The Chain Rule: How Signals Pass Between Layers

So far, we have discussed "taking derivatives directly with respect to parameters". But neural networks are compositions of many functions. The input is transformed by the first layer, the result is passed to the second layer, and so on until the loss is produced. The chain rule is the tool for handling this composite structure. It tells us how changes in an outer function pass step by step through intermediate variables to inner parameters.

Consider a concrete example:

$$
y = 3\theta, \qquad L=(y-6)^2.
$$

If $\theta=1$, then $y=3$, and the loss is

$$
L=(3-6)^2=9.
$$

We want to know how $L$ changes when $\theta$ changes a little. Of course, we can substitute $y=3\theta$ into $L$ directly and then differentiate:

$$
L=(3\theta-6)^2.
$$

Differentiating gives

$$
\frac{dL}{d\theta}=2(3\theta-6)\times3.
$$

At $\theta=1$:

$$
\frac{dL}{d\theta}=2(3-6)\times3=-18.
$$

Here $2(3\theta-6)$ is the derivative of the loss with respect to $y$, written $\frac{dL}{dy}$, and $3$ is the derivative of $y$ with respect to $\theta$, written $\frac{dy}{d\theta}$. The chain rule multiplies them:

$$
\frac{dL}{d\theta}=\frac{dL}{dy}\cdot\frac{dy}{d\theta}.
$$

In words: "$\theta$ affects $y$, and $y$ affects $L$, so the effect of $\theta$ on $L$ is the product of these two effects." Backpropagation is the efficient large-scale version of this process in neural networks. It starts from the loss and moves backward along the computation graph, applying the chain rule layer by layer.

---

## Partial Derivatives: Move One Knob and Hold the Others Fixed

So far, the derivative notation we used was $d$, which corresponds to functions with one variable. When a function has multiple variables, we need a new symbol, $\partial$, read as "partial". It means "differentiate with respect to one variable while treating the other variables as constants". This is a partial derivative. Use the same two-parameter objective function:

$$
J(\theta_1,\theta_2)=-(\theta_1-1)^2-(\theta_2-2)^2+5.
$$

It has two parameters. We can ask two separate questions: if $\theta_2$ is held fixed, how does $\theta_1$ affect the objective? Conversely, if $\theta_1$ is held fixed, how does $\theta_2$ affect it? This is what partial derivatives do.

The partial derivative with respect to $\theta_1$ is

$$
\frac{\partial J}{\partial \theta_1}=-2(\theta_1-1).
$$

The partial derivative with respect to $\theta_2$ is

$$
\frac{\partial J}{\partial \theta_2}=-2(\theta_2-2).
$$

Arranging all partial derivatives into a vector gives the gradient:

$$
\nabla_\theta J=
\begin{bmatrix}
\frac{\partial J}{\partial \theta_1} \\
\frac{\partial J}{\partial \theta_2}
\end{bmatrix}.
$$

At $(\theta_1,
\theta_2)=(0,0)$:

$$
\nabla_\theta J(0,0)=
\begin{bmatrix}
2 \\
4
\end{bmatrix}.
$$

This means both parameters should increase, and the ascent direction is stronger for the second parameter. Gradients in neural networks work the same way; only the number of parameters changes from $2$ to millions or more.

---

## Learning Rate: How Far Each Step Goes

The gradient tells us the direction, but the distance to move is a separate issue. The learning rate $\alpha$ is the knob that controls step size.

If the current parameter is

$$
\theta=
\begin{bmatrix}
0 \\
0
\end{bmatrix},
\qquad
\nabla J=
\begin{bmatrix}
2 \\
4
\end{bmatrix},
$$

and the learning rate is $\alpha=0.1$, one gradient-ascent update is

$$
\theta\leftarrow
\begin{bmatrix}
0 \\
0
\end{bmatrix}
+0.1
\begin{bmatrix}
2 \\
4
\end{bmatrix}
=
\begin{bmatrix}
0.2 \\
0.4
\end{bmatrix}.
$$

If the learning rate is too small, training is very slow because each step moves only a little. If the learning rate is too large, the parameters may jump over the optimum or even diverge because the step passes straight over the summit. Gradients in reinforcement learning already contain considerable noise, so the choice of learning rate is especially sensitive. Techniques such as gradient clipping, the Adam optimizer, and PPO clipping are all, in essence, ways to stabilize step size.

::: warning Common Pitfall
The learning rate is not "the larger, the faster the learning". An overly large learning rate can make parameters oscillate near the optimum or even diverge. In practice, small values such as $10^{-3}$ and $3\times10^{-4}$ are common, often combined with learning-rate decay or adaptive optimizers.
:::

---

## From Mathematical Formulas to Backpropagation in Code

The previous sections derived derivatives, gradients, and the chain rule on paper. In actual code, we do not hand-calculate the partial derivative of every parameter. Instead, an automatic differentiation system does it for us. Automatic differentiation relies precisely on the chain rule.

For example, consider the simple computation graph

$$
\theta \to y=3\theta \to L=(y-6)^2.
$$

Backpropagation starts from the final loss, first computes $\frac{dL}{dy}$, then multiplies by $\frac{dy}{d\theta}$ to obtain $\frac{dL}{d\theta}$.

The same principle applies to deep networks. Each layer only needs to know its local derivative, and the gradient of the whole chain can be passed back through the chain rule. Training policy networks, value networks, and reward models all depends on this mechanism.

---

## Summary

This page introduced five core calculus tools:

| Concept            | Role                                                             | Role in RL                                                   |
| ------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------ |
| Derivative         | Describes the direction and speed of change at one point         | Decides how a parameter should be adjusted                   |
| Gradient           | Vector of all partial derivatives in a multi-parameter function  | Tells every parameter where to move at once                  |
| Chain rule         | Differentiates composite functions                               | The theoretical foundation of backpropagation                |
| Partial derivative | Differentiates one variable while treating the rest as constants | Computes gradients parameter by parameter in neural networks |
| Learning rate      | Controls the step size of each parameter update                  | Balances training speed and stability                        |

Together, these tools form the complete loop of "compute a gradient, then take one step". The next page applies these mathematical tools directly to policy optimization and derives policy gradients from the intuition that the probabilities of good actions should increase.

> **Next**: [E.3.2 Policy Gradients and Advantage Functions](./calculus-policy-gradient), which applies gradients to policy optimization.
