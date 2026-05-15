---
title: 6.2 Training the Critic
---

# 6.2 Training the Critic

In the previous section, we defined the advantage function

$$
A(s,a) \approx \delta = r + \gamma V(s') - V(s),
$$

and introduced the Critic network as an estimator of the state value function $V(s)$. In this section, we expand the three classic value-estimation methods from Chapter 3,
[DP, MC, and TD](../chapter03_mdp/dp-mc-td), and show how each of them can be used to train a Critic in practice.

::: tip Prerequisites used in this section

- [Three value-estimation methods: DP/MC/TD](../chapter03_mdp/dp-mc-td): principles and comparisons
- [Bellman expectation equation](../chapter03_mdp/value-bellman): theoretical basis for DP-style updates
- [TD Error $\delta$](../chapter03_mdp/dp-mc-td): the core learning signal of TD methods
  :::

## DP: A Theoretical Baseline

If we knew the full transition dynamics $P$ and reward function $R$ of the environment (recall the
[MDP 5-tuple](../chapter03_mdp/mdp)), then we could iterate the Critic directly via the
[Bellman expectation equation](../chapter03_mdp/value-bellman):

$$
V_\phi(s) \leftarrow \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\phi(s') \right].
$$

By repeatedly applying this update to all states, $V_\phi$ converges to the exact value function $V^\pi$.
From there we can go one step further and perform **policy improvement**: at state $s$, choose the action
that maximizes $Q(s,a)$ (recall:
[the greedy optimal policy](../chapter03_mdp/value-q)). The loop

"evaluate the policy $\rightarrow$ improve the policy $\rightarrow$ evaluate again"

is exactly **Policy Iteration**, which is guaranteed (in theory) to converge to an optimal policy.

In real-world problems, however, it is almost never feasible to know the complete $P$ and $R$. For
Actor-Critic methods, DP therefore plays the role of a theoretical baseline: it tells you what the Critic
would compute in an idealized world where the environment is fully known.

## MC: Update the Critic Using Complete Trajectories

Monte Carlo (MC) updates wait until an episode finishes, then use the empirical return $G_t$ (recall:
[the return definition](../chapter03_mdp/mdp)) to train the Critic. The Critic loss is a mean squared error:

$$
L_{\text{Critic}} = \left( G_t - V_\phi(s) \right)^2. \tag{6.3}
$$

The term $G_t - V_\phi(s)$ is the Critic's prediction error: the episode actually achieved $G_t$, but the
Critic previously predicted $V_\phi(s)$. MC methods (recall the update rule
$V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$ in
[MC value updates](../chapter03_mdp/dp-mc-td)) provide an **unbiased estimate** because they use the true
return. The cost is two practical limitations:

1. You **must wait until the episode ends** to compute $G_t$, so you cannot learn step-by-step online.
2. The **variance is large**: $G_t$ can fluctuate drastically across different episodes.

In a neural-network implementation, the MC method is equivalent to: run one full episode, collect all
$(s_t, G_t)$ pairs, then perform a gradient-descent update on the Critic parameters $\phi$ using this batch.

## TD: One-Step Updates

Temporal Difference (TD) learning updates the Critic using the
[TD Error](../chapter03_mdp/dp-mc-td). The Critic loss is:

$$
L_{\text{Critic}} = \left( r + \gamma V_\phi(s') - V_\phi(s) \right)^2 = \delta^2. \tag{6.4}
$$

Minimizing $\delta^2$ means making the Critic's predictions progressively more accurate. The practical
advantages of TD methods (recall the TD(0) update
$V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$ in
[TD(0) updates](../chapter03_mdp/dp-mc-td)) are:

1. You **do not need to wait for the episode to end**. You can update at every step.
2. The **variance is lower**: $V_\phi(s')$ acts as an "anchor" that stabilizes the target.
3. The update cadence **matches the Actor**: both can update once per environment step.

The price you pay is **bias**: $V_\phi(s')$ is itself an estimate, not the true value. This is called
[bootstrapping](../chapter03_mdp/dp-mc-td): using your current estimates to improve your own estimates.
In practice, the bias introduced by bootstrapping is often far smaller than the gain from reducing variance.

## Comparing the Three Methods

|                           | **DP**               | **MC** | **TD**                         |
| ------------------------- | -------------------- | ------ | ------------------------------ |
| **Used to train Critic?** | Theoretical baseline | Usable | **Practical default**          |
| **Need episode to end?**  | No                   | Yes    | No                             |
| **Unbiased?**             | Yes                  | Yes    | No (biased but lower variance) |
| **Variance**              | Low                  | High   | Medium                         |
| **Bootstrapping**         | Yes                  | No     | Yes                            |

In practice, Actor-Critic methods almost always use TD to train the Critic. In more advanced variants
(for example, [GAE in Chapter 7](../chapter07_ppo/gae-reward-model)), MC and TD are combined: a parameter
$\lambda$ interpolates between them to achieve a better bias-variance tradeoff.

## The Full Critic-Training Workflow

Putting the pieces together, a one-step Actor-Critic training loop looks like this:

1. At state $s$, the Actor selects an action $a$. The environment returns reward $r$ and next state $s'$.
2. The Critic computes the current prediction $V_\phi(s)$ and the next-step prediction $V_\phi(s')$.
3. Compute the TD error: $\delta = r + \gamma V_\phi(s') - V_\phi(s)$.
4. Update the Critic parameters $\phi$ by minimizing the loss $\delta^2$.
5. Update the Actor parameters $\theta$ using $\delta$ as an advantage estimate.

The Critic parameters $\phi$ move in the direction that makes $\delta^2$ smaller, so the value predictions
become more accurate. The Actor parameters $\theta$ move in the direction that assigns higher probability
to actions with positive $\delta$, so the policy becomes better. This creates a virtuous cycle: the more
accurate the Critic's evaluation, the faster the Actor can improve; the more diverse actions the Actor
tries, the richer the data the Critic sees, and the more accurate its evaluation becomes.

## References

[^1]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44.

[^2]: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. _ICML_. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
