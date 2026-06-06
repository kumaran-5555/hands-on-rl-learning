---
title: Part II Summary
---

# Part 2: Theory and Methods - Knowledge Summary

## What Did We Learn in This Part?

These five chapters form the theoretical core of the book. We started from the most basic question, "How do we describe decision-making in mathematics?", and progressed all the way to PPO, the most widely used algorithm in modern industry. Mastering this part gives you the key to reading essentially all later LLM alignment algorithms.

After these five chapters, you should understand:

- **The MDP 5-tuple** $(S, A, P, R, \gamma)$: a mathematical language for "an agent makes decisions in an environment."
- **Value functions and Bellman equations**: $V^\pi(s)$ and $Q^\pi(s,a)$ measure "how valuable a state is" and "how valuable an action at a state is." Bellman equations tell us: current value = immediate reward + discounted next-step value.
- **TD error**: $\delta = r + \gamma V(s') - V(s)$ measures the gap between prediction and reality and is the learning signal behind almost all RL algorithms.
- **The three key components of DQN**: Q-network (approximate $Q$ with a neural network), experience replay (break sample correlations), target network (stabilize training targets).
- **The policy gradient theorem**: $\nabla_\theta J = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot G_t]$, directly differentiating the policy, naturally supporting continuous actions.
- **Actor-Critic**: the Actor learns the policy, the Critic learns values; they cooperate through the advantage function $A(s,a) = Q(s,a) - V(s)$.
- **PPO clipping**: use $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ to constrain changes in the policy ratio and prevent unstable, overly large updates.
- **GAE**: $\hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t+k}$, interpolating between bias and variance.

Now let us review the content chapter by chapter.

## Chapter 3: MDP - A Mathematical Description of Decision Problems

### Markov Decision Process

To discuss reinforcement learning rigorously, we need a mathematical framework for "an agent makes decisions in an environment." This framework is the **Markov Decision Process** (MDP), defined by a 5-tuple $(S, A, P, R, \gamma)$:

- $S$ is the set of states. In CartPole, $s = (\text{position}, \text{velocity}, \text{angle}, \text{angular velocity}) \in \mathbb{R}^4$.
- $A$ is the set of actions. In the discrete case, $A=\{a_1, a_2, \ldots\}$; in the continuous case, $A$ is an interval of real values.
- $P(s'|s,a)$ is the transition probability: after taking action $a$ at state $s$, the probability of moving to $s'$. "Markov" means the future depends only on the current state, not on the past. In chess, you only need the current board position; you do not need the full move history.
- $R(s,a)$ is the reward function: the immediate reward after taking action $a$ at state $s$.
- $\gamma \in [0, 1)$ is the discount factor: it controls the tradeoff between "immediate payoff" and "long-term payoff." $\gamma$ near 1 emphasizes the long term; $\gamma$ near 0 emphasizes the present.

The agent's goal is to find a policy $\pi(a|s)$ that maximizes the **discounted cumulative return** from any starting state:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

Why do we need a discount factor $\gamma$? Mathematically, an infinite series must converge; $\gamma < 1$ guarantees $G_t$ is finite. Intuitively, future rewards are less certain than immediate ones: "100 dollars today" is typically more attractive than "100 dollars next year."

### Value Functions and Bellman Equations

With return defined, the next question is: how many "points" is a state worth? How many "points" is a state-action pair worth? This leads to two core concepts.

The **state-value function** $V^\pi(s)$ is the expected return starting from state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s\right]$$

The **action-value function** $Q^\pi(s, a)$ is the expected return when we take action $a$ at state $s$ and then follow policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s, a_t = a\right]$$

They are related by:

$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$$

In words: state value is the policy-weighted average of action values.

The Bellman equation further reveals their recursive structure: you do not need to "see the future," only one step:

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

Intuitively: the value of the current state equals the expected value over actions (weighted by the policy), where each action value equals immediate reward plus discounted value of the next state. This self-consistent equation is the basis for computing value functions.

### TD Error: The Learning Signal Across RL Algorithms

In realistic settings, we usually do not know the transition probabilities $P$ or reward function $R$ (we do not know the environment model). We can only interact with the environment and obtain samples. At this point, a key learning signal appears: the **temporal-difference error (TD error)**:

$$\delta = r + \gamma V(s') - V(s)$$

TD error measures the gap between prediction and reality. $V(s)$ is our prediction of the current state's value, and $r + \gamma V(s')$ is the one-step reward plus our estimate of the next step. If our prediction is perfectly correct, $\delta = 0$. If reality is better than predicted, $\delta > 0$, and we should increase our estimate of $V(s)$.

This simple formula runs through the entire RL landscape. From Q-learning to DQN, from REINFORCE to PPO, the learning signals are essentially TD error or its variants.

### From Slot Machines to GridWorld: Understanding Q-Learning in Code

In the two-armed bandit experiment, we first experienced the "exploration vs exploitation" tension: you want to choose the arm with higher win rate (exploit), but you worry the other arm might actually be better (explore). In 4x4 GridWorld, we ran the full Q-learning loop:

```python
Q = np.zeros((n_states, n_actions))  # initialize Q-table

for episode in range(1000):
    state = env.reset()
    while not done:
        # epsilon-greedy: explore with probability epsilon, otherwise exploit the current best action
        action = epsilon_greedy(Q[state], epsilon)
        next_state, reward, done = env.step(action)
        # update Q-values using TD error
        td_target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (td_target - Q[state, action])
        state = next_state
```

This snippet contains all essential RL elements: store value estimates in a table $Q(s,a)$, balance exploration and exploitation with $\varepsilon$-greedy, and update using TD error. When the state space becomes large (for example, from a 16-cell grid to Atari frames with $210 \times 160$ pixels), the table no longer fits. This is exactly the problem DQN solves.

## Chapter 4: DQN - The Leap from Tables to Neural Networks

### The Curse of Dimensionality and Function Approximation

CartPole has continuous states, but only 4 dimensions. Atari frames, by contrast, are pixel tensors of size $84 \times 84 \times 4$, and the state space has on the order of $256^{28224}$ possibilities, a number larger than the number of atoms in the observable universe. It is impossible to store Q-values for every state in a table.

DQN solves this by **approximating the Q-function with a neural network**. The network takes state $s$ as input and outputs Q-values for each action $Q(s, a_1), Q(s, a_2), \ldots$. The training objective is to minimize squared TD error:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

Here $\theta$ are the parameters of the online network, and $\theta^-$ are the parameters of the target network. This loss means: make the network's prediction $Q(s,a)$ match "one-step reward plus the next-step maximum Q-value."

### The Three Key Components of DQN

A neural network alone is not enough. If we train directly on each new interaction step, consecutive samples are highly correlated (they come from adjacent time steps), and training becomes unstable. DQN introduces three important design choices:

**Experience replay** stores each step $(s, a, r, s')$ into a large buffer, and then samples random mini-batches for training. This breaks temporal correlations and makes gradient updates closer to i.i.d. assumptions.

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

**Target networks** keep a delayed copy of the online network. Every so often, we hard-copy the online parameters into the target network. This keeps TD targets stable for a while and avoids the difficulty of "chasing a moving target." Think of learning to shoot: if the hoop moves every second, it is hard to learn; if it moves only occasionally, you have time to adapt.

**$\varepsilon$-greedy exploration** gradually reduces the probability of random exploration over training, transitioning from broad exploration early to fine-grained exploitation later.

### The DQN Family

The original DQN became famous in the 2015 Atari paper, and many improvements followed. **Double DQN** decouples "action selection" and "action evaluation": use the online network $\theta$ to pick $\arg\max_{a'} Q(s', a'; \theta)$, then use the target network $\theta^-$ to evaluate that action. This reduces overestimation bias in original DQN, analogous to not letting the same person both set an exam and grade it.

**Dueling DQN** decomposes Q-values into state value $V(s)$ and advantage $A(s,a)$:

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$$

When all actions in a state are similarly good, $A(s,a) \approx 0$, and the network mainly needs to learn $V(s)$. This improves efficiency.

## Chapter 5: Policy Gradients - Learning the Policy Directly

### From Value-Based to Policy-Based

DQN follows an indirect route: learn $Q(s,a)$, then choose actions via $\arg\max$. This route has a fundamental limitation: it naturally handles only **discrete and finite** action spaces. A robot arm's torques are continuous values, e.g. $[-10, 10]^6$, and you cannot assign a Q-value to every combination. Text generation in LLMs is even more extreme: at each step you choose from tens of thousands of tokens.

Policy gradient methods take a different route: instead of learning value functions, they **parameterize the policy directly** as $\pi_\theta(a|s)$, then optimize $\theta$ to maximize expected return. The policy gradient theorem states that the gradient of $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$ can be estimated as:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

This formula has a clean intuition: $\nabla_\theta \log \pi_\theta(a_t|s_t)$ points in the direction that makes action $a_t$ more likely under state $s_t$, while $G_t$ tells you whether that direction is good. If $G_t > 0$, increase the action probability; if $G_t < 0$, decrease it. That is the whole REINFORCE algorithm.

```python
def reinforce_update(policy, optimizer, states, actions, returns):
    log_probs = []
    for s, a in zip(states, actions):
        dist = Categorical(policy(s))
        log_probs.append(dist.log_prob(a))

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G  # minus sign: gradient descent = maximize return
    loss /= len(returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Baselines and Variance

REINFORCE has a fatal weakness: **high variance**. If every episode return $G_t$ is around 100, then even a "good" action receives a large positive signal, just slightly less large than others. This makes gradient estimates unstable.

The standard fix is to introduce a **baseline** $b(s)$ and modify the gradient to:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

As long as $b(s)$ does not depend on the action $a$, this modification does not change the expected gradient (because $\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s)] = 0$), but it can greatly reduce variance. The most common baseline is the state-value function $V(s)$, i.e., the Critic.

## Chapter 6: Actor-Critic - Reducing Variance with a Critic

### The Actor-Critic Architecture

Combining the Actor (policy network) and the Critic (value network) yields the Actor-Critic architecture. The Actor selects actions, and the Critic evaluates "how much better this action is than average." That quantity is the **advantage function**:

$$A(s, a) = Q(s, a) - V(s)$$

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
```

During training, the Actor updates the policy using an advantage estimate $A = r + \gamma V(s') - V(s)$, while the Critic updates the value estimate using squared TD error:

```python
# update after one environment step
_, next_value = model(next_state)
td_target = reward + gamma * next_value * (1 - done)
td_error = td_target - value

actor_loss = -log_prob * td_error.detach()  # Actor: update policy using advantage
critic_loss = td_error ** 2                 # Critic: update value using TD error
loss = actor_loss + critic_loss
```

## Chapter 7: PPO - Making Policy Updates More Stable

### Trust Regions and Clipping

Actor-Critic is more stable than REINFORCE, but policy updates can still be too large. If a single update changes the policy dramatically, previously collected data becomes irrelevant, and training can oscillate violently.

PPO (Proximal Policy Optimization) uses an elegant clipping mechanism to address this. It defines the **policy ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

When $r_t = 1$, the new and old policies match. When $r_t > 1$, the new policy is more likely to choose the action; when $r_t < 1$, it is less likely. PPO maximizes:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\;\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

Typically $\varepsilon=0.2$. When $\hat{A}_t > 0$ (good actions), PPO allows $r_t$ to increase at most to $1+\varepsilon$, preventing overly aggressive probability increases. When $\hat{A}_t < 0$ (bad actions), PPO allows $r_t$ to decrease at most to $1-\varepsilon$. This "safety rail" keeps updates within a **trust region**.

The full PPO loss also includes two additional terms:

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 \mathcal{H}[\pi_\theta]$$

where $L^{\text{VF}}$ is the Critic's value-fitting loss (MSE) and $\mathcal{H}$ is the policy entropy bonus. The entropy term encourages exploration and prevents premature collapse to suboptimal deterministic behavior.

### GAE: Balancing Bias and Variance

How we estimate advantage $\hat{A}_t$ matters greatly for PPO. Two naive extremes are:

- use one-step TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$: high bias but low variance;
- use full-trajectory return $G_t - V(s_t)$: unbiased but high variance.

**GAE** (Generalized Advantage Estimation) interpolates between them:

$$\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}$$

Here $\lambda \in [0, 1]$ controls the interpolation. $\lambda = 0$ reduces to one-step TD (large bias), and $\lambda = 1$ reduces to full returns (large variance). In practice, $\lambda = 0.95$ is common.

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

### From LunarLander to LLMs

PPO was first validated on classic RL environments such as LunarLander, but its real power shows up in LLM alignment. In RLHF, PPO simultaneously manages four models: Actor (the language model being trained), Critic (value network), Reference (the frozen original model used for KL regularization), and Reward Model (the judge scoring responses). The Bradley-Terry preference model defines the reward-model training objective:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

This framework is the starting point of Chapter 8, and DPO's key observation is that this framework can be optimized without explicitly training a reward model.

## Summary

Part 2 followed a full theoretical arc: MDP provides the language of decision problems -> Bellman equations provide a recursive method to compute values -> DQN approximates $Q$ with neural networks and resolves the curse of dimensionality -> policy gradients optimize policies directly and support continuous actions -> Actor-Critic introduces a Critic to reduce variance -> PPO uses clipping and GAE for stable training.

Every concept on this path will reappear in later LLM alignment chapters. Understanding PPO clipping helps you understand GRPO's within-group normalization; understanding the Actor-Critic division of labor helps you understand the roles of the four models in RLHF.

> **Next stop**: [Part 3: The LLM Era](/en/chapter08_rlhf/intro)
