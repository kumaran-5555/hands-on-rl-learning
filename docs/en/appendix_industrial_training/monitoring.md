---
title: B.7 Monitoring
search: false
---

# B.7 Monitoring

# Legacy Page: Training Monitoring and Troubleshooting (Merged into B.3)

> This page is kept as an entry point for legacy links. The core content has been merged into the "Training Monitoring and Troubleshooting" part of [B.3 RL Post-Training and Agentic RL Benchmarks](./evaluation-badcase). The original text is preserved below (in English translation) so readers coming from old links can map the content.

RL training is not as stable as supervised learning. Reward can collapse, the policy can collapse, and KL divergence can spike, and these changes can happen within just a few training steps.

This section answers one question: **when training goes wrong, how do you detect it quickly and identify the cause?**

## A Real-Time Monitoring Dashboard

Six metrics you should watch in real time during training:

| Metric              | Normal Pattern       | What an Anomaly Usually Means                                              |
| ------------------- | -------------------- | -------------------------------------------------------------------------- |
| **Training Reward** | Steady increase      | A sudden crash = policy failure                                            |
| **KL Divergence**   | Slow growth          | A spike = policy drift too large                                           |
| **Policy Entropy**  | Slow decrease        | A sharp drop = premature convergence (policy collapses to a narrow region) |
| **Clip Fraction**   | < 0.3                | Sustained > 0.3 = updates are too aggressive                               |
| **Value Loss**      | Keeps decreasing     | Not decreasing = the critic is not learning                                |
| **Reward Margin**   | Stable or increasing | Shrinking = the reward model is losing discrimination                      |

**The most dangerous signal**: reward goes up while entropy collapses. The model may be reward hacking (learning a shortcut that "games" the reward). See Appendix A for more discussion: [Appendix A](/appendix_common_pitfalls/intro).

## Quick Lookup Table for Common Problems

| Symptom                           | Likely Cause           | Fix                                                           |
| --------------------------------- | ---------------------- | ------------------------------------------------------------- |
| Reward crashes                    | Policy collapse        | Lower the learning rate; increase the clip epsilon            |
| KL spikes                         | Step size too large    | Decrease batch size or increase the KL penalty coefficient    |
| Entropy collapses                 | Not enough exploration | Increase the entropy bonus                                    |
| Evaluation metrics do not improve | Reward hacking         | Check correlation between RM score and true quality           |
| OOM                               | Model/data too large   | Use FSDP/ZeRO-3; reduce batch size                            |
| Reward rises but quality drops    | Reward model failure   | Roll back to the best checkpoint and retrain the reward model |

## Monitoring Tools

- **Weights & Biases (wandb)**: the most common option; one line to integrate, automatically tracks all metrics
- **TensorBoard**: native PyTorch integration for local visualization
- **Custom dashboards**: Grafana + Prometheus, common in production environments

```python
# Example: wandb integration
import wandb

wandb.init(project="my-rl-training")
wandb.log(
    {
        "reward": mean_reward,
        "kl_divergence": kl_div,
        "entropy": entropy,
        "clip_fraction": clip_frac,
    }
)
```

## When You Should Stop

Do not wait until training finishes to look at results. Stop immediately if any of the following happens:

1. **KL divergence exceeds a threshold** (often 0.1-0.15): the policy has drifted too far.
2. **Reward decreases for N consecutive steps** (N depends on your tolerance; commonly 50-100 steps).
3. **Entropy drops close to zero**: the policy has collapsed; continuing is unlikely to help.
4. **Benchmark scores drop twice in a row**: the model may be regressing.

Best practice: run a benchmark evaluation every 100-500 steps and set up automatic rollback. Once the evaluation score falls below the historical best, automatically revert to the previous checkpoint. See the automation loop described in [B.3 RL Post-Training and Agentic RL Benchmarks](./evaluation-badcase).
