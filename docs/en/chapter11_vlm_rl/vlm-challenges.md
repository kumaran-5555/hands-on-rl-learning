---
title: '11.2 Special Challenges in VLM RL'
---

# 11.2 Special Challenges in VLM RL: When Vision Meets Reinforcement Learning

In the previous section we ran the VLM GRPO experiment and saw the model evolve from "guessing answers" to "describing the image first, then reasoning." The experiment itself went smoothly — swap in a different multimodal model, adjust the input format, and the core GRPO code barely changes. But when you push VLM RL toward more complex scenarios (medical image analysis, autonomous driving decisions, robotic visual navigation), you encounter three problems that simply do not exist in text-only RL: should the reward be attributed to visual understanding or text reasoning? Should the visual encoder be updated along with RL? Will the model "see" things that are not in the image?

These three questions are not independent minor annoyances — they are the core theoretical challenges of VLM RL. How well they are solved directly determines whether the system can work in real scenarios.

![VISTA-Gym Main Results](../../chapter11_vlm_rl/images/ref-vista-gym-workflow.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: VISTA-Gym / VISTA-R1 main results table. The w/o Tools and w/o Reasoning ablations show that capability gains in visual tasks come not only from scaling the model but also from the combined effect of tool verification, reasoning trajectories, and reward design. Source: <a href="https://www.eigenai.com/blog/vista-gym-vista-r1" target="_blank" rel="noopener noreferrer">VISTA-Gym / VISTA-R1 Blog</a></em>
</div>

## 11.2.1 Reward Attribution: Visual Tokens and Text Tokens

In text-only RL, reward attribution is not a problem — the entire response is generated from text tokens, so the reward naturally belongs to the whole generation process. But in a VLM, a response's quality depends on two stages of capability: **visual understanding** (did the model correctly "see" the image content) and **text reasoning** (did the model make a reasonable derivation based on correct visual information).

For example, show the model an image with 3 circles and 2 triangles, and ask "How many circles are in the image?" The model answers "There are 2 circles in the image." This answer is wrong. But the cause of the error could be one of two things:

- **Visual error**: The model "saw" wrong, identifying 3 circles as 2. In this case, the reward signal should tell the visual encoder "you need to look more carefully."
- **Reasoning error**: The model "saw" correctly (its internal representation did identify 3 circles) but said the wrong number when generating text. In this case, the reward signal should tell the text decoder "your reasoning is wrong."

The problem is that in current VLM RL frameworks, we typically have only one scalar reward score and cannot naturally distinguish between these two cases. A more practical approach is to decompose an error into several observable checkpoints:

| Checkpoint         | What to Observe                                                  | Possible Training Action                                    |
| ------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| Visual grounding   | Whether attention, selected regions, or IoU are reasonable       | Add grounding reward or visual consistency checks           |
| Text reasoning     | Whether the visual description is right but the derivation fails | Strengthen reasoning format, process reward, or verifier    |
| Cross-modal fusion | Whether image evidence and text conclusion agree                 | Differential learning rates, freeze ViT, or staged training |

The current mainstream approach is **holistic attribution** — distributing the reward across the entire sequence (visual tokens + text tokens), letting gradients update both the visual encoder and text decoder simultaneously. This is simple and direct, but has a fundamental problem: if the visual encoder's parameters are damaged by RL gradients, the model may lose image understanding ability — like someone being shoved hard while solving a math problem and then being unable to even read the question.

Another approach is **freezing the visual encoder** — RL only updates the text decoder's parameters. This guarantees that visual understanding is not degraded, but the cost is that the model cannot improve visual understanding through RL. In the geometric figure experiment from the previous section, this could mean the model can never accurately distinguish overlapping figures — because the visual encoder has no opportunity to learn this capability.

| Strategy                    | Strength                           | Weakness                            | Best Fit                                          |
| --------------------------- | ---------------------------------- | ----------------------------------- | ------------------------------------------------- |
| Full update                 | Jointly optimizes vision + text    | Visual encoder may be damaged       | Tasks where visual understanding must improve     |
| Frozen encoder              | Protects visual ability            | Cannot improve visual understanding | Scenarios with a strong pretrained visual encoder |
| Differential learning rates | Balances protection and adaptation | More hyperparameter tuning          | General recommended setting                       |

Differential learning rates is the most common compromise — the visual encoder uses 1/10 the learning rate of the text decoder. This both protects visual features and allows moderate visual optimization.

```python
# ==========================================
# Differential learning-rate configuration
# ==========================================

def setup_optimizer_with_lr_decay(model, text_lr=1e-6, vision_lr=1e-7):
    """Use different learning rates for visual encoder and text decoder"""
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters()
                       if 'vision' in n or 'vit' in n],
            'lr': vision_lr,  # Visual encoder: smaller learning rate
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if 'vision' not in n and 'vit' not in n],
            'lr': text_lr,    # Text decoder: normal learning rate
            'weight_decay': 0.01,
        },
    ]
    return torch.optim.AdamW(param_groups)
```

## 11.2.2 Visual Hallucination: The Model "Sees" Things That Are Not There

Visual hallucination is one of the most troublesome problems for VLMs. It refers to the model describing content in its response that simply does not exist in the image. For example, the image contains only one red triangle, but the model says "I see 3 red triangles and 2 blue circles in the image."

Visual hallucination does not exist in text-only RL — because a text-only model does not "see" anything; all its outputs are generated from text input. But a VLM's input includes an image, and the model must make judgments about the image's content, and those judgments can be wrong.

In RL training, visual hallucination can appear in a particularly insidious way. If one of the model's hallucinations happens to receive a high reward (e.g., it "fabricated" the correct number of figures), RL will reinforce this behavior — the model learns that "guessing" is more cost-effective than "looking." This is essentially the same as the reward hacking discussed in Chapter 7, but with an additional dimension: the model can cheat not only in text generation but also in visual understanding.

Several strategies for addressing visual hallucination:

**Strategy 1: Visual grounding checks.** Add visual consistency checks to the reward function — does the model's description match the image? This requires an additional verification model, or cross-validation using OCR/object detection tools.

**Strategy 2: Uncertainty penalties.** If the model is overly certain about visual content (e.g., saying "there are 3 circles" rather than "there seem to be 2-3 circles") and the description does not match reality, apply an additional penalty. Encourage the model to express uncertainty when unsure.

**Strategy 3: Multi-turn verification.** First have the model describe the image content, then use another model (or rule system) to verify the description's accuracy. Only responses that pass verification receive full reward. This essentially embeds a "fact-checking" step in the reward function.

<details>
<summary>Exercise: Why is visual hallucination more likely to worsen in RL training than in SFT training?</summary>

In SFT training, the model's output is constrained within the range of human-annotated "standard answers" — if the standard answer is "there are 2 circles in the image," the model is trained to say "there are 2 circles in the image." Human annotations serve as a "safety net."

But in RL training, the model discovers high-reward behaviors through trial and error. If it occasionally "fabricates" a coincidentally correct answer and receives a high reward, that behavior gets reinforced. Worse, RL's exploration mechanism encourages the model to try various strategies — including the "don't look at the image, just guess" strategy. If this strategy happens to work (on simple tasks, the probability of guessing correctly is not low), it gets rapidly reinforced.

This is why reward function design in VLM RL is more critical than in text-only RL — you must evaluate not only "is the answer correct" but also "did the model actually look at the image."

</details>

## 11.2.3 VLM-RL in Autonomous Driving

VLM RL is not just an academic experiment; it has already shown tremendous potential in real-world applications. Autonomous driving is one of the most prominent directions.

Imagine an autonomous driving system architecture: the VLM receives images of the road ahead, generates an understanding of the current scene ("50 meters ahead a pedestrian is crossing, a truck in the left lane is changing lanes"), and then generates driving decisions based on this understanding. RL's role is to train the VLM to produce better scene understanding and decisions — not by training with human-annotated "standard answers" (because you cannot enumerate all possible scenarios) but by using reward signals to guide the model's learning.

In engineering practice, this closed loop is typically not written as simply "image → action → reward" but decomposed into a more conservative chain:

| Stage                 | Main Question                             | Safety Constraint                                               |
| --------------------- | ----------------------------------------- | --------------------------------------------------------------- |
| Scene understanding   | Were key traffic participants identified? | Trigger fallback on low confidence                              |
| Action candidates     | Is the current action reasonable?         | Hard-filter dangerous actions                                   |
| Simulation replay     | Are the consequences stable?              | Cover edge cases in simulation                                  |
| Reward update         | Does long-term safety improve?            | Prioritize safety reward over comfort and efficiency            |
| Deployment monitoring | Did out-of-distribution scenes appear?    | Allow only controlled online updates, no direct trial-and-error |

In autonomous driving, reward function design is much more complex than geometric figure counting. It typically includes three dimensions:

**Safety reward.** This is the most important dimension — any behavior leading to collision or danger should be severely punished. Safety constraints are typically implemented as **hard constraints**: certain actions (running red lights, driving the wrong way) never receive positive rewards, regardless of what justification the VLM provides.

**Comfort reward.** Driving should be not only safe but also comfortable — hard braking and sharp turns make passengers uncomfortable. Comfort constraints are typically implemented as **soft constraints**: as a regularization term in the reward function, traded off against the safety reward.

**Efficiency reward.** While ensuring safety and comfort, reach the destination as quickly as possible. Efficiency rewards encourage the model to choose shorter routes and more reasonable speeds.

```python
# ==========================================
# Autonomous-driving VLM-RL reward function
# ==========================================

def driving_reward(scene_description, action, telemetry):
    """
    Autonomous driving reward function
    - scene_description: VLM's description of the scene
    - action: driving action (steering, acceleration, braking)
    - telemetry: sensor data (speed, distance, lane position)
    """
    reward = 0.0

    # 1. Safety (hard constraint)
    if telemetry['collision_risk'] > 0.8:
        return -10.0  # High collision risk → large penalty

    if telemetry['red_light_violation']:
        return -10.0  # Running red light → large penalty

    if telemetry['speed_limit_exceeded']:
        reward -= 5.0  # Speeding → heavy penalty

    # 2. Comfort (soft constraint)
    jerk = abs(telemetry['acceleration_change'])  # Acceleration change rate
    reward -= 0.1 * jerk  # Hard acceleration/braking → small penalty

    lateral_error = abs(telemetry['lane_deviation'])  # Lane deviation
    reward -= 0.05 * lateral_error

    # 3. Efficiency (positive reward)
    if telemetry['speed'] > 0:  # Moving
        reward += 0.1  # Encourage forward progress
    if telemetry['distance_to_goal'] < telemetry['prev_distance']:
        reward += 0.2  # Approaching destination → reward

    # 4. Scene understanding quality (if VLM's description matches sensors)
    if scene_matches_sensors(scene_description, telemetry):
        reward += 0.3  # VLM correctly understood the scene

    return reward
```

A unique challenge in autonomous driving VLM-RL is the **contradiction between safety and exploration**. RL needs to explore new strategies to find better driving approaches, but exploration itself may produce unsafe driving behaviors. You cannot let the model "trial-and-error" on real roads — one mistake could cause an accident. Therefore, autonomous driving VLM-RL is trained almost entirely in simulation environments before transferring to the real world. This is fundamentally the same Sim-to-Real problem discussed in Section 12.1 on embodied intelligence.

Another challenge is **latency constraints**. In text-only scenarios, a model taking 2 seconds to generate a response is perfectly acceptable. But in autonomous driving, a 2-second delay means the vehicle travels 60 meters blind on a highway. VLM inference must complete in milliseconds — requiring the model to be small enough and fast enough, creating a tension with the large models typically used in RL training.

## 11.2.4 Architecture Choices for Multimodal Policies

Finally, let us summarize the architectural choices for VLM RL:

| Architecture                     | Visual Encoding                  | Fusion Method     | RL Update Range      | Best Fit                    |
| -------------------------------- | -------------------------------- | ----------------- | -------------------- | --------------------------- |
| ViT + Transformer                | Independent ViT                  | Cross-attention   | Full or differential | General VLM RL              |
| Unified Transformer              | Shared Transformer               | Patch embedding   | Full                 | Resource-limited settings   |
| Frozen ViT + lightweight decoder | Pretrained ViT (frozen)          | Linear projection | Text decoder only    | Fast iteration              |
| Multiple visual encoders         | Multiple ViTs (different scales) | Attention fusion  | Selective update     | High-precision visual tasks |

"ViT + Transformer" is currently the most mainstream architecture — the ViT encodes images into visual tokens, which then interact with text tokens through cross-attention. During RL training, you can choose full updates or differential learning rates.

"Frozen ViT + lightweight decoder" suits fast-iteration scenarios — for example, if you want to quickly validate a reward function design without spending significant compute on visual encoding. Freezing the ViT eliminates gradient computation for the visual component, potentially speeding up training 3-5x.

"Multiple visual encoders" suits tasks requiring extremely high visual precision — such as medical image analysis or satellite image interpretation. Multiple ViTs process visual information at different scales or modalities (e.g., one handles overall layout, another handles fine texture), then integrate through an attention fusion layer. During RL updates, you can choose to update only the fusion layer, preserving each ViT's independent feature extraction capability.

```python
# ==========================================
# VLM RL architecture comparison: training speed
# ==========================================

# Assume total model parameters: 3B, with ViT 1B, Transformer 2B
architectures = {
    "Full update": {
        "Trainable params": "3B",
        "Per-step time": "~8s",
        "Visual protection": "Weak",
        "Visual optimization": "Strong",
    },
    "Differential LR": {
        "Trainable params": "3B (ViT lr=1e-7, LM lr=1e-6)",
        "Per-step time": "~8s",
        "Visual protection": "Medium",
        "Visual optimization": "Medium",
    },
    "Frozen ViT": {
        "Trainable params": "2B",
        "Per-step time": "~5s",
        "Visual protection": "Strong",
        "Visual optimization": "None",
    },
}

for name, config in architectures.items():
    print(f"Strategy: {name}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
```

## 11.2.5 VLM RL Challenge Checklist

Summarizing this section's challenges and corresponding mitigations:

| Challenge                  | Root Cause                                          | Mitigation                                         | Tradeoff                                                |
| -------------------------- | --------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| Reward attribution         | Visual and text tokens share one reward signal      | Differential learning rates / holistic attribution | Protecting vision vs improving vision                   |
| Visual hallucination       | Model fabricates content not in the image           | Visual grounding checks / uncertainty penalties    | Training complexity vs reliability                      |
| Visual encoder degradation | RL gradients damage pretrained features             | Freeze ViT / small learning rate                   | Visual understanding protection vs room for improvement |
| Safety vs exploration      | RL exploration may produce unsafe actions           | Simulation training + hard constraints             | Simulation-reality gap                                  |
| Inference latency          | Large models are slow                               | Distillation / speculative decoding                | Accuracy vs speed                                       |
| Multi-scale vision         | Different tasks need different visual granularities | Multi-encoder architecture                         | Compute cost vs precision                               |

These challenges are not isolated problems but interconnected systemic difficulties. For example, solving visual hallucination requires finer-grained reward attribution; finer attribution requires a better visual encoder; a better encoder requires a larger model and more compute. Every improvement involves trade-offs across multiple dimensions.

The challenges of VLM RL go well beyond those listed here — multimodal policy representation, cross-modal reward attribution, visual hallucination prevention, and the safety-efficiency balance are all active research directions. Next we look at frameworks that are trying to address these problems — [VLM RL Frameworks and Frontiers](./vlm-frameworks).

## References

- [VISTA-Gym / VISTA-R1 Blog](https://www.eigenai.com/blog/vista-gym-vista-r1) — shows ablation results for tools, reasoning trajectories, and reward design in visual QA tasks.
- [VLM-R1 GitHub](https://github.com/om-ai-lab/VLM-R1) — provides grounding reward curves, useful for understanding how visual rewards enter VLM RL training.
