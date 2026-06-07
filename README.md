Autonomous Driving with Reinforcement Learning

**Student:** Ilhan Caner
**Course:** CMP4501 – Intro to Artifical Intelligence and Expert Systems
**Project Track:** Autonomous Driving with Highway-Env



Training Evolution

https://github.com/user-attachments/assets/37b17dcc-67ee-4d31-8d55-c6384ce2a386 (untrained)

https://github.com/user-attachments/assets/64d0c30c-4df1-4f4d-88e5-ac0454848470 (partially trained)

https://github.com/user-attachments/assets/74077869-6f26-4c8f-9467-aade20e988a0 (fully trained)




The animation demonstrates the progression of the reinforcement learning agent through three stages:

1. **Untrained Agent** – Random driving behavior with frequent crashes.
2. **Partially Trained Agent (10,000 Steps)** – Begins avoiding collisions and maintaining lanes.
3. **Fully Trained Agent (50,000 Steps)** – Consistently drives at high speed while safely navigating traffic.


Project Overview:

The goal of this project is to train an autonomous vehicle capable of driving safely and efficiently in a dynamic highway environment using Reinforcement Learning.

The agent must learn to balance multiple objectives:

* Maintain high speed
* Avoid collisions
* Follow lane discipline
* Adapt to changing traffic conditions

The environment used is Highway-Env integrated with Gymnasium.

---

Methodology

## Reward Function

The reward function provided by the environment was customized using the following components:


R_t = 1.5S_t + 0.6L_t - 5.0C_t - 0.05LC_t

Where:
- S_t = Speed reward
- L_t = Right lane reward
- C_t = Collision penalty
- LC_t = Lane change penalty

  

Reward Components:

| Component           | Value | Purpose                             |
| ------------------- | ----- | ----------------------------------- |
| High Speed Reward   | +1.5  | Encourages efficient driving        |
| Right Lane Reward   | +0.6  | Promotes lane discipline            |
| Collision Penalty   | -5.0  | Strongly discourages crashes        |
| Lane Change Penalty | -0.05 | Prevents unnecessary lane switching |

### Why This Reward Function?

Autonomous driving is a multi-objective task.

A reward focused only on speed would cause reckless driving. A reward focused only on safety could result in slow or stationary behavior.

The chosen reward function balances:

* Efficiency
* Safety
* Stability

allowing the agent to learn realistic highway driving behavior.

---

## Reinforcement Learning Algorithm

### Proximal Policy Optimization (PPO)

This project uses PPO from Stable-Baselines3.

PPO was selected because:

* Stable training performance
* Good sample efficiency
* Widely used for continuous decision-making problems
* Strong performance in Gymnasium environments

---

## Hyperparameters

| Parameter             | Value  |
| --------------------- | ------ |
| Learning Rate         | 3e-4   |
| Number of Steps       | 512    |
| Batch Size            | 128    |
| Epochs                | 4      |
| Discount Factor (γ)   | 0.99   |
| Training Timesteps    | 50,000 |
| Parallel Environments | 4      |

### Neural Network

The PPO implementation uses the default MLP policy provided by Stable-Baselines3.

The policy network receives the environment observations and outputs probabilities over the available driving actions.

---

## State Space

The agent observes kinematic information about nearby vehicles.

Observed features:

* Vehicle presence
* Relative x position
* Relative y position
* Velocity in x direction
* Velocity in y direction

The observation contains information for the ego vehicle and nearby traffic vehicles.

This allows the agent to estimate:

* Traffic density
* Relative distances
* Closing speeds
* Lane occupancy

---

## Action Space

The environment uses **DiscreteMetaAction**.

Available actions include:

| Action            |
| ----------------- |
| Maintain Speed    |
| Accelerate        |
| Decelerate        |
| Change Lane Left  |
| Change Lane Right |

The agent selects one action at each decision step.

---

#Training Configuration

### Environment Settings

| Setting              | Value |
| -------------------- | ----- |
| Lanes                | 3     |
| Vehicles             | 40    |
| Initial Vehicles     | 20    |
| Spawn Probability    | 0.6   |
| Simulation Frequency | 5     |
| Policy Frequency     | 1     |
| Episode Duration     | 80    |

Dense traffic was intentionally used to increase task difficulty and encourage more robust behavior.

---

#Training Analysis

## Reward Curve

<img width="819" height="420" alt="3c6dc5da-a47d-46cf-92da-8705f1b8e177" src="https://github.com/user-attachments/assets/c92656b5-16b0-4d7c-96c7-5ec32a99f826" />


### Observations

During the initial training phase, rewards were highly unstable because the agent frequently collided with other vehicles.

Between approximately 10,000 and 30,000 training steps, the agent began learning collision avoidance and lane management strategies, resulting in a noticeable improvement in episode rewards.

After around 40,000 training steps, performance became more stable and the agent consistently survived longer while maintaining higher speeds.

The final policy demonstrates significantly safer and more efficient driving compared to the untrained baseline.

---

#Challenges and Failures

## Problem: Frequent Early Collisions

One major challenge was that the agent initially learned highly aggressive behavior.

The vehicle attempted to maximize speed without considering surrounding traffic, leading to frequent crashes and poor episode rewards.

### Solution

Several reward parameters were adjusted:

* Collision penalty increased
* Lane change penalty introduced
* Traffic density increased for better generalization

These modifications encouraged safer driving strategies while preserving efficient highway navigation.

---

## Problem: Overfitting to Simple Traffic Patterns

When traffic density was low, the agent could achieve good performance without learning robust avoidance behavior.

### Solution

The number of surrounding vehicles was increased and new vehicles were spawned throughout episodes.

This forced the policy to react to a wider variety of traffic situations.

---

# 📁 Project Structure

```text
project/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── train.py
│   └── evaluate.py
│
├── assets/
│   ├── reward_plot.png
│   └── checkpoints/
│
└── videos/
```

---

#Installation

```bash
pip install -r requirements.txt
```

---

#Training

```bash
python train.py
```

---

#Evaluation

```bash
python evaluate.py
```

This generates videos demonstrating the evolution of the reinforcement learning agent across different training stages.

---

#Technologies Used

* Python
* Gymnasium
* Highway-Env
* Stable-Baselines3
* PyTorch
* NumPy
* Matplotlib

---

#Conclusion

A PPO-based reinforcement learning agent was successfully trained to drive in dense highway traffic while balancing speed, safety, and lane discipline.

The final policy demonstrates clear improvement over the untrained baseline and highlights the effectiveness of PPO for autonomous driving tasks in simulated environments.
