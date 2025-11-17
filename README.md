# DQN Atari Agent - Training and Evaluation

This project implements a Deep Q-Network (DQN) agent using Stable Baselines3 to train and evaluate an agent playing the Atari game **BreakoutNoFrameskip-v4**. The agent learns to play the game through reinforcement learning, and we systematically explore hyperparameter configurations to optimize performance.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Environment Selection](#environment-selection)
-   [Installation](#installation)
-   [Project Structure](#project-structure)
-   [Usage](#usage)
-   [Policy Comparison: CNN vs MLP](#policy-comparison-cnn-vs-mlp)
-   [Hyperparameter Tuning Experiments](#hyperparameter-tuning-experiments)
-   [Results and Discussion](#results-and-discussion)
-   [Video Demonstration](#video-demonstration)
-   [Team Members](#team-members)

## Project Overview

This assignment demonstrates the implementation of a DQN agent using Stable Baselines3 and Gymnasium. The project consists of two main scripts:

1. **`train.py`**: Trains a DQN agent on the Atari Breakout environment and saves the trained model
2. **`play.py`**: Loads the trained model and evaluates the agent's performance with video recording

## Environment Selection

**Selected Environment**: `BreakoutNoFrameskip-v4`

Breakout is a classic Atari game where the agent controls a paddle to bounce a ball and break bricks. This environment is well-suited for DQN training because:

-   It provides clear visual feedback
-   The state space is manageable but complex enough to demonstrate learning
-   Success metrics (score/reward) are straightforward to measure
-   It's a standard benchmark in reinforcement learning research

## Installation

### Prerequisites

-   Python 3.8 or higher
-   pip package manager

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/reponseashimwe/dqn-atari
    cd dqn-atari
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    This will install:

    - `gymnasium[atari]` - Atari environments
    - `ale-py` - Arcade Learning Environment
    - `stable-baselines3[extra]` - DQN implementation
    - `torch` - PyTorch backend
    - `opencv-python` - Image processing
    - `autorom[accept-rom-license]` - Atari ROMs
    - `numpy` - Numerical operations

## 📁 Project Structure

```
dqn-atari/
├── train.py              # Training script
├── play.py               # Evaluation and video recording script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── dqn_model.zip        # Trained model (best performing)
├── video.mp4            # Best episode video recording
├── videos/              # All recorded episodes
├── experiments/         # Experiment results and models
│   └── reponse/
│       ├── experiment_results.json
│       └── Set_*_*/     # Individual experiment folders
└── notebooks/           # Jupyter notebooks for experimentation
```

## 🚀 Usage

### Training the Agent

To train a DQN agent with the default (winning) hyperparameters:

```bash
python train.py
```

The script will:

-   Create a vectorized environment with 4 parallel environments
-   Train for 1,000,000 timesteps
-   Save checkpoints every 200,000 steps
-   Save the final model as `dqn_model.zip`
-   Log training metrics to TensorBoard (in `dqn_logs_set_5/`)

**Key Training Parameters**:

-   **Environment**: BreakoutNoFrameskip-v4
-   **Policy**: CNNPolicy (Convolutional Neural Network)
-   **Total Timesteps**: 1,000,000
-   **Parallel Environments**: 4
-   **Frame Stacking**: 4 frames

### Evaluating the Agent

To evaluate the trained agent and generate video recordings:

```bash
python play.py
```

The script will:

-   Load the trained model from `dqn_model.zip`
-   Run 10 evaluation episodes
-   Record videos of all episodes
-   Calculate average reward and identify the best episode
-   Save the best episode as `video.mp4` in the root directory
-   Keep all episode videos in the `videos/` folder

**Evaluation Settings**:

-   **Episodes**: 10
-   **Policy**: GreedyQPolicy (deterministic, best action selection)
-   **Video Recording**: All episodes recorded, best one saved

## 🔬 Policy Comparison: CNN vs MLP

### Overview

We compared two policy architectures:

-   **CNNPolicy**: Convolutional Neural Network (suitable for image-based observations)
-   **MLPPolicy**: Multilayer Perceptron (typically for vector observations)

### Results

_[This section will be updated after completing the MLP policy experiments]_

**Expected Findings**:

-   CNNPolicy should perform better for Atari games due to spatial feature extraction
-   MLPPolicy may struggle with high-dimensional image observations
-   Training time and memory usage comparisons

### Implementation Notes

To test MLPPolicy, modify `train.py` line 59:

```python
# Change from:
model = DQN("CnnPolicy", env, ...)

# To:
model = DQN("MlpPolicy", env, ...)
```

_Note: MLPPolicy experiments are pending and will be documented here upon completion._

##  Hyperparameter Tuning Experiments

Each team member conducted 10 different hyperparameter experiments. The following table documents all configurations tested and their observed behaviors.

### Hyperparameter Definitions

-   **Learning Rate (lr)**: Controls how quickly the agent updates its Q-network weights
-   **Gamma (γ)**: Discount factor for future rewards (0.0 to 1.0)
-   **Batch Size**: Number of experiences sampled from replay buffer per update
-   **Epsilon Start**: Initial exploration rate (1.0 = fully random)
-   **Epsilon End**: Final exploration rate (lower = more exploitation)
-   **Epsilon Decay**: Fraction of training over which epsilon decays

### Experiment Results Table

| Member Name      | Hyperparameter Set                                                                                               | Noted Behavior                                                                                                                  | Avg Reward | Performance   |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------- |
| **Reponse**      | **Set 1 (Baseline)** - lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1     | Baseline configuration. Stable learning curve with consistent improvement. Good balance between exploration and exploitation.   | 5.40       | Good          |
|                  | **Set 2 (High LR)** - lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1      | High learning rate caused unstable training. Agent failed to learn effectively, likely due to overshooting optimal Q-values.    | 0.0        | Failed        |
|                  | **Set 3 (Low Gamma)** - lr=1e-4, gamma=0.90, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | Low gamma (0.90) reduced long-term planning. Agent focused too much on immediate rewards, limiting performance.                 | 1.40       | Poor          |
|                  | **Set 4 (Extended Eps)** - lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.5 | Extended exploration period. Slower convergence but maintained exploration longer, resulting in moderate performance.           | 3.0        | Moderate      |
|                  | **Set 5 (Large Batch)** - lr=1e-4, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | **WINNING** - Large batch size improved stability. More stable gradient updates led to better learning and highest performance. | 7.0        | Best          |
| **Kanisa Thiak** | Set 1 - lr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.05                          | Baseline configuration. Stable but needs more timesteps; occasional scores observed.                                            | 0.00       | Failed/Poor   |
|                  | Set 2 - lr=5e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.10                           | Lower learning rate: slower, more stable learning expected; may require longer training.                                        | 0.50       | Good          |
|                  | Set 3 - lr=0.00025, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_fraction=0.10                         | Faster updates but risk of instability.                                                                                         | 0.60       | Moderate      |
|                  | Set 4 - lr=0.0001, gamma=0.997, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.05                         | Higher gamma favors long-term reward; best observed.                                                                            | 0.70       | Best          |
|                  | Set 5 - lr=0.0001, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.10                          | Larger batch size → more stable gradients. Expected small improvement.                                                          | 0.55       | Good          |
|                  | Set 6 - lr=5e-05, gamma=0.997, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.05                          | Very stable but slow. Likely moderate long-term reward.                                                                         | 0.65       | Very Good     |
|                  | Set 7 - lr=0.00025, gamma=0.997, batch=32, eps_start=1.0, eps_end=0.05, eps_fraction=0.10                        | Aggressive LR + high gamma → high variance.                                                                                     | 0.45       | Moderate/Poor |
|                  | Set 8 - lr=7.5e-05, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.05                         | Balanced setup; stable learning expected.                                                                                       | 0.60       | Good          |
|                  | Set 9 - lr=0.00015, gamma=0.995, batch=32, eps_start=1.0, eps_end=0.01, eps_fraction=0.08                        | Slightly reduced gamma; decent convergence.                                                                                     | 0.58       | Good          |
|                  | Set 10 - lr=0.0001, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.005, eps_fraction=0.05                        | Very small final eps → strong exploitation but risk of suboptimal policy.                                                       | 0.40       | Moderate/Poor |


| MEMBER NAME       | Hyperparameter Set                                                                           | Noted Behavior                                                                                                                           | Avg Reward | Performance |
| ----------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------- |

| **Jolly Umulisa** | **Set 1** – lr = 1e-4, γ = 0.99, batch = 32, eps_start=1.0, eps_end=0.05, eps_fraction=0.10  | Stable and consistent early learning. The agent explored well and achieved moderate performance with small improvements across episodes. | **0.5**    | Moderate    |
| **Jolly Umulisa** | **Set 2** – lr = 5e-4, γ = 0.99, batch = 32, eps_start=1.0, eps_end=0.05, eps_fraction=0.10  | Higher learning rate caused unstable updates. The agent performed worse and rewards fluctuated heavily, showing poor convergence.        | **0.2**    | Poor        |
| **Jolly Umulisa** | **Set 3** – lr = 1e-4, γ = 0.90, batch = 32, eps_start=1.0, eps_end=0.05, eps_fraction=0.10  | Low gamma made the agent focus on short-term rewards. Learning was inconsistent and overall performance remained low.                    | **0.4**    | Weak        |
| **Jolly Umulisa** | **Set 4** – lr = 1e-4, γ = 0.99, batch = 32, eps_start=1.0, eps_end=0.05, eps_fraction=0.50  | Slower decay improved exploration. The agent discovered more rewarding actions and achieved significantly better performance.            | **0.7**    | Good        |
| **Jolly Umulisa** | **Set 5** – lr = 1e-4, γ = 0.99, batch = 128, eps_start=1.0, eps_end=0.05, eps_fraction=0.10 | Larger batch size stabilized updates. Provided the best learning curve and the highest average reward among all 5 experiments.           | **0.8**    | **Best**    |

## Jolly Umulisa – Summary of Results

For my contribution, I ran five Deep Q-Learning experiments using different combinations of learning rate, discount factor, batch size, and exploration decay. Each experiment was trained for 100,000 timesteps in the Breakout Atari environment. The results showed that higher learning rates and lower gamma values led to unstable or short-sighted learning, while slower exploration decay improved consistency. The best overall performance came from the configuration with a larger batch size (Set 5), achieving an average reward of 0.8. This model was selected as my final agent and used for evaluation in play.py.


##  Results and Discussion

### Key Findings

#### 1. Learning Rate Impact

**Observation**: Learning rate significantly affects training stability and convergence speed.

-   **Low LR (1e-4)**: Provided stable learning with consistent improvement. This was the optimal learning rate for our experiments.
-   **High LR (5e-4)**: Caused complete training failure (reward: 0.0). The agent was unable to learn, likely due to:
    -   Overshooting optimal Q-values
    -   Unstable gradient updates
    -   Poor convergence behavior

**Conclusion**: A learning rate of 1e-4 provides the best balance between learning speed and stability for this environment.

#### 2. Discount Factor (Gamma) Importance

**Observation**: Gamma controls the agent's planning horizon and significantly impacts performance.

-   **High Gamma (0.99)**: Enabled long-term planning, resulting in better performance (rewards: 5.40, 7.0)
-   **Low Gamma (0.90)**: Limited the agent to short-term rewards, reducing performance to 1.40

**Conclusion**: A gamma of 0.99 is optimal for Breakout, as the game requires planning multiple steps ahead to maximize score.

#### 3. Batch Size Effects

**Observation**: Batch size had a substantial impact on training stability and final performance.

-   **Small Batch (32)**: Standard configuration, achieved moderate performance (5.40 average reward)
-   **Large Batch (128)**: **Best performing configuration** (7.0 average reward)
    -   More stable gradient estimates
    -   Reduced variance in updates
    -   Better generalization

**Conclusion**: Larger batch sizes (128) provide more stable learning and better final performance, though they require more memory.

#### 4. Exploration Strategy

**Observation**: Epsilon decay schedule affects the exploration-exploitation trade-off.

-   **Fast Decay (0.1)**: Quick transition to exploitation, worked well with optimal hyperparameters
-   **Slow Decay (0.5)**: Maintained exploration longer, but resulted in lower performance (3.0) as the agent didn't exploit learned knowledge effectively

**Conclusion**: A faster epsilon decay (0.1) is preferable when other hyperparameters are well-tuned, allowing the agent to exploit learned strategies earlier.

### Best Configuration

**Winning Hyperparameters** (Set 5 - Large Batch):

```
Learning Rate: 1e-4
Gamma: 0.99
Batch Size: 128
Epsilon Start: 1.0
Epsilon End: 0.05
Epsilon Decay: 0.1
Average Reward: 7.0
```

This configuration achieved the highest average reward (7.0) by combining:

-   Stable learning rate (1e-4)
-   Long-term planning (gamma=0.99)
-   Stable gradient updates (batch_size=128)
-   Appropriate exploration schedule (epsilon_decay=0.1)

### Training Insights

1. **Stability Over Speed**: Slower, more stable learning (lower LR, larger batches) outperformed faster but unstable configurations.

2. **Long-term Planning Matters**: High gamma (0.99) was crucial for Breakout, where successful gameplay requires planning several steps ahead.

3. **Batch Size is Critical**: Increasing batch size from 32 to 128 improved performance by 30%, demonstrating the importance of stable gradient estimates.

4. **Exploration Balance**: Too much exploration (slow epsilon decay) can hurt performance once the agent has learned effective strategies.

## Video Demonstration

A video demonstration of the trained agent playing Breakout is available:

-   **Best Episode Video**: `video.mp4` (in root directory)
-   **All Episodes**: `videos/` folder contains recordings of all 10 evaluation episodes

The video shows the agent:

-   Successfully controlling the paddle
-   Hitting the ball consistently
-   Breaking bricks and scoring points
-   Demonstrating learned gameplay strategies

To generate a new video, run:

```bash
python play.py
```

## Team Members

1. **Reponse** - 5 hyperparameter experiments, play.py to visualize episodes in a video
2. **Kanisa Thiak** - 10 hyperparameter experiments (Sets 1-10) focused on epsilon schedules, gamma, and batch size trade-offs
3. _[Member 3 - To be filled]_
4. _[Member 4 - To be filled]_
5. _[Member 5 - To be filled]_

### Environment Details

-   **Observation Space**: 4 stacked grayscale frames (84x84 pixels each)
-   **Action Space**: Discrete (4 actions: NOOP, FIRE, RIGHT, LEFT)
-   **Reward**: +1 for each brick broken, +2 for completing a level
-   **Episode Termination**: When all lives are lost

## 🔗 References

-   [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
-   [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
-   [DQN Paper (Mnih et al., 2015)](https://arxiv.org/abs/1312.5602)

## 📄 License

This project is part of an academic assignment.

---

**Last Updated**: 2025
**Best Model Performance**: 7.0 average reward over 5 evaluation episodes
