<div align="center">
  <img src="https://via.placeholder.com/800x300/1a1a2e/ffffff?text=AI+Car+Racing+-+PPO+Autopilot" alt="AI Car Racing Banner" width="100%" />
  
  <h1>🏎️ AI Car Racing - PPO Autopilot 🤖</h1>
  
  <p>
    <strong>A state-of-the-art Deep Reinforcement Learning agent trained using Proximal Policy Optimization (PPO) to master the Gymnasium <code>CarRacing-v3</code> environment.</strong>
  </p>

  <p>
    <a href="https://pytorch.org/"><img src="https://img-shields.vercel.app/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch" /></a>
    <a href="https://gymnasium.farama.org/"><img src="https://img-shields.vercel.app/badge/Gymnasium-000000.svg?style=for-the-badge&logo=OpenAI&logoColor=white" alt="Gymnasium" /></a>
    <a href="https://www.pygame.org/"><img src="https://img-shields.vercel.app/badge/Pygame-2C9A14.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Pygame" /></a>
  </p>
</div>

---

## ✨ Key Features

- **🧠 PPO Implementation**: Highly robust reinforcement learning using the Actor-Critic architecture and Proximal Policy Optimization.
- **⏱️ Frame Stacking**: Stacks 4 consecutive grayscale frames to provide the agent with crucial temporal awareness (velocity and momentum).
- **🎛️ Custom HUD Companion**: A beautiful, real-time Pygame telemetry window featuring action bars (Steer, Gas, Brake) and reward tracking.
- **🎮 Hybrid Control Mode**: Seamlessly switch between manual human driving and AI autopilot on the fly using the `T` key.
- **📊 Built-in Analytics**: Automated training loggers and visualization tools to track learning progress over time.

---

## 🏗️ Neural Architecture

The agent leverages a custom **Actor-Critic CNN** model designed for visual continuous control:

| Component | Description |
| :--- | :--- |
| **Input State** | `4 x 84 x 84` Tensor (4 stacked grayscale frames). |
| **CNN Backbone** | 3 Convolutional layers for spatial feature extraction. |
| **Actor Head** | Outputs a Normal distribution mean for continuous control variables (Steering, Gas, Brake). |
| **Critic Head** | Estimates the State-Value function for advantage calculation. |

---

## 🚀 Getting Started

### 1️⃣ Installation

Clone the repository and set up your virtual environment:

```bash
# Clone the repository
git clone https://github.com/Nishwanth2809/AI-Car-Racing---PPO-Autopilot.git
cd AI-CAR-RACE

# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** If you run into issues with `Box2D`, ensure you have Swig installed: `pip install swig gymnasium[box2d]`

### 2️⃣ Play & Test (Companion Mode)

Launch the hybrid Pygame companion window to drive manually or watch the AI perform!

```bash
python test_game_companion.py
```

#### 🎮 Controls
| Key | Action |
| :---: | :--- |
| `W` / `A` / `S` / `D` | Drive the car manually |
| `SPACE` | Full Emergency Brake |
| `T` | **Toggle AI Autopilot** 🤖 |
| `R` | Reset the Episode |
| `ESC` | Quit |

### 3️⃣ Training the Agent

Want to train your own agent from scratch? Run the training script:

```bash
python train.py --episodes 800 --lr 1e-4
```
*The script will automatically save the best weights to `car_racing_best.pth`.*

### 4️⃣ Progress Visualization

Visualize the agent's learning curve and rewards:

```bash
python plot_training.py
```
*(This will generate and save a `training_curve.png` file).*

---

## 📊 Results & Performance

By maximizing the reward signal (staying near the track center and maintaining high speed), the PPO agent learns complex cornering and drifting behaviors.

- **Environment:** `CarRacing-v3`
- **Action Space:** Continuous (Steering, Gas, Brake)
- **Peak Reward:** ~900+ points for a perfect, flawless lap.

---

## 🛠️ Tech Stack & Requirements

- `gymnasium[box2d]`
- `torch` (PyTorch)
- `numpy`
- `pygame`
- `opencv-python`
- `matplotlib` & `pandas`

---
<div align="center">
  <i>Created as part of an Advanced AI Reinforcement Learning Project.</i>
</div>
