# AI Car Racing - PPO Autopilot 🏎️🤖

A state-of-the-art AI agent trained using **Proximal Policy Optimization (PPO)** to master the `CarRacing-v3` environment from Gymnasium. This project features a custom CNN architecture, frame stacking for temporal awareness, and a high-performance game companion with a real-time HUD.

![Demo Placeholder](https://via.placeholder.com/800x400.png?text=AI+Car+Racing+Demo)

## ✨ Features
- **PPO Implementation**: Robust reinforcement learning using Proximal Policy Optimization.
- **Frame Stacking**: Uses 4-frame stacking to provide the agent with velocity and momentum information.
- **Custom HUD**: A beautiful Pygame companion window with real-time telemetry, action bars, and reward tracking.
- **Hybrid Control**: Switch instantly between human control and AI autopilot using the `T` key.
- **Analytics**: Built-in training logger and visualization tools.

## 🏗️ Architecture
The agent uses an **Actor-Critic CNN** model:
- **CNN Backbone**: 3 convolutional layers for spatial feature extraction from 84x84 grayscale frames.
- **Actor Head**: Outputs a Normal distribution mean for continuous control (Steer, Gas, Brake).
- **Critic Head**: Estimates the state-value function for advantage calculation.
- **Temporal Context**: Stacked input layers (4 x 84 x 84).

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd AI-CAR-RACE

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Manual Control & AI Testing
Launch the companion mode to play or watch the AI:
```bash
python test_game_companion.py
```
**Controls:**
- `W / A / S / D`: Drive manually
- `T`: Toggle AI Autopilot 🤖
- `R`: Reset Episode
- `ESC`: Quit

### 3. Training
To retrain the model from scratch:
```bash
python train.py --episodes 800 --lr 1e-4
```

### 4. Progress Visualization
Generate the training curve:
```bash
python plot_training.py
```

## 📊 Results
The agent learns to navigate the track by maximizing the reward signal (staying on track and maintaining speed).
- **Environment**: CarRacing-v3
- **Action Space**: Continuous (3)
- **Reward**: ~900 for a perfect lap.

## 🛠️ Requirements
- `gymnasium[box2d]`
- `torch`
- `numpy`
- `pygame`
- `opencv-python`
- `matplotlib`
- `pandas`

---
*Created as part of an Advanced AI Reinforcement Learning Project.*
