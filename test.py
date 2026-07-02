"""
Test script: Watch the trained AI agent play CarRacing-v3.
Uses 4-frame stacking to match training configuration.
"""

import gymnasium as gym
import torch
import numpy as np
import argparse
from model import ActorCritic
from preprocess import FrameStack

def parse_args():
    parser = argparse.ArgumentParser(description="Test trained PPO agent on CarRacing-v3")
    parser.add_argument("--model", type=str, default="car_racing_ppo.pth", help="Path to model weights")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make("CarRacing-v3", render_mode="human")

    model = ActorCritic(input_channels=4).to(DEVICE)

    # Support both new checkpoint format and legacy state_dict
    checkpoint = torch.load(args.model, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    frame_stack = FrameStack(n=4)
    all_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action_mean, _, _ = model(state_tensor)

            action = action_mean.cpu().numpy()[0]
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], 0, 1)
            action[2] = np.clip(action[2], 0, 1)

            obs, reward, done, truncated, _ = env.step(action)
            state = frame_stack.step(obs)
            total_reward += reward

            if done or truncated:
                break

        all_rewards.append(total_reward)
        print(f"Episode {ep+1}/{args.episodes} | Reward: {int(total_reward)}")

    env.close()
    print(f"\nAverage Reward over {args.episodes} episodes: {np.mean(all_rewards):.1f}")
    print(f"Best: {int(max(all_rewards))} | Worst: {int(min(all_rewards))}")


if __name__ == "__main__":
    main()
