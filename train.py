"""
PPO Training Script for CarRacing-v3.

Key improvements over basic implementation:
- 4-frame stacking for velocity/motion perception
- Generalized Advantage Estimation (GAE)
- Advantage normalization
- Gradient clipping
- Negative reward early stopping (avoid wasting time off-track)
- Training log saved to CSV
- Best model checkpoint saved separately
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import argparse
from model import ActorCritic
from preprocess import FrameStack

# ======================== CONFIG ========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing-v3")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--k_epochs", type=int, default=10, help="PPO update epochs per episode")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--neg_reward_patience", type=int, default=100, help="Consecutive negative reward steps before early stop (only after 100 steps)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    return parser.parse_args()


# ======================== MAIN ========================

def main():
    args = parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    env = gym.make("CarRacing-v3")
    model = ActorCritic(input_channels=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    start_episode = 0
    best_reward = -float("inf")

    # Resume from checkpoint
    if args.resume and os.path.exists("car_racing_ppo.pth"):
        checkpoint = torch.load("car_racing_ppo.pth", map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_episode = checkpoint.get("episode", 0)
            best_reward = checkpoint.get("best_reward", -float("inf"))
            print(f"Resumed from episode {start_episode}, best reward: {int(best_reward)}")
        else:
            # Legacy format: just state_dict
            model.load_state_dict(checkpoint)
            print("Loaded legacy model (no optimizer state)")

    frame_stack = FrameStack(n=4)

    # ======================== HELPER FUNCTIONS ========================

    def select_action(state):
        """Sample action from policy and return action, log_prob, value."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        mean, std, value = model(state_t)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy()[0], log_prob.detach(), value.detach()

    def compute_gae(rewards, values, dones, gamma=args.gamma, lam=args.gae_lambda):
        """Compute Generalized Advantage Estimation."""
        # Ensure all values are 1-dim tensors
        vals = [v.view(1) if v.dim() == 0 else v.view(-1) for v in values]
        vals.append(torch.tensor([0.0]).to(DEVICE))  # terminal bootstrap

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * vals[t + 1].item() * (1 - dones[t]) - vals[t].item()
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        values_t = torch.cat(vals[:-1])
        returns = advantages + values_t
        return advantages, returns

    # ======================== TRAINING LOG ========================

    log_file = "training_log.csv"
    log_exists = os.path.exists(log_file) and args.resume
    csv_file = open(log_file, "a" if log_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not log_exists:
        csv_writer.writerow(["episode", "reward", "steps", "avg_loss"])

    # ======================== TRAINING LOOP ========================

    reward_history = []

    for episode in range(start_episode, start_episode + args.episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)

        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        total_reward = 0
        neg_reward_count = 0

        for step in range(args.max_steps):
            action, log_prob, value = select_action(state)

            # Clip actions to valid ranges
            action[0] = np.clip(action[0], -1, 1)    # steering
            action[1] = np.clip(action[1], 0, 1)      # gas
            action[2] = np.clip(action[2], 0, 1)      # brake

            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = frame_stack.step(next_obs)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(float(done or truncated))

            state = next_state
            total_reward += reward

            # Early stop: only after 100 steps and 100 consecutive negative rewards
            # This prevents cutting off episodes during the initial acceleration phase
            if reward < 0:
                neg_reward_count += 1
            else:
                neg_reward_count = 0

            if step > 100 and neg_reward_count >= args.neg_reward_patience:
                break

            if done or truncated:
                break

        # ==================== PPO UPDATE ====================

        advantages, returns = compute_gae(
            rewards,
            values,
            dones
        )

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = torch.stack(log_probs)
        total_loss = 0

        for _ in range(args.k_epochs):
            new_log_probs, new_values, entropies = [], [], []

            for s, a in zip(states, actions):
                s_t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
                a_t = torch.FloatTensor(a).to(DEVICE)

                mean, std, value = model(s_t)
                dist = torch.distributions.Normal(mean, std)

                new_log_probs.append(dist.log_prob(a_t).sum())
                new_values.append(value)
                entropies.append(dist.entropy().sum())

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.cat(new_values).squeeze()
            entropy = torch.stack(entropies).mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = mse_loss(new_values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / args.k_epochs
        reward_history.append(total_reward)

        # Rolling average
        recent = reward_history[-50:]
        avg_reward = sum(recent) / len(recent)

        # ==================== SAVE ====================

        # Save latest checkpoint
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "episode": episode + 1,
            "best_reward": best_reward,
        }, "car_racing_ppo.pth")

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), "car_racing_best.pth")

        # Log to CSV
        csv_writer.writerow([episode + 1, int(total_reward), step + 1, f"{avg_loss:.4f}"])
        csv_file.flush()

        print(f"Ep {episode+1:4d} | Reward: {int(total_reward):5d} | "
              f"Steps: {step+1:4d} | Avg(50): {avg_reward:7.1f} | "
              f"Best: {int(best_reward):5d} | Loss: {avg_loss:.4f}")

    csv_file.close()
    env.close()
    print("\nTraining complete!")
    print(f"Best reward: {int(best_reward)}")


if __name__ == "__main__":
    main()
