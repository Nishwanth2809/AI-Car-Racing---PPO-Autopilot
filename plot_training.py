import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training(log_path='training_log.csv', output_path='training_curve.png'):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    # Load data
    df = pd.read_csv(log_path)
    
    # Calculate rolling average for better visualization
    df['avg_reward_50'] = df['reward'].rolling(window=50, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards with transparency
    plt.plot(df['episode'], df['reward'], alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot rolling average
    plt.plot(df['episode'], df['avg_reward_50'], color='red', linewidth=2, label='Avg Reward (50 ep)')
    
    plt.title('PPO Training Progress - CarRacing-v3')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_training()
