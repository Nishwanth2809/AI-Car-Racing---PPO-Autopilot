"""
ActorCritic CNN model for PPO on CarRacing-v3.
Input: 4 stacked grayscale frames (4 x 84 x 84)
Output: action mean (3), action std (3), state value (1)
"""

import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )

        # Actor head: outputs mean of continuous action distribution
        self.actor_mean = nn.Linear(512, 3)
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(3))
        # Critic head: outputs estimated state value
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std)

        return mean, std, self.critic(x)
