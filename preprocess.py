"""
Preprocessing utilities for CarRacing-v3 observations.
Converts RGB frames to grayscale 84x84 normalized arrays.
Includes FrameStack helper for stacking consecutive frames.
"""

import cv2
import numpy as np
from collections import deque

def preprocess(obs):
    """Convert a single RGB observation to grayscale 84x84 normalized float."""
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0


class FrameStack:
    """Maintains a stack of N preprocessed frames for temporal context."""

    def __init__(self, n=4):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, obs):
        """Initialize the stack by repeating the first frame N times."""
        frame = preprocess(obs)
        for _ in range(self.n):
            self.frames.append(frame)
        return self.get()

    def step(self, obs):
        """Push a new frame and return the current stack."""
        frame = preprocess(obs)
        self.frames.append(frame)
        return self.get()

    def get(self):
        """Return stacked frames as a numpy array of shape (N, 84, 84)."""
        return np.array(self.frames, dtype=np.float32)
