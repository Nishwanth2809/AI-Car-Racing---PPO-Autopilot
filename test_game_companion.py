"""
Game Companion: Play CarRacing-v3 yourself or let the AI drive.
Press T to toggle between HUMAN and AI mode.

Controls (Human mode):
  W / Up    - Gas
  S / Down  - Brake
  A / Left  - Steer left
  D / Right - Steer right
  SPACE     - Full brake
  T         - Toggle AI/Human
  R         - Reset episode
  ESC       - Quit
"""

import gymnasium as gym
import torch
import numpy as np
import pygame
from model import ActorCritic
from preprocess import FrameStack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================== PYGAME INIT ========================

pygame.init()
PANEL_W, PANEL_H = 320, 260
screen = pygame.display.set_mode((PANEL_W, PANEL_H))
pygame.display.set_caption("AI Car Racing - Companion")

# Fonts
font_large = pygame.font.SysFont("Consolas", 22, bold=True)
font_med = pygame.font.SysFont("Consolas", 16)
font_small = pygame.font.SysFont("Consolas", 13)

# Colors
BG_COLOR = (20, 20, 30)
ACCENT = (0, 200, 255)
AI_COLOR = (0, 255, 120)
HUMAN_COLOR = (255, 180, 50)
BAR_BG = (50, 50, 65)
TEXT_DIM = (140, 140, 160)
WHITE = (255, 255, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 80)

# ======================== LOAD ENV & MODEL ========================

env = gym.make("CarRacing-v3", render_mode="human")

model = ActorCritic(input_channels=4).to(DEVICE)
checkpoint = torch.load("car_racing_ppo.pth", map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)
model.eval()

frame_stack = FrameStack(n=4)

# ======================== STATE ========================

AI_MODE = False

def get_keyboard_action():
    keys = pygame.key.get_pressed()
    steer = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer = -1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer = 1.0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        gas = 1.0
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        brake = 0.8
    if keys[pygame.K_SPACE]:
        gas = 0.0
        brake = 1.0

    return np.array([steer, gas, brake], dtype=np.float32)


def draw_bar(surface, x, y, w, h, value, max_val, color, label):
    """Draw a labeled horizontal bar."""
    # Label
    lbl = font_small.render(label, True, TEXT_DIM)
    surface.blit(lbl, (x, y - 15))
    # Background
    pygame.draw.rect(surface, BAR_BG, (x, y, w, h), border_radius=3)
    # Fill
    fill_w = int(w * min(abs(value) / max_val, 1.0))
    if value < 0:
        pygame.draw.rect(surface, RED, (x + w // 2 - fill_w, y, fill_w, h), border_radius=3)
    else:
        pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=3)
    # Value text
    val_text = font_small.render(f"{value:+.2f}" if label == "STEER" else f"{value:.2f}", True, WHITE)
    surface.blit(val_text, (x + w + 8, y - 2))


def draw_hud(surface, action, total_reward, episode_num):
    """Draw the companion HUD panel."""
    surface.fill(BG_COLOR)

    # Header
    mode_color = AI_COLOR if AI_MODE else HUMAN_COLOR
    mode_text = "AI DRIVING" if AI_MODE else "HUMAN"
    header = font_large.render(mode_text, True, mode_color)
    surface.blit(header, (PANEL_W // 2 - header.get_width() // 2, 12))

    # Separator line
    pygame.draw.line(surface, mode_color, (20, 42), (PANEL_W - 20, 42), 2)

    # Action bars
    bar_x, bar_w, bar_h = 70, 150, 14
    draw_bar(surface, bar_x, 70, bar_w, bar_h, action[0], 1.0, ACCENT, "STEER")
    draw_bar(surface, bar_x, 105, bar_w, bar_h, action[1], 1.0, GREEN, "GAS")
    draw_bar(surface, bar_x, 140, bar_w, bar_h, action[2], 1.0, RED, "BRAKE")

    # Reward
    rew_label = font_med.render("Reward:", True, TEXT_DIM)
    rew_value = font_med.render(f"{int(total_reward)}", True, WHITE)
    surface.blit(rew_label, (20, 170))
    surface.blit(rew_value, (110, 170))

    # Episode
    ep_label = font_med.render("Episode:", True, TEXT_DIM)
    ep_value = font_med.render(f"{episode_num}", True, WHITE)
    surface.blit(ep_label, (180, 170))
    surface.blit(ep_value, (270, 170))

    # Controls help
    pygame.draw.line(surface, (50, 50, 65), (20, 200), (PANEL_W - 20, 200), 1)
    controls = "[T] Toggle AI  [R] Reset  [ESC] Quit"
    ctrl_text = font_small.render(controls, True, TEXT_DIM)
    surface.blit(ctrl_text, (PANEL_W // 2 - ctrl_text.get_width() // 2, 210))

    if not AI_MODE:
        keys_text = "WASD/Arrows: Drive  SPACE: Brake"
        keys_render = font_small.render(keys_text, True, TEXT_DIM)
        surface.blit(keys_render, (PANEL_W // 2 - keys_render.get_width() // 2, 232))


# ======================== MAIN LOOP ========================

obs, _ = env.reset()
state = frame_stack.reset(obs)

clock = pygame.time.Clock()
total_reward = 0
episode_num = 1
current_action = np.array([0.0, 0.0, 0.0])

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                AI_MODE = not AI_MODE
                print(f"Mode: {'AI' if AI_MODE else 'HUMAN'}")

            if event.key == pygame.K_r:
                obs, _ = env.reset()
                state = frame_stack.reset(obs)
                total_reward = 0
                episode_num += 1
                print(f"Reset! Episode {episode_num}")

            if event.key == pygame.K_ESCAPE:
                running = False

    if AI_MODE:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_mean, _, _ = model(state_tensor)
        current_action = action_mean.cpu().numpy()[0]
    else:
        current_action = get_keyboard_action()

    # Clip actions
    current_action[0] = np.clip(current_action[0], -1, 1)
    current_action[1] = np.clip(current_action[1], 0, 1)
    current_action[2] = np.clip(current_action[2], 0, 1)

    obs, reward, done, truncated, _ = env.step(current_action)
    state = frame_stack.step(obs)
    total_reward += reward

    # Draw HUD
    draw_hud(screen, current_action, total_reward, episode_num)
    pygame.display.flip()

    if done or truncated:
        print(f"Episode {episode_num} Reward: {int(total_reward)}")
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0
        episode_num += 1

    clock.tick(60)

env.close()
pygame.quit()
