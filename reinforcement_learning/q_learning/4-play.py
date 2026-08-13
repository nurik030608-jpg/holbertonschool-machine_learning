#!/usr/bin/env python3
"""Module to test a trained agent playing FrozenLake."""

import numpy as np


def play(env, Q, max_steps=100):
  """Has the trained agent play an episode using pure exploitation.

  Args:
      env: The FrozenLakeEnv instance (configured with render_mode="ansi").
      Q (np.ndarray): The trained Q-table containing state-action values.
      max_steps (int): The maximum number of steps allowed in the episode.

  Returns:
      tuple: (total_reward, rendered_outputs)
          - total_reward: Total accumulated reward for the episode.
          - rendered_outputs: List of board renders captured at each step.
  """
  rendered_outputs = []
  total_reward = 0

  state, _ = env.reset()

  # Capture initial state before any action
  board_state = env.render()
  rendered_outputs.append(board_state)

  for step in range(max_steps):
    # Pure exploitation: always pick action with highest Q-value
    action = np.argmax(Q[state])

    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    # Capture board state after step
    board_state = env.render()
    rendered_outputs.append(board_state)

    if terminated or truncated:
      break

  return total_reward, rendered_outputs
