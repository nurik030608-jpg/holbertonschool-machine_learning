#!/usr/bin/env python3
"""Epsilon-greedy action selection for Q-learning."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
  """Uses epsilon-greedy strategy to determine the next action.

  Args:
      Q (np.ndarray): The Q-table containing Q-values for (state, action).
      state (int): The current state index.
      epsilon (float): The epsilon value (probability of exploration).

  Returns:
      int: The index of the selected action.
  """
  p = np.random.uniform(0, 1)

  if p < epsilon:
    # Exploration: pick a random action
    action = np.random.randint(0, Q.shape[1])
  else:
    # Exploitation: pick the action with the highest Q-value
    action = np.argmax(Q[state])

  return action
