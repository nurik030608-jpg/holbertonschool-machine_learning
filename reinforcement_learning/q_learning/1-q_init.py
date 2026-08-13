#!/usr/bin/env python3
"""Initializes the Q-table for reinforcement learning."""

import numpy as np


def q_init(env):
  """Initializes the Q-table as a numpy.ndarray of zeros.

  Args:
      env: The FrozenLakeEnv instance.

  Returns:
      A numpy.ndarray of zeros with shape (number of states, number of actions).
  """
  action_space_size = env.action_space.n
  state_space_size = env.observation_space.n

  return np.zeros((state_space_size, action_space_size))
