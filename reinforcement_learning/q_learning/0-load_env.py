#!/usr/bin/env python3
"""Loads the FrozenLake environment from Gymnasium."""

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
  """Loads the pre-made FrozenLakeEnv environment from gymnasium.

  Args:
      desc: list of lists containing a custom description of the map, or None.
      map_name: string containing the pre-made map name to load, or None.
      is_slippery: boolean determining if the ice is slippery.

  Returns:
      The initialized gymnasium environment.
  """
  if desc is None and map_name is None:
    desc = generate_random_map(size=8)

  env = gym.make(
      "FrozenLake-v1", desc=desc, map_name=map_name, is_slippery=is_slippery
  )
  return env
