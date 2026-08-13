#!/usr/bin/env python3
"""Loads the FrozenLake environment from Gymnasium."""

import gymnasium as gym


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
    map_name = '8x8'

  if desc is not None:
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery)
  else:
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)

  return env
