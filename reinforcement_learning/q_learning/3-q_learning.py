#!/usr/bin/env python3
"""Q-learning algorithm implementation for Gymnasium environments."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env,
    Q,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
  """Performs Q-learning training on a FrozenLake environment.

  Args:
      env: The FrozenLakeEnv instance.
      Q (np.ndarray): The initial Q-table.
      episodes (int): Total number of episodes to train over.
      max_steps (int): Maximum number of steps per episode.
      alpha (float): Learning rate.
      gamma (float): Discount factor.
      epsilon (float): Initial epsilon value for epsilon-greedy action selection.
      min_epsilon (float): Minimum threshold for epsilon decay.
      epsilon_decay (float): Exponential decay rate for epsilon.

  Returns:
      tuple: (Q, total_rewards)
          - Q: The updated Q-table.
          - total_rewards: A list containing the total reward per episode.
  """
  total_rewards = []
  init_epsilon = epsilon

  for episode in range(episodes):
    # Reset environment for new episode (Gymnasium returns state, info)
    state, _ = env.reset()
    episode_reward = 0

    for _ in range(max_steps):
      # Select action using epsilon-greedy strategy
      action = epsilon_greedy(Q, state, epsilon)

      # Take step in environment
      next_state, reward, terminated, truncated, _ = env.step(action)

      # Modify reward if agent falls into a hole
      if terminated and reward == 0:
        reward = -1

      # Update Q-table using the Q-learning formula
      best_next_q = np.max(Q[next_state])
      Q[state, action] = Q[state, action] + alpha * (
          reward + gamma * best_next_q - Q[state, action]
      )

      episode_reward += reward
      state = next_state

      # End episode if terminal state reached (hole or goal, or truncated)
      if terminated or truncated:
        break

    total_rewards.append(episode_reward)

    # Exponential decay of epsilon after each episode
    epsilon = min_epsilon + (init_epsilon - min_epsilon) * np.exp(
        -epsilon_decay * episode
    )

  return Q, total_rewards
