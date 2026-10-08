#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm

    env: environment instance
    V: numpy.ndarray of shape (s,) with the value estimate
    policy: function that takes a state and returns the next action
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns: V, the updated value estimate
    """
    for i in range(episodes):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            state_next, reward, terminated, truncated, _ = env.step(action)
            episode.append([state, reward])
            if terminated or truncated:
                break
            state = state_next
        episode = np.array(episode, dtype=int)
        G = 0
        for step in episode[::-1]:
            G = step[1] + gamma * G
            if step[0] not in episode[:i, 0]:
                V[step[0]] += alpha * (G - V[step[0]])
    return V
