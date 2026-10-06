#!/usr/bin/env python3
"""Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm and returns V"""
    desc = env.unwrapped.desc.reshape(-1)

    for i in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            if desc[new_state] == b'H':
                reward = -1
            episode.append([state, action, reward])
            if terminated or truncated:
                break
            state = new_state

        episode = np.array(episode, dtype=int)
        G = 0
        for j, step in enumerate(episode[::-1]):
            state, action, reward = step
            G = gamma * G + reward
            if state not in episode[:i - j, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
