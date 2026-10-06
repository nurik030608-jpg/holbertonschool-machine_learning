#!/usr/bin/env python3
"""Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the first-visit Monte Carlo algorithm and returns V"""
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        states = [s for s, _ in episode]
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            s, r = episode[t]
            G = r + gamma * G
            if s not in states[:t]:
                V[s] += alpha * (G - V[s])

    return V
