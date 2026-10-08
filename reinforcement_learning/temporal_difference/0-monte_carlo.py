#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the first-visit Monte Carlo algorithm.

    env: environment instance
    V: numpy.ndarray of shape (s,) with the value estimate
    policy: function that takes a state and returns the next action
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns: V, the updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # generate an episode following the policy
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        # first-visit: return that follows the first occurrence of each state
        first_visit = {}
        for t, (s, _) in enumerate(episode):
            if s not in first_visit:
                first_visit[s] = t

        # compute returns backwards
        G = 0
        returns = [0] * len(episode)
        for t in range(len(episode) - 1, -1, -1):
            G = episode[t][1] + gamma * G
            returns[t] = G

        for s, t in first_visit.items():
            V[s] = V[s] + alpha * (returns[t] - V[s])

    return V
