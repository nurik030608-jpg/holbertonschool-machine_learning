#!/usr/bin/env python3
"""TD(lambda) algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm and returns V"""
    for _ in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            eligibility *= lambtha * gamma
            eligibility[state] += 1

            delta = reward + gamma * V[new_state] - V[state]
            V += alpha * delta * eligibility

            if terminated or truncated:
                break
            state = new_state

    return V
