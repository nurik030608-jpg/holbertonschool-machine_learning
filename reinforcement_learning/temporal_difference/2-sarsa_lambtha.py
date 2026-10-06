#!/usr/bin/env python3
"""SARSA(lambda) algorithm"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Chooses an action using the epsilon-greedy policy"""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Performs SARSA(lambda) and returns the updated Q table"""
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)

            eligibility *= lambtha * gamma
            eligibility[state, action] += 1

            delta = (reward + gamma * Q[new_state, new_action]
                     - Q[state, action])
            Q += alpha * delta * eligibility

            if terminated or truncated:
                break
            state, action = new_state, new_action

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

    return Q
