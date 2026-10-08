#!/usr/bin/env python3
"""Module to perform the Monte Carlo algorithm for policy evaluation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm for policy evaluation.

    Args:
        env: Environment instance.
        V (numpy.ndarray): Array of shape (s,) containing the value estimate.
        policy: Function that takes in a state and returns the next action.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: V, the updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate an episode following the policy
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            if terminated or truncated:
                break
            state = next_state

        G = 0
        visited_states = set()

        # Process the episode in reverse (First-Visit Monte Carlo)
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if state not in visited_states:
                visited_states.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
