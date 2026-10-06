#!/usr/bin/env python3
"""
Moojulii algorithmii Monte Carlo qabu.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Algorithmii Monte Carlo fayyadamuun value function hojjetaa.

    Args:
        env: Instansii naannoo (environment)
        V: Array numpy isa gosa (s,) qabu
        policy: Funkshinii haala fi tarkaanfii itti aanu kennu
        episodes: Baay'ina episoodii leenjii
        max_steps: Tarkaanfii guddaa episoodii tokko keessatti
        alpha: Reetii barnootaa (learning rate)
        gamma: Reetii gadi xiqqessuu (discount rate)

    Returns:
        V: Tilmaama gatii haaromfame (updated value estimate)
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))

            if terminated or truncated:
                break

            state = next_state

        G = 0
        visited_states = set()

        # Episoodii boodarraa gara jalqabaatti deebi'uun maallaqa/gatii hisaabuu
        for s, a, r in reversed(episode):
            G = gamma * G + r

            # First-visit Monte Carlo
            if s not in visited_states:
                visited_states.add(s)
                V[s] = V[s] + alpha * (G - V[s])

    return V
