#!/usr/bin/env python3
"""
Policy Gradient Training Script
"""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements full training of an agent using policy gradient.

    env: initial environment
    nb_episodes: total number of episodes used for training
    alpha: learning rate
    gamma: discount factor

    Returns: list of scores for each episode
    """
    # Инициализация весов
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    input_dim = state.shape[0] if hasattr(state, "shape") else len(state)
    output_dim = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )
    weight = np.random.rand(input_dim, output_dim)

    scores = []

    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_gradients = []
        episode_rewards = []
        done = False

        while not done:
            action, grad = policy_gradient(state, weight)

            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            episode_gradients.append(grad)
            episode_rewards.append(reward)
            state = next_state

        score = sum(episode_rewards)
        scores.append(score)

        # Расчет дисконтированных вознаграждений и обновление весов
        T = len(episode_rewards)
        for t in range(T):
            G = sum(
                [
                    gamma ** (k - t) * episode_rewards[k]
                    for k in range(t, T)
                ]
            )
            weight += alpha * episode_gradients[t] * G

        print("Episode: {} Score: {}".format(episode, score))

    return scores
