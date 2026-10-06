#!/usr/bin/env python3
"""
Module that contains the train function for policy gradient RL.
"""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains a policy gradient agent in the given environment.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Returns:
        scores: list of all scores per episode
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state = env.reset()
        episode_rewards = []
        episode_gradients = []

        while True:
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)

            episode_rewards.append(reward)
            episode_gradients.append(grad)

            if done:
                break

            state = next_state

        score = sum(episode_rewards)
        scores.append(score)

        # Вычисление дисконтированных наград и обновление весов
        for i in range(len(episode_rewards)):
            G = sum([r * (gamma ** idx)
                     for idx, r in enumerate(episode_rewards[i:])])
            weight += alpha * episode_gradients[i] * G

        print(f"Episode: {episode} Score: {score}")

    return scores
