#!/usr/bin/env python3
"""
Module to calculate the Expectation step in the EM algorithm for a GMM.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        pi (numpy.ndarray): Priors for each cluster of shape (k,).
        m (numpy.ndarray): Centroid means for each cluster of shape (k, d).
        S (numpy.ndarray): Covariance matrices for each cluster of shape
                           (k, d, d).

    Returns:
        g (numpy.ndarray): Posterior probabilities for each data point in each
                           cluster of shape (k, n).
        l (float): The total log likelihood of the model.
        Or None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    # Проверка, что сумма априорных вероятностей близка к 1
    if not np.isclose(np.sum(pi), 1.0):
        return None, None

    try:
        # Инициализируем матрицу для совместной вероятности (priors * pdf) формы (k, n)
        probabilities = np.zeros((k, n))

        # Разрешено использовать максимум 1 цикл (по количеству кластеров k)
        for cluster_idx in range(k):
            # Считаем функцию плотности вероятности (PDF) для конкретного кластера
            P = pdf(X, m[cluster_idx], S[cluster_idx])
            if P is None:
                return None, None
            
            # Совместная вероятность: априорная вероятность * PDF
            probabilities[cluster_idx] = pi[cluster_idx] * P

        # Знаменатель формулы Байеса (сумма совместных вероятностей по всем кластерам)
        # Получаем маргинальную вероятность для каждой точки, форма (n,)
        marginal_prob = np.sum(probabilities, axis=0)

        # Вычисляем апостериорные вероятности g (responsibilities) формы (k, n)
        g = probabilities / marginal_prob

        # Общий логарифм правдоподобия — это сумма логарифмов маргинальных вероятностей
        l = np.sum(np.log(marginal_prob))

        return g, float(l)

    except Exception:
        return None, None
