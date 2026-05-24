#!/usr/bin/env python3
"""
Module to find the optimum number of clusters.
Determines the best k value by analyzing the variance across
different cluster sizes using K-means.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check (inclusive).
        kmax (int): Maximum number of clusters to check (inclusive).
        iterations (int): Maximum number of iterations for K-means.

    Returns:
        results (list): Outputs of K-means for each cluster size.
        d_vars (list): Difference in variance from the smallest cluster size.
        Or None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []
    d_vars = []

    try:
        # Разрешено максимум 2 цикла во всей функции.
        # Первый цикл: собираем результаты работы kmeans и считаем дисперсии
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            if C is None or clss is None:
                return None, None
            results.append((C, clss))

            var = variance(X, C)
            if var is None:
                return None, None
            variances.append(var)

        # Вытаскиваем значение дисперсии для самого маленького размера кластера
        base_variance = variances[0]

        # Второй цикл: вычисляем разницу в дисперсиях
        for var in variances:
            d_vars.append(base_variance - var)

        return results, d_vars

    except Exception:
        return None, None
