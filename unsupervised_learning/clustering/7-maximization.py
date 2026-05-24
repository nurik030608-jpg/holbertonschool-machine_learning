#!/usr/bin/env python3
"""
Module to calculate the Maximization step in the EM algorithm for a GMM.
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        g (numpy.ndarray): Posterior probabilities for each data point
                           in each cluster of shape (k, n).

    Returns:
        pi (numpy.ndarray): Updated priors for each cluster of shape (k,).
        m (numpy.ndarray): Updated centroid means for each cluster
                           of shape (k, d).
        S (numpy.ndarray): Updated covariance matrices for each cluster
                           of shape (k, d, d).
        Or None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    # Проверка согласованности количества точек данных
    if n != n_g:
        return None, None, None

    # Проверка валидности вероятностей (сумма по кластерам для точки близка к 1)
    if not np.isclose(np.sum(np.sum(g, axis=0)), n):
        return None, None, None

    try:
        # 1. Вычисляем мягкое количество точек в каждом кластере: shape (k,)
        N_k = np.sum(g, axis=1)

        # 2. Обновляем априорные вероятности (priors): shape (k,)
        pi = N_k / n

        # 3. Обновляем центроиды (means) без циклов: shape (k, d)
        # Умножаем g (k, n) на X (n, d) и делим на N_k, расширенный до (k, 1)
        m = np.matmul(g, X) / N_k[:, np.newaxis]

        # 4. Обновляем матрицы ковариации (S): shape (k, d, d)
        # Разрешено использовать максимум 1 цикл (по количеству кластеров k)
        S = np.zeros((k, d, d))
        for cluster_idx in range(k):
            # Центрируем данные относительно
