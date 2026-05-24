#!/usr/bin/env python3
"""
Module to initialize variables for a Gaussian Mixture Model.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model (GMM).

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters (positive integer).

    Returns:
        pi (numpy.ndarray): Priors for each cluster, shape (k,), evenly init.
        m (numpy.ndarray): Centroid means for each cluster, shape (k, d).
        S (numpy.ndarray): Covariance matrices, shape (k, d, d), init as identity.
        Or None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    try:
        n, d = X.shape

        # 1. Априорные вероятности (priors) распределены равномерно: 1 / k
        pi = np.full(shape=(k,), fill_value=1 / k)

        # 2. Математические ожидания (means) инициализируются через K-means
        m, _ = kmeans(X, k)
        if m is None:
            return None, None, None

        # 3. Ковариационные матрицы (covariance) как единичные матрицы
        # Создаем одну матрицу (d, d) и размножаем её на k кластеров
        S = np.tile(np.eye(d), (k, 1, 1))

        return pi, m, S

    except Exception:
        return None, None, None
