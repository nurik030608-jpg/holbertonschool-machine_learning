#!/usr/bin/env python3
"""
Module for K-means clustering.
Provides a function to perform K-means clustering on a dataset
with handling for empty clusters and early stopping.
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate
    uniform distribution along each dimension.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        d = X.shape[1]
        low = np.min(X, axis=0)
        high = np.max(X, axis=0)
        centroids = np.random.uniform(low, high, size=(k, d))
        return centroids
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters (positive integer).
        iterations (int): Maximum number of iterations (positive integer).

    Returns:
        C (numpy.ndarray): Centroids of shape (k, d), or None on failure.
        clss (numpy.ndarray): Cluster indices of shape (n,), or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Шаг 1: Инициализируем центроиды
    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    d = X.shape[1]

    # Основной цикл итераций алгоритма
    for _ in range(iterations):
        C_old = C.copy()

        # Вычисляем квадраты расстояний между всеми точками и центроидами
        # Форма distances будет (n, k)
        distances = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)

        # ИСПРАВЛЕНО: берем axis=1, чтобы получить индекс ближайшего кластера
        # для каждой из n точек. Теперь clss имеет форму (n,)
        clss = np.argmin(distances, axis=1)

        # Цикл обновления центроидов
        for cluster_idx in range(k):
            points_in_cluster = X[clss == cluster_idx]

            if len(points_in_cluster) > 0:
                # Если в кластере есть точки, пересчитываем центр масс
                C[cluster_idx] = np.mean(points_in_cluster, axis=0)
            else:
                # Если кластер пустой, реинициализируем его (2-й вызов random.uniform)
                C[cluster_idx] = np.random.uniform(low, high, size=(1, d))

        # Если центроиды не изменились, досрочно выходим
        if np.all(C_old == C):
            break

    return C, clss
