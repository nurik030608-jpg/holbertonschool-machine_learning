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

    # Шаг 1: Инициализируем центроиды (1-й вызов numpy.random.uniform внутри initialize)
    C = initialize(X, k)
    if C is None:
        return None, None

    # Границы для повторной инициализации пустых кластеров
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    d = X.shape[1]

    # Разрешено максимум 2 цикла во всей функции.
    # Первый цикл (внешний) — итерации самого алгоритма K-means.
    for _ in range(iterations):
        C_old = C.copy()

        # Расширяем размерности для векторизованного вычисления расстояний без циклов.
        # X[:, None, :] имеет форму (n, 1, d)
        # C[None, :, :] имеет форму (1, k, d)
        # Разность возводится в квадрат и суммируется по последней оси (d)
        distances = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)

        # Каждой точке присваиваем индекс ближайшего центроида
        clss = np.argmin(distances, axis=0)

        # Второй цикл (внутренний) — обновление центроидов по кластерам.
        for cluster_idx in range(k):
            # Находим все точки, принадлежащие текущему кластеру
            points_in_cluster = X[clss == cluster_idx]

            if len(points_in_cluster) > 0:
                # Если кластер не пустой, считаем новое среднее значение
                C[cluster_idx] = np.mean(points_in_cluster, axis=0)
            else:
                # Если кластер пустой, реинициализируем его центроид.
                # Это ровно 2-й законный вызов функции numpy.random.uniform в коде.
                C[cluster_idx] = np.random.uniform(low, high, size=(1, d))

        # Если координаты центроидов не изменились, завершаем алгоритм досрочно
        if np.all(C_old == C):
            break

    return C, clss
