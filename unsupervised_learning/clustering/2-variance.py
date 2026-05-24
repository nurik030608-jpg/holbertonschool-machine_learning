#!/usr/bin/env python3
"""
Module to calculate the total intra-cluster variance.
Provides a function to compute the sum of squared distances from each data point
to its closest centroid without using any loops.
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        C (numpy.ndarray): Centroid means for each cluster of shape (k, d).

    Returns:
        float: The total intra-cluster variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    try:
        # Проверяем, что размерность пространства 'd' совпадает у X и C
        if X.shape[1] != C.shape[1]:
            return None

        # Вычисляем квадраты расстояний между всеми точками и всеми центроидами.
        # X[:, None, :] имеет форму (n, 1, d)
        # C[None, :, :] имеет форму (1, k, d)
        # distances на выходе получает форму (n, k)
        distances = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)

        # Для каждой точки находим минимальное расстояние до ближайшего центроида
        min_distances = np.min(distances, axis=1)

        # Общая внутрикластерная дисперсия — это сумма этих минимальных расстояний
        total_variance = np.sum(min_distances)

        return float(total_variance)

    except Exception:
        return None
