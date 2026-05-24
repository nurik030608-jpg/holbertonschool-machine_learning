#!/usr/bin/env python3
"""
Module for K-means centroid initialization.
Provides a function to initialize cluster centroids using a multivariate
uniform distribution without any loops.
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate
    uniform distribution along each dimension.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters (positive integer)

    Returns:
        numpy.ndarray: Initialized centroids of shape (k, d),
                       or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    try:
        # Получаем количество измерений 'd'
        d = X.shape[1]

        # Находим минимальные и максимальные границы по каждой оси (axis=0)
        low = np.min(X, axis=0)
        high = np.max(X, axis=0)

        # Генерируем случайные точки ровно один раз через numpy.random.uniform.
        # Магия broadсasting автоматически растянет векторы low и high (shape (d,))
        # под итоговую матрицу центроидов нужного размера (shape (k, d)).
        centroids = np.random.uniform(low, high, size=(k, d))

        return centroids

    except Exception:
        return None


# Пример использования (можно запустить для проверки)
if __name__ == "__main__":
    # Создаем тестовый двумерный датасет (10 точек, 2 признака)
    # Например: первый признак лежит в районе 0-10, второй в районе 100-200
    np.random.seed(42)  # Фиксируем сид для воспроизводимости
    data = np.zeros((10, 2))
    data[:, 0] = np.random.uniform(0, 10, size=10)
    data[:, 1] = np.random.uniform(100, 200, size=10)

    print("--- Исходный датасет (первые 3 точки) ---")
    print(data[:3])

    # Инициализируем 3 центроида
    num_clusters = 3
    initialized_centroids = initialize(data, num_clusters)

    print(f"\n--- Сгенерированные центроиды ({num_clusters} кластера) ---")
    print(initialized_centroids)
