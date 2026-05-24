#!/usr/bin/env python3
"""
Module to calculate the Probability Density Function (PDF)
of a multi-variate Gaussian distribution.
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X (numpy.ndarray): Data points of shape (n, d).
        m (numpy.ndarray): Mean of the distribution of shape (d,).
        S (numpy.ndarray): Covariance of the distribution of shape (d, d).

    Returns:
        numpy.ndarray: PDF values for each data point of shape (n,),
                       with a minimum value of 1e-300, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    try:
        # Вычисляем детерминант и обратную матрицу ковариации
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)

        # Если матрица вырожденная (детерминант равен или очень близок к 0)
        if det <= 0:
            return None

        # Центрируем данные: (X - m) -> shape (n, d)
        X_centered = X - m

        # Находим показатель экспоненты (квадратичную форму) без циклов:
        # (X - m) @ inv -> shape (n, d)
        # Умножаем поэлементно на (X - m) и суммируем по строкам (axis=1)
        # Получаем массив формы (n,)
        fac = np.sum((X_centered @ inv) * X_centered, axis=1)

        # Нормировочный коэффициент многомерного нормального распределения
        norm_coeff = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

        # Вычисляем итоговые значения плотности вероятности
        P = norm_coeff * np.exp(-0.5 * fac)

        # Устанавливаем минимальное пороговое значение 1e-300
        P = np.maximum(P, 1e-300)

        return P

    except Exception:
        return None
