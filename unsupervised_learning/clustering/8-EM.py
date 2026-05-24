#!/usr/bin/env python3
"""
Module for Expectation Maximization in GMM.
Provides a function to perform the full EM algorithm for a Gaussian
Mixture Model with early stopping based on log-likelihood tolerance.
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters (positive integer).
        iterations (int): Maximum number of iterations (positive integer).
        tol (float): Tolerance for early stopping (non-negative float).
        verbose (bool): Whether to print information about the algorithm.

    Returns:
        pi, m, S, g, l, or None, None, None, None, None on failure.
        pi: priors for each cluster, shape (k,)
        m: centroid means for each cluster, shape (k, d)
        S: covariance matrices for each cluster, shape (k, d, d)
        g: probabilities for each data point in each cluster, shape (k, n)
        l: log likelihood of the model (float)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Инициализация параметров модели
    init_res = initialize(X, k)
    if init_res is None:
        return None, None, None, None, None
    pi, m, S = init_res

    # Начальное значение логарифма правдоподобия для отслеживания сходимости
    l_old = 0.0

    # Разрешено использовать максимум 1 цикл (для итераций EM-алгоритма)
    for i in range(iterations + 1):
        # Шаг Expectation (E-шаг): считаем скрытые переменные g и правдоподобие l
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Вывод логов каждые 10 итераций и на последней итерации цикла
        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        # Проверка условия ранней остановки (early stopping) по значению tol
        if i > 0 and abs(l - l_old) <= tol:
            # Если verbose=True, выводим лог финальной итерации при досрочной остановке,
            # если индекс итерации не совпал с шагом 10 (чтобы избежать дублирования).
            if verbose and (i % 10 != 0):
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(l, 5)))
            break

        # Если мы достигли лимита итераций без сходимости, завершаем работу алгоритма
        if i == iterations:
            break

        # Сохраняем текущее правдоподобие перед обновлением параметров
        l_old = l

        # Шаг Maximization (M-шаг): пересчитываем pi, m, S на основе новых весов g
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

    return pi, m, S, g, l
