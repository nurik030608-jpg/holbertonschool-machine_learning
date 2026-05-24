#!/usr/bin/env python3
"""
Module to find the best number of clusters for a GMM using BIC.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion (BIC).

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check (inclusive).
        kmax (int): Maximum number of clusters to check (inclusive).
        iterations (int): Maximum number of iterations for the EM algorithm.
        tol (float): Tolerance for the EM algorithm.
        verbose (bool): Whether to print information during EM execution.

    Returns:
        best_k (int): Best value for k based on its BIC.
        best_result (tuple): (pi, m, S) for the best number of clusters.
        l (numpy.ndarray): Log likelihood for each cluster size tested.
        b (numpy.ndarray): BIC value for each cluster size tested.
        Or None, None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmin > kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    results = []
    log_likelihoods = []
    bic_values = []

    for k in range(kmin, kmax + 1):
        res = expectation_maximization(X, k, iterations, tol, verbose)
        if res is None:
            return None, None, None, None

        pi, m, S, log_lh = res
        results.append((pi, m, S))

        # Если из EM приходит массив значений, берем последнее
        if isinstance(log_lh, (list, np.ndarray)):
            log_lh_value = log_lh[-1]
        else:
            log_lh_value = log_lh

        log_likelihoods.append(log_lh_value)

        # Вычисляем количество параметров p для модели GMM:
        # (k - 1) для априорных вероятностей pi
        # (k * d) для средних значений m
        # (k * d * (d + 1) / 2) для симметричных матриц ковариации S
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        # Формула расчета BIC
        bic = p * np.log(n) - 2 * log_lh_value
        bic_values.append(bic)

    l = np.array(log_likelihoods)
    b = np.array(bic_values)

    # Оптимальным считается k с наименьшим значением BIC
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
