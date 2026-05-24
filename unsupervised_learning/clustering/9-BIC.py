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

    # Инициализи
