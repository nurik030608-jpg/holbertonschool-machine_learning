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

    if n != n_g:
        return None, None, None

    if not np.isclose(np.sum(np.sum(g, axis=0)), n):
        return None, None, None

    try:
        N_k = np.sum(g, axis=1)
        pi = N_k / n
        m = np.matmul(g, X) / N_k[:, np.newaxis]

        S = np.zeros((k, d, d))
        for cluster_idx in range(k):
            X_centered = X - m[cluster_idx]
            weighted_diff = g[cluster_idx, :, np.newaxis] * X_centered
            S[cluster_idx] = np.matmul(weighted_diff.T, X_centered) / N_k[cluster_idx]

        return pi, m, S

    except Exception:
        return None, None, None
