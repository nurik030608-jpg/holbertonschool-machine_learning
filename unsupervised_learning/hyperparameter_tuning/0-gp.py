#!/usr/bin/env python3
"""Module that defines a noiseless 1D Gaussian process."""
import numpy as np


class GaussianProcess:
    """Class that represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix using RBF kernel.

        Args:
            X1 (numpy.ndarray): Matrix of shape (m, 1).
            X2 (numpy.ndarray): Matrix of shape (n, 1).

        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n).
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)
