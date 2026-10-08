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
        sqdist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """Predicts the mean and variance of points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): Matrix of shape (s, 1) containing sample
                points.

        Returns:
            tuple: (mu, sigma)
                - mu (numpy.ndarray): Array of shape (s,) containing the mean
                  for each point in X_s.
                - sigma (numpy.ndarray): Array of shape (s,) containing the
                  variance for each point in X_s.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma
