#!/usr/bin/env python3
"""Module that defines the BayesianOptimization class."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Class that performs Bayesian optimization on a 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize BayesianOptimization.

        Args:
            f: The black-box function to be optimized.
            X_init (numpy.ndarray): Matrix of shape (t, 1) representing initial
                sampled inputs.
            Y_init (numpy.ndarray): Matrix of shape (t, 1) representing initial
                sampled outputs.
            bounds (tuple): Tuple of (min, max) representing space bounds.
            ac_samples (int): Number of samples analyzed during acquisition.
            l (float): Length parameter for the kernel.
            sigma_f (float): Standard deviation of black-box function output.
            xsi (float): Exploration-exploitation factor for acquisition.
            minimize (bool): True for minimization, False for maximization.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using Expected Improvement.

        Returns:
            tuple: (X_next, EI)
                - X_next (numpy.ndarray): Array of shape (1,) representing the
                  next best sample point.
                - EI (numpy.ndarray): Array of shape (ac_samples,) containing
                  the expected improvement of each potential sample.
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            y_opt = np.min(self.gp.Y)
            improvement = y_opt - mu - self.xsi
        else:
            y_opt = np.max(self.gp.Y)
            improvement = mu - y_opt - self.xsi

        with np.errstate(divide='ignore'):
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei
