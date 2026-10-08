#!/usr/bin/env python3
"""Module that defines the BayesianOptimization class."""
import numpy as np
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
            sigma_f (float): Standard deviation of the black-box function output.
            xsi (float): Exploration-exploitation factor for acquisition.
            minimize (bool): True for minimization, False for maximization.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
