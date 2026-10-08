#!/usr/bin/env python3
"""Module to calculate the intersection of binomial data and prior values."""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining data given probabilities."""
    factorial = np.math.factorial
    comb = factorial(n) / (factorial(x) * factorial(n - x))

    return comb * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining data with probabilities.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array containing hypothetical probabilities.
        Pr (numpy.ndarray): 1D array containing prior beliefs of P.

    Returns:
        numpy.ndarray: 1D array containing the intersection for each
        probability in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    L = likelihood(x, n, P)
    return L * Pr
