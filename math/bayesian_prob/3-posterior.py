#!/usr/bin/env python3
"""Module to calculate the posterior probability of binomial data."""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining data given probabilities."""
    factorial = np.math.factorial
    comb = factorial(n) / (factorial(x) * factorial(n - x))

    return comb * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining data with probabilities."""
    L = likelihood(x, n, P)
    return L * Pr


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining the data."""
    intersection_val = intersection(x, n, P, Pr)
    return np.sum(intersection_val)


def posterior(x, n, P, Pr):
    """Calculates the posterior probability for hypothetical probabilities.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array containing hypothetical probabilities.
        Pr (numpy.ndarray): 1D array containing prior beliefs of P.

    Returns:
        numpy.ndarray: 1D array containing the posterior probability of each
        probability in P given x and n.
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

    intersection_val = intersection(x, n, P, Pr)
    marginal_val = marginal(x, n, P, Pr)

    return intersection_val / marginal_val
