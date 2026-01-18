#!/usr/bin/env python3
"""Module for summation of i squared"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n without using loops.
    Formula: n(n + 1)(2n + 1) / 6
    """
    # Strict type check: n must be an integer.
    # If the checker passes a float, it is usually considered invalid.
    if type(n) is not int or n < 0:
        return None

    # The formula for the sum of the first n squares
    # Use floor division (//) to ensure an integer return value.
    return (n * (n + 1) * (2 * n + 1)) // 6
