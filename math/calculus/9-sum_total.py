#!/usr/bin/env python3
"""Module to calculate summation"""


def summation_i_squared(n):
    """
    Sum of squares from 1 to n
    """
    if type(n) is not int or n < 0:
        return None

    # Using formula: n(n + 1)(2n + 1) / 6
    return (n * (n + 1) * (2 * n + 1)) // 6
