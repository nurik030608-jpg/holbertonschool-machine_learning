#!/usr/bin/env python3
"""Module that defines the Binomial distribution class."""


class Binomial:
    """Class that represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize the Binomial distribution."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Method of Moments estimations:
            # mean = n * p, variance = n * p * (1 - p)
            # p_initial = 1 - (variance / mean)
            p_initial = 1 - (variance / mean)
            self.n = int(round(mean / p_initial))
            self.p = float(mean / self.n)
