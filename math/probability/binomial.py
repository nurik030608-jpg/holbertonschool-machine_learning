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

            p_initial = 1 - (variance / mean)
            self.n = int(round(mean / p_initial))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(num):
            res = 1
            for i in range(1, num + 1):
                res *= i
            return res

        n_fact = factorial(self.n)
        k_fact = factorial(k)
        nk_fact = factorial(self.n - k)

        combination = n_fact / (k_fact * nk_fact)
        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes."""
        k = int(k)
        if k < 0:
            return 0

        total = 0
        for i in range(k + 1):
            total += self.pmf(i)
        return total
