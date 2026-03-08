#!/usr/bin/env python3
"""
Module that defines a single neuron performing binary classification
with private attributes and getters.
"""
import numpy as np


class Neuron:
    """
    Class Neuron that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Class constructor
        Args:
            nx: number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Private weights vector initialized using a random normal distribution
        self.__W = np.random.randn(1, nx)

        # Private bias initialized to 0
        self.__b = 0

        # Private activated output initialized to 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the private weights vector __W"""
        return self.__W

    @property
    def b(self):
        """Getter for the private bias __b"""
        return self.__b

    @property
    def A(self):
        """Getter for the private activated output __A"""
        return self.__A
