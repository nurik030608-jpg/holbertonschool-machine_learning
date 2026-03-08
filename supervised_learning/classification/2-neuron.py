#!/usr/bin/env python3
#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class Neuron that defines a single neuron"""

    def __init__(self, nx):
        """Initializes the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m)
        """
        # Z = WX + b (Linear Transformation)
        # np.matmul handles the dot product across all m examples at once
        Z = np.matmul(self.__W, X) + self.__b

        # Sigmoid Activation: A = 1 / (1 + e^-Z)
        # np.exp is applied element-wise to the entire array
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
