#!/usr/bin/env python3
import numpy as np

class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initializes the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data.
            
        Returns:
            The private attribute __A (the activated output).
        """
        # Linear transformation: Z = WX + b
        # Using np.matmul for vectorization across all m examples
        Z = np.matmul(self.__W, X) + self.__b
        
        # Sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))
        
        return self.__A
