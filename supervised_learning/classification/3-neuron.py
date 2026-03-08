#!/usr/bin/env python3
import numpy as np

class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        nx is the number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights, bias, and activated output
        self.__W = np.random.randn(1, nx)
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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        
        Y: numpy.ndarray with shape (1, m) containing the correct labels.
        A: numpy.ndarray with shape (1, m) containing the activated outputs.
        
        Returns the cost.
        """
        m = Y.shape[1]
        
        # We use 1.0000001 - A to avoid division by zero/log(0) errors
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        
        return cost
