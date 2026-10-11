#!/usr/bin/env python3
"""Module that defines a neural network with one hidden layer

for binary classification.
"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer

    performing classification.
    """

    def __init__(self, nx, nodes):
        """Initialize the NeuralNetwork instance.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer private attributes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output layer private attributes
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Get the weights vector for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Get the bias for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Get the activated output for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Get the weights vector for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Get the bias for the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Get the activated output for the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (__A1, __A2)
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated outputs of shape (1, m).

        Returns:
            float: The calculated cost.
        """
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost
