#!/usr/bin/env python3
"""Module that defines a neural network with one hidden layer

for binary classification.
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions and cost.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple: (prediction, cost)
                - prediction: numpy.ndarray of shape (1, m) with binary labels.
                - cost: float representing the network cost.
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A1 (numpy.ndarray): Output of the hidden layer.
            A2 (numpy.ndarray): Predicted output.
            alpha (float, optional): Learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2

    def train(
        self, X, Y, iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
    ):
        """Train the neural network over a specified number of iterations.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            iterations (int, optional): Number of training iterations.
                Defaults to 5000.
            alpha (float, optional): Learning rate. Defaults to 0.05.
            verbose (bool, optional): Print training information.
                Defaults to True.
            graph (bool, optional): Plot training cost. Defaults to True.
            step (int, optional): Step size for logging and plotting.
                Defaults to 100.

        Returns:
            tuple: Evaluation of training data after training completes.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        # 0th iteration state before any training
        _, A2_init = self.forward_prop(X)
        cost_init = self.cost(Y, A2_init)
        if verbose:
            print(f"Cost after 0 iterations: {cost_init}")
        if graph:
            costs.append(cost_init)
            iters.append(0)

        for i in range(1, iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            if verbose or graph:
                if i % step == 0 or i == iterations:
                    _, current_A2 = self.forward_prop(X)
                    current_cost = self.cost(Y, current_A2)
                    if verbose:
                        print(f"Cost after {i} iterations: {current_cost}")
                    if graph:
                        costs.append(current_cost)
                        iters.append(i)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
