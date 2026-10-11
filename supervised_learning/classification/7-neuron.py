#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the Neuron instance.

        Args:
            nx (int): Number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights vector."""
        return self.__W

    @property
    def b(self):
        """Get the bias term."""
        return self.__b

    @property
    def A(self):
        """Get the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: Activated output.
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

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
        """Evaluate the neuron's predictions and cost.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple: (prediction, cost)
                - prediction: numpy.ndarray of shape (1, m) with binary labels.
                - cost: float representing the network cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent

        on the neuron and update weights.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated output of shape (1, m).
            alpha (float, optional): Learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(
        self, X, Y, iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
    ):
        """Train the neuron over a specified number of iterations.

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
        A_init = self.forward_prop(X)
        cost_init = self.cost(Y, A_init)
        if verbose:
            print(f"Cost after 0 iterations: {cost_init}")
        if graph:
            costs.append(cost_init)
            iters.append(0)

        for i in range(1, iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if verbose or graph:
                if i % step == 0 or i == iterations:
                    current_cost = self.cost(Y, self.forward_prop(X))
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
