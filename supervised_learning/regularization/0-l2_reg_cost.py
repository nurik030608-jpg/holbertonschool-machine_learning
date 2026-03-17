#!/usr/bin/env python3
"""
Module to calculate the L2 regularization cost of a neural network.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: The cost of the network without L2 regularization.
        lambtha: The regularization parameter.
        weights: A dictionary of the weights and biases (numpy.ndarrays).
        L: The number of layers in the neural network.
        m: The number of data points used.

    Returns:
        The cost of the network accounting for L2 regularization.
    """
    l2_term = 0

    # Iterate through all layers from 1 to L
    for i in range(1, L + 1):
        # We only apply L2 regularization to the weights (W), not the biases (b)
        W = weights.get('W' + str(i))
        if W is not None:
            # The Frobenius norm squared is the sum of the squares of all elements
            l2_term += np.linalg.norm(W)**2

    # L2 Regularization formula: J_total = J_original + (lambda / 2m) * sum(W^2)
    l2_cost = cost + (lambtha / (2 * m)) * l2_term

    # Returning the value as a float/array consistent with the input cost type
    return l2_cost
