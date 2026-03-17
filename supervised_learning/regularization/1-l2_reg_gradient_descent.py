#!/usr/bin/env python3
"""
Updates weights using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of the outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers
    """
    m = Y.shape[1]
    # Initial backward pass trigger: dZ for the last layer (softmax)
    # dZ = A[L] - Y
    A_last = cache['A' + str(L)]
    dZ = A_last - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)
        
        # 1. Calculate Gradients
        # Standard gradient + L2 penalty term (lambtha / m * W)
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m * weights[W_key])
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        # 2. Prepare dZ for the next layer (moving backwards)
        if i > 1:
            W = weights[W_key]
            # Derivative of tanh: 1 - A^2
            dZ = np.matmul(W.T, dZ) * (1 - (A_prev ** 2))
            
        # 3. Update weights in place
        weights[W_key] = weights[W_key] - (alpha * dW)
        weights[b_key] = weights[b_key] - (alpha * db)
