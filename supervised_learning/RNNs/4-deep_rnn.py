#!/usr/bin/env python3
"""Module that performs forward propagation for a deep RNN."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances of length l.
        X (numpy.ndarray): Input data of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden states of shape (l, m, h).

    Returns:
        tuple: (H, Y)
            - H: numpy.ndarray of shape (t + 1, l, m, h) containing all hidden states.
            - Y: numpy.ndarray of shape (t, m, o) containing all outputs.
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    outputs = []

    for step in range(t):
        x_current = X[step]
        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]
            h_next, y = cell.forward(h_prev, x_current)
            H[step + 1, layer] = h_next
            x_current = h_next
        outputs.append(y)

    Y = np.array(outputs)
    return H, Y
