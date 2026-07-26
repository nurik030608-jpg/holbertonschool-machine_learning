#!/usr/bin/env python3
"""Module that contains the RNNCell class."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Class constructor for RNNCell.

        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        x_t is a numpy.ndarray of shape (m, i) containing data input
        h_prev is a numpy.ndarray of shape (m, h) containing previous
        hidden state

        Returns: h_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear)
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
