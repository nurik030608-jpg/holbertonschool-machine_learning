#!/usr/bin/env python3
"""Module that represents a gated recurrent unit (GRU) cell."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell."""

    def __init__(self, i, h, o):
        """Initialize the GRUCell instance.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            x_t (numpy.ndarray): Data input of shape (m, i).

        Returns:
            tuple: (h_next, y)
                - h_next: Next hidden state of shape (m, h).
                - y: Output of the cell of shape (m, o).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))

        # Reset gate
        r_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        # Intermediate hidden state
        concat_r = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output with softmax activation
        logits = np.matmul(h_next, self.Wy) + self.by
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
