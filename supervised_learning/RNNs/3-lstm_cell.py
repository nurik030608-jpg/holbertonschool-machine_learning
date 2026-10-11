#!/usr/bin/env python3
"""Module that represents a Long Short-Term Memory (LSTM) cell."""
import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory (LSTM) cell."""

    def __init__(self, i, h, o):
        """Initialize the LSTMCell instance.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            c_prev (numpy.ndarray): Previous cell state of shape (m, h).
            x_t (numpy.ndarray): Data input of shape (m, i).

        Returns:
            tuple: (h_next, c_next, y)
                - h_next: Next hidden state of shape (m, h).
                - c_next: Next cell state of shape (m, h).
                - y: Output of the cell of shape (m, o).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))

        # Update (input) gate
        u_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))

        # Intermediate cell state
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate
        o_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))

        # Next hidden state
        h_next = o_t * np.tanh(c_next)

        # Output with softmax activation
        logits = np.matmul(h_next, self.Wy) + self.by
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, c_next, y
