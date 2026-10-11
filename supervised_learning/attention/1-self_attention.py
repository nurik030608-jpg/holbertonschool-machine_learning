#!/usr/bin/env python3
"""Module that defines the SelfAttention class for machine translation."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates attention for machine translation (Bahdanau attention)."""

    def __init__(self, units):
        """Initialize the SelfAttention instance.

        Args:
            units (int): Number of hidden units in the alignment model.
        """
        super(SelfAttention, self).__init__()
        self.Wa = tf.keras.layers.Dense(units=units)
        self.Ua = tf.keras.layers.Dense(units=units)
        self.Va = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Forward pass to calculate context and attention weights.

        Args:
            s_prev (tf.Tensor): Tensor of shape (batch, units) containing
                the previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape
                (batch, input_seq_len, units) containing encoder outputs.

        Returns:
            tuple: (context, weights)
                - context: Tensor of shape (batch, units).
                - weights: Tensor of shape (batch, input_seq_len, 1).
        """
        s_prev_exp = tf.expand_dims(s_prev, 1)

        score = self.Va(
            tf.nn.tanh(self.Wa(s_prev_exp) + self.Ua(hidden_states))
        )

        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
