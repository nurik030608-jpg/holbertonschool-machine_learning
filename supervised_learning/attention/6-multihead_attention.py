#!/usr/bin/env python3
"""Module to perform multi-head attention for transformers."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi-head attention for transformer architectures."""

    def __init__(self, dm, h):
        """Initialize the MultiHeadAttention instance.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def _split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth) and transpose.

        Args:
            x (tf.Tensor): Input tensor.
            batch_size (int): Current batch size.

        Returns:
            tf.Tensor: Transposed tensor of shape (batch, h, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Forward pass for multi-head attention.

        Args:
            Q (tf.Tensor): Query tensor of shape (batch, seq_len_q, dk).
            K (tf.Tensor): Key tensor of shape (batch, seq_len_v, dk).
            V (tf.Tensor): Value tensor of shape (batch, seq_len_v, dv).
            mask (tf.Tensor, optional): Mask tensor (always None here).

        Returns:
            tuple: (output, weights)
                - output: Tensor of shape (batch, seq_len_q, dm).
                - weights: Tensor of shape (batch, h, seq_len_q, seq_len_v).
        """
        batch_size = tf.shape(Q)[0]

        # Linear projections
        q = self.Wq(Q)  # (batch, seq_len_q, dm)
        k = self.Wk(K)  # (batch, seq_len_v, dm)
        v = self.Wv(V)  # (batch, seq_len_v, dm)

        # Split heads -> (batch, h, seq_len, depth)
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Scaled dot product attention across all heads
        scaled_attention, weights = sdp_attention(q, k, v, mask)

        # Transpose back and concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm)
        )

        # Final linear projection layer
        output = self.linear(concat_attention)

        return output, weights
