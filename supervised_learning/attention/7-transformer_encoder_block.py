#!/usr/bin/env python3
"""Module to create an encoder block for a transformer architecture."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Represents a single transformer encoder block with self-attention."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the EncoderBlock instance.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully connected layer.
            drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm=dm, h=h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """Forward pass for the encoder block.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len, dm).
            training (bool): Boolean to determine if the model is training.
            mask (tf.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            tf.Tensor: Block output tensor of shape (batch, input_seq_len, dm).
        """
        # Multi-head attention sub-layer
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward sub-layer
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)

        return output
