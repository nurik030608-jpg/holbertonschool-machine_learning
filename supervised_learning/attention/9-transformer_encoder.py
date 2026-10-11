#!/usr/bin/env python3
"""Module to create the full encoder stack for a transformer architecture."""
import tensorflow as tf
import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Represents the complete transformer encoder stack."""

    def __init__(
        self, N, dm, h, hidden, input_vocab,
        max_seq_len, drop_rate=0.1
    ):
        """Initialize the Encoder instance.

        Args:
            N (int): Number of blocks in the encoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in FFN.
            input_vocab (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length possible.
            drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab,
            output_dim=dm
        )
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm=dm, h=h, hidden=hidden, drop_rate=drop_rate)
            for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """Forward pass for the transformer encoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len).
            training (bool): Boolean to determine if the model is training.
            mask (tf.Tensor): Mask to be applied for multi-head attention.

        Returns:
            tf.Tensor: Encoder output of shape (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        # Embed input tokens and scale by square root of dm
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout layer
        x = self.dropout(x, training=training)

        # Pass through all N encoder blocks
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
