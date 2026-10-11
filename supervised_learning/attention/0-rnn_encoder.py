#!/usr/bin/env python3
"""Module that defines the RNNEncoder class for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Represents an encoder for machine translation using an RNN/GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNNEncoder instance.

        Args:
            vocab (int): Size of the input vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            kernel_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of zeros.

        Returns:
            tf.Tensor: A tensor of shape (batch, units) containing zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Forward pass for the encoder layer.

        Args:
            x (tf.Tensor): Tensor of shape (batch, input_seq_len) containing word
                indices.
            initial (tf.Tensor): Tensor of shape (batch, units) containing the
                initial hidden state.

        Returns:
            tuple: (outputs, hidden)
                - outputs: Tensor of shape (batch, input_seq_len, units).
                - hidden: Tensor of shape (batch, units).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
