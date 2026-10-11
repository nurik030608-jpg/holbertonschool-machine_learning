#!/usr/bin/env python3
"""Module that defines the RNNDecoder class for machine translation."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Represents a decoder for machine translation using an RNN/GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNNDecoder instance.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units=units)

    def call(self, x, s_prev, hidden_states):
        """Forward pass for the decoder layer.

        Args:
            x (tf.Tensor): Tensor of shape (batch, 1) containing previous
                word in target sequence as a vocabulary index.
            s_prev (tf.Tensor): Tensor of shape (batch, units) containing
                previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape
                (batch, input_seq_len, units) containing encoder outputs.

        Returns:
            tuple: (y, s)
                - y: Tensor of shape (batch, vocab) containing output word.
                - s: Tensor of shape (batch, units) containing
                  new hidden state.
        """
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        context_exp = tf.expand_dims(context, 1)
        x = tf.concat([context_exp, x], axis=-1)

        outputs, s = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, s
