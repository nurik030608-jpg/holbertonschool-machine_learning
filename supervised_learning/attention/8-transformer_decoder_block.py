#!/usr/bin/env python3
"""Module to create a decoder block for a transformer architecture."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Represents a single transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the DecoderBlock instance.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully connected layer.
            drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm=dm, h=h)
        self.mha2 = MultiHeadAttention(dm=dm, h=h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Forward pass for the decoder block.

        Args:
            x (tf.Tensor): Target input tensor of shape
                (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): Encoder output tensor of shape
                (batch, input_seq_len, dm).
            training (bool): Boolean to determine if the model is training.
            look_ahead_mask (tf.Tensor): Mask applied to first MHA.
            padding_mask (tf.Tensor): Mask applied to second MHA.

        Returns:
            tf.Tensor: Block output tensor of shape (batch, target_seq_len, dm).
        """
        # First MHA sub-layer (Masked Multi-Head Self-Attention)
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Second MHA sub-layer (Encoder-Decoder Attention)
        attn2, _ = self.mha2(out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward sub-layer
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layernorm3(out2 + ffn_output)

        return output
