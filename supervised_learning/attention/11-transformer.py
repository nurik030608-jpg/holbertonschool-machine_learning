#!/usr/bin/env python3
"""Module to create the complete Transformer network architecture."""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Represents the complete Transformer model (Encoder-Decoder stack)."""

    def __init__(
        self, N, dm, h, hidden, input_vocab, target_vocab,
        max_seq_input, max_seq_target, drop_rate=0.1
    ):
        """Initialize the Transformer instance.

        Args:
            N (int): Number of blocks in encoder and decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in FFN layers.
            input_vocab (int): Size of the input vocabulary.
            target_vocab (int): Size of the target vocabulary.
            max_seq_input (int): Maximum sequence length for input.
            max_seq_target (int): Maximum sequence length for target.
            drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N=N, dm=dm, h=h, hidden=hidden,
            input_vocab=input_vocab, max_seq_len=max_seq_input,
            drop_rate=drop_rate
        )
        self.decoder = Decoder(
            N=N, dm=dm, h=h, hidden=hidden,
            target_vocab=target_vocab, max_seq_len=max_seq_target,
            drop_rate=drop_rate
        )
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(
        self, inputs, target, training,
        encoder_mask, look_ahead_mask, decoder_mask
    ):
        """Forward pass for the complete Transformer network.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, input_seq_len).
            target (tf.Tensor): Target tensor of shape (batch, target_seq_len).
            training (bool): Boolean to determine if the model is training.
            encoder_mask (tf.Tensor): Padding mask for the encoder.
            look_ahead_mask (tf.Tensor): Look-ahead mask for decoder's 1st MHA.
            decoder_mask (tf.Tensor): Padding mask for decoder's 2nd MHA.

        Returns:
            tf.Tensor: Transformer output tensor of shape
                (batch, target_seq_len, target_vocab).
        """
        # Pass inputs through the encoder stack
        encoder_output = self.encoder(inputs, training, encoder_mask)

        # Pass target and encoder output through the decoder stack
        decoder_output = self.decoder(
            target, encoder_output, training,
            look_ahead_mask, decoder_mask
        )

        # Map decoder output to target vocabulary space via final linear layer
        output = self.linear(decoder_output)

        return output
