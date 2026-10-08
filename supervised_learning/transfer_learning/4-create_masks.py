#!/usr/bin/env python3
"""Module to create padding and look-ahead masks for Transformer training."""
import tensorflow as tf


def create_masks(inputs, target):
    """Creates encoder, combined, and decoder masks for training/validation.

    Args:
        inputs (tf.Tensor): Tensor of shape (batch_size, seq_len_in) containing
            the input sentence.
        target (tf.Tensor): Tensor of shape (batch_size, seq_len_out) containing
            the target sentence.

    Returns:
        tuple: (encoder_mask, combined_mask, decoder_mask)
            - encoder_mask: tf.Tensor padding mask of shape
              (batch_size, 1, 1, seq_len_in) for the encoder.
            - combined_mask: tf.Tensor of shape
              (batch_size, 1, seq_len_out, seq_len_out) combining look-ahead
              and target padding masks for the decoder's 1st attention block.
            - decoder_mask: tf.Tensor padding mask of shape
              (batch_size, 1, 1, seq_len_in) for the decoder's 2nd attention
              block.
    """
    # 1. Encoder padding mask (pads 0 values in inputs)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # 2. Decoder padding mask for the 2nd attention block (pads 0 in inputs)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # 3. Target padding mask for the 1st attention block in decoder
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # 4. Look-ahead mask to mask future tokens in target sequence
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )

    # Combined mask takes the maximum between look-ahead and target padding
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
