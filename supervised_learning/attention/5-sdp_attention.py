#!/usr/bin/env python3
"""Module to calculate scaled dot product attention for transformers."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates scaled dot product attention.

    Args:
        Q (tf.Tensor): Query matrix of shape (..., seq_len_q, dk).
        K (tf.Tensor): Key matrix of shape (..., seq_len_v, dk).
        V (tf.Tensor): Value matrix of shape (..., seq_len_v, dv).
        mask (tf.Tensor, optional): Optional mask tensor. Defaults to None.

    Returns:
        tuple: (output, weights)
            - output: Tensor of shape (..., seq_len_q, dv).
            - weights: Tensor of shape (..., seq_len_q, seq_len_v).
    """
    # Get the last dimension dk from Q to compute scaling factor
    dk = tf.cast(tf.shape(Q)[-1], tf.float32)

    # MatMul Q and K transpose: (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale matmul_qk by the square root of dk
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided (-1e9 multiplied to mask and added)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Compute softmax weights along the last dimension (seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Compute final attention output by multiplying weights with V
    output = tf.matmul(weights, V)

    return output, weights
