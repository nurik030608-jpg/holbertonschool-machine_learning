#!/usr/bin/env python3
"""
Defines a sparse autoencoder model.
"""
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder network.

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
        lambtha: regularization parameter used for L1 regularization on
                 the encoded output

    Returns:
        encoder, decoder, auto
    """
    # --- Encoder ---
    inputs = tf.keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = tf.keras.layers.Dense(
