#!/usr/bin/env python3
"""
Defines a vanilla autoencoder model.
"""
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder network.

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    # --- Encoder ---
    inputs = tf.keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)

    latent = tf.keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = tf.keras.Model(inputs=inputs, outputs=latent)

    # --- Decoder ---
    latent_inputs = tf.keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)

    outputs = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs)

    # --- Autoencoder ---
    auto_outputs = decoder(encoder(inputs))
    auto = tf.keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
