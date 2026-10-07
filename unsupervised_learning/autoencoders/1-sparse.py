#!/usr/bin/env python3
"""
Defines a sparse autoencoder model.
"""
import tensorflow.keras as keras


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
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    l1_reg = keras.regularizers.l1(lambtha)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=l1_reg
    )(x)

    encoder = keras.Model(inputs=inputs, outputs=latent)

    # --- Decoder ---
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # --- Autoencoder ---
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
