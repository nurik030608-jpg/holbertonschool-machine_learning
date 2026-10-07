#!/usr/bin/env python3
"""
Defines a convolutional autoencoder model.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder network.

    Args:
        input_dims: tuple of integers containing dimensions of model input
        filters: list containing number of filters for each conv layer
                 in encoder
        latent_dims: tuple of integers containing dimensions of latent space

    Returns:
        encoder, decoder, auto
    """
    # --- Encoder ---
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = keras.Model(inputs=inputs, outputs=x)

    # --- Decoder ---
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    # All convolutions in decoder except the last two
    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Second to last convolution: uses valid padding and upsampling
    x = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Last convolution: same channels as input, sigmoid activation, no upsampling
    outputs = keras.
