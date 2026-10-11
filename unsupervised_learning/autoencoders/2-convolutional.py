#!/usr/bin/env python3
"""Module that creates a convolutional autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (h, w, c).
        filters (list): Number of filters for each conv layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
    """
    # --- Encoder ---
    input_enc = keras.Input(shape=input_dims)
    x = input_enc
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(units=latent_dims[0], activation='relu')(x)
    encoder = keras.Model(inputs=input_enc, outputs=latent, name='encoder')

    # --- Decoder ---
    d_x = input_enc
    for f in filters:
        d_x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(d_x)
        d_x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(d_x)
    shape_before_flatten = keras.backend.int_shape(d_x)[1:]
    flatten_size = shape_before_flatten[0] * shape_before_flatten[1] * \
        shape_before_flatten[2]

    input_dec = keras.Input(shape=latent_dims)
    x_dec = keras.layers.Dense(
        units=flatten_size,
        activation='relu'
    )(input_dec)
    x_dec = keras.layers.Reshape(target_shape=shape_before_flatten)(x_dec)

    reversed_filters = filters[::-1]
    num_layers = len(reversed_filters)

    for i, f in enumerate(reversed_filters):
        if i == num_layers - 2:
            x_dec = keras.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                padding='valid',
                activation='relu'
            )(x_dec)
        elif i == num_layers - 1:
            x_dec = keras.layers.Conv2D(
                filters=input_dims[2],
                kernel_size=(3, 3),
                padding='same',
                activation='sigmoid'
            )(x_dec)
        else:
            x_dec = keras.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'
            )(x_dec)
            x_dec = keras.layers.UpSampling2D(size=(2, 2))(x_dec)

    decoder = keras.Model(inputs=input_dec, outputs=x_dec, name='decoder')

    # --- Full Autoencoder ---
    auto_output = decoder(encoder(input_enc))
    auto = keras.Model(inputs=input_enc, outputs=auto_output, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
