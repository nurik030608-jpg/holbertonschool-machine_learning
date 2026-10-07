#!/usr/bin/env python3
"""
Модуль для создания сверточного автоэнкодера
"""
import tensorflow as tf


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.

    Args:
        input_dims: tuple of integers containing the dimensions of the model input
        filters: list containing the number of filters for each convolutional
                 layer in the encoder, respectively
        latent_dims: tuple of integers containing the dimensions of the latent
                     space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # -------------------
    # ENCODER
    # -------------------
    encoder_inputs = tf.keras.Input(shape=input_dims)
    x = encoder_inputs

    for f in filters:
        x = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder_outputs = x
    encoder = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_outputs, name='encoder')

    # -------------------
    # DECODER
    # -------------------
    decoder_inputs = tf.keras.Input(shape=latent_dims)
    x = decoder_inputs

    reversed_filters = filters[::-1]

    # Все свертки, кроме последних двух
    for f in reversed_filters[:-1]:
        x = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    # Предпоследняя свертка: padding='valid' с UpSampling2D
    x = tf.keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    # Последняя
