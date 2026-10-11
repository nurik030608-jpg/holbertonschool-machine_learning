#!/usr/bin/env python3
"""Module that defines convolutional generator and discriminator for faces."""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """Builds and returns a convolutional generator and discriminator."""

    def generator():
        """Constructs the convolutional generator network."""
        inputs = keras.Input(shape=(16,))
        hidden = keras.layers.Dense(2048, activation='tanh')(inputs)
        hidden = keras.layers.Reshape((2, 2, 512))(hidden)

        # Block 1
        hidden = keras.layers.UpSampling2D(size=(2, 2))(hidden)
        hidden = keras.layers.Conv2D(
            64, kernel_size=(3, 3), padding='same'
        )(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        # Block 2
        hidden = keras.layers.UpSampling2D(size=(2, 2))(hidden)
        hidden = keras.layers.Conv2D(
            16, kernel_size=(3, 3), padding='same'
        )(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        # Block 3 (Output)
        hidden = keras.layers.UpSampling2D(size=(2, 2))(hidden)
        hidden = keras.layers.Conv2D(
            1, kernel_size=(3, 3), padding='same'
        )(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        outputs = keras.layers.Activation('tanh')(hidden)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """Constructs the convolutional discriminator network."""
        inputs = keras.Input(shape=(16, 16, 1))

        # Block 1
        hidden = keras.layers.Conv2D(
            32, kernel_size=(3, 3), padding='same'
        )(inputs)
        hidden = keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        # Block 2
        hidden = keras.layers.Conv2D(
            64, kernel_size=(3, 3), padding='same'
        )(hidden)
        hidden = keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        # Block 3
        hidden = keras.layers.Conv2D(
            128, kernel_size=(3, 3), padding='same'
        )(hidden)
        hidden = keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        # Block 4
        hidden = keras.layers.Conv2D(
            256, kernel_size=(3, 3), padding='same'
        )(hidden)
        hidden = keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        # Flatten & Output Dense layer
        hidden = keras.layers.Flatten()(hidden)
        outputs = keras.layers.Dense(1, activation='tanh')(hidden)

        return keras.Model(inputs, outputs, name="discriminator")

    return generator(), get_discriminator()
