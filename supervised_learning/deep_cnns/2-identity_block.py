#!/usr/bin/env python3
"""Module that builds an identity block for a Deep Residual Network."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in ResNet (2015).

    Args:
        A_prev: tf.Tensor, output from the previous layer.
        filters: tuple or list containing (F11, F3, F12):
            - F11: number of filters in the first 1x1 convolution.
            - F3: number of filters in the 3x3 convolution.
            - F12: number of filters in the second 1x1 convolution.

    Returns:
        tf.Tensor, the activated output of the identity block.
    """
    F11, F3, F12 = filters

    initializer = K.initializers.HeNormal(seed=0)

    # Main Path - First Component (1x1 Conv)
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Main Path - Second Component (3x3 Conv)
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Main Path - Third Component (1x1 Conv)
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add Shortcut Connection (A_prev) to Main Path and Activation
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
