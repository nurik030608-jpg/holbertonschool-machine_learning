#!/usr/bin/env python3
"""Module that builds a projection block for a Deep Residual Network."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in ResNet (2015).

    Args:
        A_prev: tf.Tensor, output from the previous layer.
        filters: tuple or list containing (F11, F3, F12):
            - F11: number of filters in the first 1x1 convolution.
            - F3: number of filters in the 3x3 convolution.
            - F12: number of filters in the second 1x1 convolution and
                   shortcut 1x1 convolution.
        s: int, stride of the first convolution in both the main path
           and shortcut connection.

    Returns:
        tf.Tensor, the activated output of the projection block.
    """
    F11, F3, F12 = filters

    initializer = K.initializers.HeNormal(seed=0)

    # Main Path - First Component (1x1 Conv with stride s)
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Main Path - Second Component (3x3 Conv with stride 1)
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Main Path - Third Component (1x1 Conv with stride 1)
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut Path (1x1 Conv with stride s)
    X_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Final Step: Add shortcut value to main path and pass through ReLU
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
