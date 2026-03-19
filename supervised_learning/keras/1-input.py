#!/usr/bin/env python3
"""Module to build a neural network using Keras Functional API"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library (Functional API).
    Args:
        nx: number of input features
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    Returns:
        The keras model
    """
    # Since we can't use Input class, we use K.layers.Input
    inputs = K.layers.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    # Initialize the first hidden layer with the inputs
    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=reg
    )(inputs)

    # Loop through the rest of the layers
    for i in range(1, len(layers)):
        # Add Dropout before the next Dense layer
        x = K.layers.Dropout(1 - keep_prob)(x)

        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

    # Create the model by connecting inputs to the final output (x)
    model = K.Model(inputs=inputs, outputs=x)

    return model
