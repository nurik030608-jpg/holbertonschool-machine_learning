#!/usr/bin/env python3
"""
Module to save and load only the weights of a Keras model
"""


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a specific file.

    Args:
        network: The model whose weights should be saved.
        filename: The path of the file where the weights should be saved.
        save_format: The format in which the weights should be saved
                     (default is 'keras').

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights from a specific file.

    Args:
        network: The model to which the weights should be loaded.
        filename: The path of the file from which the weights should be loaded.

    Returns:
        None
    """
    network.load_weights(filename)
