#!/usr/bin/env python3
"""
Module to save and load a Keras model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a specific file.

    Args:
        network: The model to save.
        filename: The path of the file where the model should be saved.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a specific file.

    Args:
        filename: The path of the file from which the model should be loaded.

    Returns:
        The loaded model.
    """
    return K.models.load_model(filename)
