#!/usr/bin/env python3
"""
Module to save and load model weights
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a specific file
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a specific file
    """
    return K.models.load_model(filename)


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a specific file.

    Args:
        network: the model whose weights should be saved
        filename: the path of the file that the weights should be saved to
        save_format: the format in which the weights should be saved
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights from a specific file.

    Args:
        network: the model to which the weights should be loaded
        filename: the path of the file that the weights should be loaded from
    """
    network.load_weights(filename)
