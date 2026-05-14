#!/usr/bin/env python3
"""
Module to adjust the contrast of an image randomly
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Parameters:
    image: A 3D tf.Tensor representing the input image.
    lower: A float representing the lower bound of the contrast factor range.
    upper: A float representing the upper bound of the contrast factor range.

    Returns:
    The contrast-adjusted image 3D tf.Tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
