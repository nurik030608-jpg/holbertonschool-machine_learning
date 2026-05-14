#!/usr/bin/env python3
"""
Module to adjust the brightness of an image randomly
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Parameters:
    image: A 3D tf.Tensor containing the image to change.
    max_delta: A float representing the maximum amount the image
               should be brightened (or darkened). Must be non-negative.

    Returns:
    The altered image 3D tf.Tensor.
    """
    return tf.image.random_brightness(image, max_delta)
