#!/usr/bin/env python3
"""
Module to adjust the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Parameters:
    image: A 3D tf.Tensor containing the image to change.
    delta: A float representing the amount the hue should change.
           Must be in the interval [-0.5, 0.5].

    Returns:
    The altered image 3D tf.Tensor.
    """
    return tf.image.adjust_hue(image, delta)
