#!/usr/bin/env python3
"""
Defines a function that flips an image horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally using TensorFlow

    Args:
        image: a 3D tf.Tensor containing the image to flip

    Returns:
        The horizontally flipped image
    }
    """
    return tf.image.flip_left_right(image)
