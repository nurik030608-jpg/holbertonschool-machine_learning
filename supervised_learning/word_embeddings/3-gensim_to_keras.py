#!/usr/bin/env python3
"""Module containing the gensim_to_keras function."""
import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: Trained gensim word2vec model.

    Returns:
        keras.layers.Embedding: Trainable Keras Embedding layer.
    """
    return model.wv.as_embedding(trainable=True)
