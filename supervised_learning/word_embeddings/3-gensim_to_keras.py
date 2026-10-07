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
    # Получаем матрицу весов из gensim модели
    weights = model.wv.vectors

    # Создаём обучаемый слой Embedding с полученными весами
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding_layer
