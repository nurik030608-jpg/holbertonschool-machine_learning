#!/usr/bin/env python3
"""Module containing the gensim_to_keras function."""


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: Trained gensim word2vec model.

    Returns:
        keras Embedding layer populated with weights from the gensim model.
    """
    return model.wv.get_keras_embedding(trainable=True)
