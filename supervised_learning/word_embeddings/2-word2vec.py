#!/usr/bin/env python3
"""Module containing the word2vec_model function."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds and trains a gensim Word2Vec model.

    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences for training.
        window (int): Maximum distance between current and predicted word.
        negative (int): Size of negative sampling.
        cbow (bool): True for CBOW; False for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
