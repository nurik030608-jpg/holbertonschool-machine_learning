#!/usr/bin/env python3
"""Module containing the bag_of_words function."""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): List of vocabulary words to use.
            If None, all words within sentences are used.

    Returns:
        embeddings (numpy.ndarray): Shape (s, f) containing word counts.
        features (numpy.ndarray): Array of the features used for embeddings.
    """
    cleaned_sentences = []
    for sentence in sentences:
        text = sentence.lower()
        # Remove possessives like 's
        text = re.sub(r"'s\b", "", text)
        # Extract alphanumeric words
        words = re.findall(r"\b\w+\b", text)
        cleaned_sentences.append(words)

    if vocab is None:
        features_set = set()
        for words in cleaned_sentences:
            features_set.update(words)
        features = sorted(list(features_set))
    else:
        features = list(vocab)

    feat_to_idx = {feat: idx for idx, feat in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feat_to_idx:
                embeddings[i, feat_to_idx[word]] += 1

    # Convert features list to numpy array for expected string representation
    features = np.array(features)

    return embeddings, features
