#!/usr/bin/env python3
"""Module that contains the bag_of_words function."""
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
        features (list): List of the features used for embeddings.
    """
    # Preprocess sentences: convert to lowercase and remove punctuation/s-possessives
    cleaned_sentences = []
    for sentence in sentences:
        # Convert to lowercase
        text = sentence.lower()
        # Remove possessives like 's
        text = re.sub(r"'s\b", "", text)
        # Extract words consisting of alphanumeric characters
        words = re.findall(r"\b\w+\b", text)
        cleaned_sentences.append(words)

    # Determine features list
    if vocab is None:
        features_set = set()
        for words in cleaned_sentences:
            features_set.update(words)
        features = sorted(list(features_set))
    else:
        features = vocab

    # Map each feature to its index for quick lookup
    feat_to_idx = {feat: idx for idx, feat in enumerate(features)}

    # Initialize the embeddings matrix with zeros
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Count occurrences of features in each sentence
    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feat_to_idx:
                embeddings[i, feat_to_idx[word]] += 1

    return embeddings, features
