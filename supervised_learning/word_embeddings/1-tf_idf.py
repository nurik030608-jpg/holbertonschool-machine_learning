#!/usr/bin/env python3
"""Module containing the tf_idf function."""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): List of vocabulary words to use.
            If None, all words within sentences are used.

    Returns:
        embeddings (numpy.ndarray): Shape (s, f) containing TF-IDF values.
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

    tf = np.zeros((s, f), dtype=float)
    df = np.zeros(f, dtype=float)

    # Calculate raw Term Frequency (TF) and Document Frequency (DF)
    for i, words in enumerate(cleaned_sentences):
        seen_in_sentence = set()
        for word in words:
            if word in feat_to_idx:
                idx = feat_to_idx[word]
                tf[i, idx] += 1
                seen_in_sentence.add(idx)

        for idx in seen_in_sentence:
            df[idx] += 1

    # Compute IDF using natural log: ln(N / DF)
    idf = np.zeros(f, dtype=float)
    nonzero_df = df > 0
    idf[nonzero_df] = np.log(s / df[nonzero_df])

    # Unnormalized TF-IDF
    tf_idf_matrix = tf * idf

    # L2 normalize embeddings row by row
    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    embeddings = tf_idf_matrix / norms

    # Convert features to numpy array for exact print formatting
    features = np.array(features)

    return embeddings, features
