#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    labels is a one-hot numpy.ndarray of shape (m, classes)
    logits is a one-hot numpy.ndarray of shape (m, classes)
    Returns: a confusion numpy.ndarray of shape (classes, classes)
    """
    # Convert one-hot to indices
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)

    num_classes = labels.shape[1]

    # Initialize the confusion matrix with zeros
    confusion = np.zeros((num_classes, num_classes))

    # Fill the matrix: rows are true labels, columns are predicted labels
    for i in range(len(y_true)):
        confusion[y_true[i], y_pred[i]] += 1

    return confusion
