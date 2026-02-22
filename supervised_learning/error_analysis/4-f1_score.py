#!/usr/bin/env python3
"""
Module to calculate the F1 score of a confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix
    Args:
        confusion: numpy.ndarray of shape (classes, classes)
    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score
    """
    # Calculate Precision and Sensitivity (Recall) using existing modules
    prec = precision(confusion)
    sens = sensitivity(confusion)

    # F1 Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    f1 = 2 * (prec * sens) / (prec + sens)

    return f1
