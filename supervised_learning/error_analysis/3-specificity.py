#!/usr/bin/env python3
"""
Module to calculate the specificity of a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    Args:
        confusion: numpy.ndarray of shape (classes, classes)
    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
    """
    # TP is the diagonal of the matrix
    tp = np.diag(confusion)

    # FP is the sum of the columns minus the true positives
    fp = np.sum(confusion, axis=0) - tp

    # FN is the sum of the rows minus the true positives
    fn = np.sum(confusion, axis=1) - tp

    # TN is the total sum minus (TP + FP + FN)
    total = np.sum(confusion)
    tn = total - (tp + fp + fn)

    # Specificity = TN / (TN + FP)
    spec = tn / (tn + fp)

    return spec
