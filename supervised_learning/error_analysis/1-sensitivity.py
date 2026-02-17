#!/usr/bin/env python3
"""
Module to calculate sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    """
    tp = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    res = tp / actual_positives
    return res
