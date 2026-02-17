#!/usr/bin/env python3
"""
Module to calculate precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    """
    # TP — это элементы на главной диагонали
    tp = np.diag(confusion)

    # Для Precision суммируем по столбцам (axis=0)
    # Это общее количество предсказаний для каждого класса
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    res = tp / predicted_positives

    return res
