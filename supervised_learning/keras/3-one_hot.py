#!/usr/bin/env python3
"""Модуль для преобразования меток в One-Hot матрицу"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Преобразует вектор меток в матрицу One-Hot.

    Args:
        labels: вектор меток (numpy.ndarray или тензор)
        classes: количество классов (если None, берется максимум из labels)

    Returns:
        Матрица One-Hot (последняя размерность — количество классов)
    """
    # Используем встроенную функцию Keras для One-Hot кодирования
    return K.utils.to_categorical(labels, num_classes=classes)
