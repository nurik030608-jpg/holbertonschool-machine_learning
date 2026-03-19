#!/usr/bin/env python3
"""Модуль для обучения модели Keras с валидацией"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Обучает модель и анализирует валидационные данные.

    Args:
        network: модель для обучения
        data: входные данные (numpy.ndarray)
        labels: метки (one-hot numpy.ndarray)
        batch_size: размер батча
        epochs: количество эпох
        validation_data: кортеж (val_data, val_labels) для валидации
        verbose: печатать ли лог в консоль
        shuffle: перемешивать ли данные каждую эпоху

    Returns:
        Объект History после обучения
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
