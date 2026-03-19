#!/usr/bin/env python3
"""Модуль для обучения модели Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Обучает модель с использованием градиентного спуска по мини-батчам.

    Args:
        network: модель для обучения
        data: входные данные (numpy.ndarray) формы (m, nx)
        labels: метки (one-hot numpy.ndarray) формы (m, classes)
        batch_size: размер батча
        epochs: количество эпох
        verbose: печатать ли лог в консоль
        shuffle: перемешивать ли данные каждую эпоху

    Returns:
        Объект History после обучения
    """
    # Метод fit возвращает объект History, который содержит
    # данные о потерях (loss) и метриках на каждой эпохе.
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
