#!/usr/bin/env python3
"""Модуль для обучения модели Keras с Early Stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Обучает модель с использованием Early Stopping при необходимости.

    Args:
        network: модель для обучения
        data: входные данные
        labels: метки (one-hot)
        batch_size: размер батча
        epochs: количество эпох
        validation_data: данные для валидации (X_val, y_val)
        early_stopping: использовать ли раннюю остановку
        patience: сколько эпох ждать улучшения перед остановкой
        verbose: вывод процесса
        shuffle: перемешивание данных

    Returns:
        Объект History
    """
    callbacks = []

    # Проверяем условия для Early Stopping
    if validation_data and early_stopping:
        # monitor='val_loss' следит за ошибкой на валидации
        # patience=patience определяет задержку остановки
        early_stop_cb = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop_cb)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
