#!/usr/bin/env python3
"""Модуль для настройки оптимизации модели Keras"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Настраивает оптимизацию Adam для модели Keras.

    Args:
        network: модель, которую нужно оптимизировать
        alpha: скорость обучения (learning rate)
        beta1: первый параметр оптимизации Adam
        beta2: второй параметр оптимизации Adam

    Returns:
        None
    """
    # Создаем экземпляр оптимизатора Adam с заданными параметрами
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    # Компилируем модель
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
