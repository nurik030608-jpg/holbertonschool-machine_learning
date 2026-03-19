#!/usr/bin/env python3
"""Модуль для обучения модели Keras с затуханием скорости обучения"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Обучает модель с использованием Early Stopping и Learning Rate Decay.
    """
    callbacks = []

    # Настройка Learning Rate Decay
    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """Вычисляет обратное затухание во времени (inverse time decay)"""
            return alpha / (1 + decay_rate * epoch)

        # LearningRateScheduler вызывает функцию в начале каждой эпохи
        # verbose=1 заставляет Keras печатать сообщение об обновлении LR
        lr_decay_cb = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay_cb)

    # Настройка Early Stopping
    if validation_data and early_stopping:
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
