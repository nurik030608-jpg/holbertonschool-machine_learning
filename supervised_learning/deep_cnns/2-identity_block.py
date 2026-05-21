#!/usr/bin/env python3
"""
Функция для построения identity block архитектуры ResNet.
"""
import tensorflow as tf


def identity_block(A_prev, filters):
    """
    Строит identity block (блок идентичности) для ResNet.

    Аргументы:
        A_prev: тензор, выход из предыдущего слоя.
        filters: список/кортеж с количеством фильтров [F11, F3, F12]:
            F11: число фильтров в первой свертке 1x1.
            F3: число фильтров в свертке 3x3.
            F12: число фильтров во второй свертке 1x1.

    Возвращает:
        Тензор, активированный выход блока идентичности.
    """
    F11, F3, F12 = filters

    # Инициализатор весов He normal с фиксированным seed=0
    he_normal = tf.keras.initializers.HeNormal(seed=0)

    # --- ПЕРВЫЙ КОМПОНЕНТ ОСНОВНОГО ПУТИ ---
    # Свертка 1х1 для уменьшения размерности (Bottleneck)
    X = tf.keras.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=he_normal
    )(A_prev)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # --- ВТОРОЙ КОМПОНЕНТ ОСНОВНОГО ПУТИ ---
    # Свертка 3х3 (основное извлечение признаков)
    X = tf.keras.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',  # 'same' чтобы сохранить пространственные размеры
        kernel_initializer=he_normal
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # --- ТРЕТИЙ КОМПОНЕНТ ОСНОВНОГО ПУТИ ---
    # Свертка 1х1 для восстановления (увеличения) размерности каналов
    X = tf.keras.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=he_normal
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    # --- ДОБАВЛЕНИЕ Shortcut (Skip Connection) ---
    # В identity block мы просто складываем вход и выход без трансформаций
    X = tf.keras.layers.Add()([X, A_prev])
    X = tf.keras.layers.Activation('relu')(X)

    return X
