#!/usr/bin/env python3
"""
Модуль для создания обычного (vanilla) автоэнкодера.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Создает энкодер, декодер и общую модель автоэнкодера.

    Параметры:
        input_dims (int): размерность входных данных
        hidden_layers (list): список с количеством нейронов для скрытых
                              слоев энкодера
        latent_dims (int): размерность скрытого пространства (латентного кода)

    Возвращает:
        encoder: модель энкодера
        decoder: модель декодера
        auto: скомпилированная модель автоэнкодера
    """
    # ------------------- ENCODER -------------------
    inputs = keras.Input(shape=(
