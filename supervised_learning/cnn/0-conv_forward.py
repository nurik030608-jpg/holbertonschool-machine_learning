#!/usr/bin/env python3
"""
Модуль для реализации прямого распространения сверточного слоя.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Осуществляет прямое распространение через сверточный слой.

    Аргументы:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - выход предыдущего слоя
        W: numpy.ndarray (kh, kw, c_prev, c_new) - веса (ядра)
        b: numpy.ndarray (1, 1, 1, c_new) - смещения
        activation: функция активации
        padding: строка "same" или "valid"
        stride: кортеж (sh, sw) - шаги свертки

    Возвращает:
        Z: выход сверточного слоя после активации
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # 1. Расчет Padding
    if padding == "same":
        # Формула для same: p = ((h_prev - 1) * s + kh - h_prev) / 2
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # 2. Применяем Padding к входным данным (только по осям H и W)
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant')

    # 3. Расчет размеров выходной матрицы
    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    # Инициализация выходного тензора
    output = np.zeros((m, h_out, w_out, c_new))

    # 4. Свертка (проход по всем измерениям)
    for i in range(h_out):
        for j in range(w_out):
            # Определяем границы "окна" во входной матри
