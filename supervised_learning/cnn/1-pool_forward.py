#!/usr/bin/env python3
"""
Module for forward propagation over a pooling layer
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # 1. Расчет размеров выхода (паддинга в пулинге обычно нет)
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # Выходной массив имеет столько же каналов, сколько и входной (c_prev)
    output = np.zeros((m, h_out, w_out, c_prev))

    # 2. Циклы по новой сетке (высота и ширина)
    for h in range(h_out):
        for w in range(w_out):
            h_start = h * sh
            h_end = h_start + kh
            w_start = w * sw
            w_end = w_start + kw

            # Вырезаем фрагмент сразу для всех примеров и всех каналов
            # Форма фрагмента: (m, kh, kw, c_prev)
            a_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            # 3. Применяем операцию в зависимости от режима
            if mode == 'max':
                # Выбираем максимум по осям высоты и ширины (1 и 2)
                output[:, h, w, :] = np.max(a_slice, axis=(1, 2))
            elif mode == 'avg':
                # Считаем среднее по осям высоты и ширины (1 и 2)
                output[:, h, w, :] = np.mean(a_slice, axis=(1, 2))

    return output
