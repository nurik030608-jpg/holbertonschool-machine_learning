#!/usr/bin/env python3
"""
Module for backward propagation over a convolutional layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation over a convolutional layer
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # 1. Вычисляем padding, который был использован при forward
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # 2. Инициализируем градиенты
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Паддинг для входных данных и их градиента
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant')
    dA_padded = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                       mode='constant')

    # 3. Основной цикл по выходу
    for i in range(m):  # по примерам
        for h in range(h_new):  # по высоте выхода
            for w in range(w_new):  # по ширине выхода
                for f in range(c_new):  # по фильтрам
                    # Границы текущего окна
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    # Срез входных данных
                    a_slice = A_padded[i, h_start:h_end, w_start:w_end, :]

                    # Обновляем градиент по весам (dW)
                    dW[:, :, :, f] += a_slice * dZ[i, h, w, f]

                    # Обновляем градиент по входу (dA_padded)
                    dA_padded[i, h_start:h_end, w_start:w_end, :] += \
                        W[:, :, :, f] * dZ[i, h, w, f]

    # Убираем паддинг из dA_prev, чтобы вернуть исходный размер
    if padding == "same":
        dA_prev = dA_padded[:, ph:-ph if ph != 0 else None,
                            pw:-pw if pw != 0 else None, :]
    else:
        dA_prev = dA_padded

    return dA_prev, dW, db
