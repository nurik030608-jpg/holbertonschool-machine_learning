#!/usr/bin/env python3
"""
Module for backward propagation over a pooling layer
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer of a neural network
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Создаем пустой массив такой же формы, как вход A_prev
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):  # По примерам в батче
        for h in range(h_new):  # По высоте выхода
            for w in range(w_new):  # По ширине выхода
                for f in range(c):  # По каналам
                    # 1. Находим границы окна (как в forward)
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        # Вырезаем фрагмент из ОРИГИНАЛЬНОГО входа
                        a_slice = A_prev[i, h_start:h_end, w_start:w_end, f]
                        
                        # Создаем маску: где был максимум? (True/False)
                        mask = (a_slice == np.max(a_slice))
                        
                        # Ошибка из dA идет только туда, где True
                        dA_prev[i, h_start:h_end, w_start:w_end, f] += \
                            mask * dA[i, h, w, f]

                    elif mode == 'avg':
                        # Ошибка делится поровну на все пиксели окна
                        avg_gradient = dA[i, h, w, f] / (kh * kw)
                        
                        # Создаем "заплатку" из одинаковых значений
                        shape = (kh, kw)
                        dist = np.ones(shape) * avg_gradient
                        
                        # Добавляем эту заплатку к градиенту входа
                        dA_prev[i, h_start:h_end, w_start:w_end, f] += dist

    return dA_prev
