#!/usr/bin/env python3
"""
Функция для свертки изображений с использованием нескольких фильтров.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Выполняет свертку над изображениями с использованием нескольких ядер.

    Аргументы:
        images: numpy.ndarray формы (m, h, w, c).
        kernels: numpy.ndarray формы (kh, kw, c, nc).
        padding: кортеж (ph, pw), 'same' или 'valid'.
        stride: кортеж (sh, sw).

    Возвращает:
        numpy.ndarray формы (m, output_h, output_w, nc).
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # 1. Определение параметров padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        ph, pw = padding

    # 2. Применяем padding
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # 3. Вычисляем размеры выходного изображения
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    # 4. Инициализируем выходной массив (теперь 4 измерения)
    convolved = np.zeros((m, output_h, output_w, nc))

    # 5. Свертка (три разрешенных цикла)
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):  # Цикл по количеству фильтров
                h_start = i * sh
                w_start = j * sw
                
                # Извлекаем фрагмент (m, kh, kw, c)
                slice_images = images_padded[:,
                                             h_start:h_start + kh,
                                             w_start:w_start + kw,
                                             :]
                
                # Умножаем фрагмент на k-й фильтр и суммируем по осям (1, 2, 3)
                # kernels[:, :, :, k] имеет форму (kh, kw, c)
                convolved[:, i, j, k] = np.sum(
                    slice_images * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved
