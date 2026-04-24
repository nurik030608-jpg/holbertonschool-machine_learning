#!/usr/bin/env python3
"""
Функция для свертки многоканальных изображений.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Выполняет свертку над изображениями с несколькими каналами.

    Аргументы:
        images: numpy.ndarray формы (m, h, w, c).
        kernel: numpy.ndarray формы (kh, kw, c).
        padding: кортеж (ph, pw), 'same' или 'valid'.
        stride: кортеж (sh, sw).

    Возвращает:
        numpy.ndarray формы (m, output_h, output_w) с результатом свертки.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # 1. Определение параметров padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        ph, pw = padding

    # 2. Применяем padding (только к высоте и ширине)
    # Формат pad: ((m), (h), (w), (c))
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # 3. Вычисляем размеры выходного изображения
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    # 4. Инициализируем массив для результата
    convolved = np.zeros((m, output_h, output_w))

    # 5. Свертка (два цикла по пространственным координатам выхода)
    for i in range(output_h):
        for j in range(output_w):
            h_start = i * sh
            w_start = j * sw
            # Извлекаем фрагмент (m, kh, kw, c)
            slice_images = images_padded[:,
                                         h_start:h_start + kh,
                                         w_start:w_start + kw,
                                         :]
            # Поэлементное умножение и сумма по осям высоты, ширины и каналов
            # axis=(1, 2, 3) схлопывает (kh, kw, c) в одно число для каждого m
            convolved[:, i, j] = np.sum(slice_images * kernel, axis=(1, 2, 3))

    return convolved
