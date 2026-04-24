#!/usr/bin/env python3
"""
Функция для универсальной свертки полутоновых изображений.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Выполняет свертку над набором полутоновых изображений.

    Аргументы:
        images: numpy.ndarray формы (m, h, w).
        kernel: numpy.ndarray формы (kh, kw).
        padding: кортеж (ph, pw), 'same' или 'valid'.
        stride: кортеж (sh, sw).

    Возвращает:
        numpy.ndarray с результатом свертки.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Определение параметров padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        # Формула для 'same' с учетом stride:
        # ph = ceil(((h - 1) * sh + kh - h) / 2)
        # Но стандартная реализация "same" обычно ориентируется на stride=1
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    else:
        ph, pw = padding

    # Применяем padding
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Вычисляем размеры выходного изображения
    # Формула: floor((size + 2*p - k) / s) + 1
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # Инициализация результата
    convolved = np.zeros((m, output_h, output_w))

    # Основной цикл свертки (итерируемся по координатам выхода)
    for i in range(output_
