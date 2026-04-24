#!/usr/bin/env python3
"""
Функция для универсальной свертки полутоновых изображений.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Выполняет свертку над набором полутоновых изображений.

    Аргументы:
        images: numpy.ndarray формы (m, h, w) с изображениями.
        kernel: numpy.ndarray формы (kh, kw) с ядром.
        padding: кортеж (ph, pw), 'same' или 'valid'.
        stride: кортеж (sh, sw) с шагом свертки.

    Возвращает:
        numpy.ndarray, содержащий результат свертки.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # 1. Определяем размеры паддинга
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        # Для 'same' рассчитываем паддинг так, чтобы выход был h x w
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        ph, pw = padding

    # 2. Применяем паддинг к изображениям
    # Паддинг только по высоте (ось 1) и ширине (ось 2)
    img_pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                     mode='constant', constant_values=0)

    # 3. Вычисляем размеры выходного изображения
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w +
