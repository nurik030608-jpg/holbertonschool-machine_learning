#!/usr/bin/env python3
"""
Функция для выполнения валидной свертки на полутоновых изображениях.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Выполняет "valid" свертку над набором полутоновых изображений.

    Аргументы:
        images: numpy.ndarray формы (m, h, w) с изображениями.
        kernel: numpy.ndarray формы (kh, kw) с ядром свертки.

    Возвращает:
        numpy.ndarray, содержащий результат свертки.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Вычисление размеров выходного изображения (valid convolution)
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Инициализация массива для результата
    convolved = np.zeros((m, output_h, output_w))

    # Используем два цикла для прохода по высоте и ширине ядра/выхода
    for i in range(output_h):
        for j in range(output_w):
            # Извлекаем фрагмент из всех изображений сразу и умножаем на ядро
            # Складываем результат по осям высоты (1) и ширины (2) ядра
            slice_images = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(slice_images * kernel, axis=(1, 2))

    return convolved
