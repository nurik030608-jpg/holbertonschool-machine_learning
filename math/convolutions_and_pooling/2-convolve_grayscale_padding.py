#!/usr/bin/env python3
"""
Функция для выполнения свертки с кастомным паддингом.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Выполняет свертку над набором полутоновых изображений с заданным padding.

    Аргументы:
        images: numpy.ndarray формы (m, h, w) с изображениями.
        kernel: numpy.ndarray формы (kh, kw) с ядром свертки.
        padding: кортеж (ph, pw) с параметрами отступа.

    Возвращает:
        numpy.ndarray, содержащий результат свертки.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Применяем отступы (0 для количества картинок m, ph для высоты, pw для ширины)
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Вычисляем размеры выходного изображения после паддинга
    # Формула: H_out = H_in + 2*ph - kh + 1
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Инициализируем массив для результата
    convolved = np.zeros((m, output_h, output_w))

    # Проходим циклами по высоте и ширине выходного изображения
    for i in range(output_h):
        for j in range(output_w):
            # Извлекаем фрагмент из всех дополненных изображений сразу
            slice_images = images_padded[:, i:i + kh, j:j + kw]
            # Поэлементное умножение и сумма по осям ядра (1 и 2)
            convolved[:, i, j] = np.sum(slice_images * kernel, axis=(1, 2))

    return convolved
