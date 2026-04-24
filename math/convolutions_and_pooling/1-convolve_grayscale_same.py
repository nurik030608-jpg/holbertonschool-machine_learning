#!/usr/bin/env python3
"""
Функция для выполнения свертки 'same' на полутоновых изображениях.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Выполняет свертку 'same' над набором полутоновых изображений.

    Аргументы:
        images: numpy.ndarray формы (m, h, w) с изображениями.
        kernel: numpy.ndarray формы (kh, kw) с ядром свертки.

    Возвращает:
        numpy.ndarray, содержащий результат свертки.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Вычисляем необходимый padding для сохранения размерности (same)
    # Формула для padding: p = ceil((k - 1) / 2)
    pad_h = kh // 2
    pad_w = kw // 2

    # Добавляем нули по краям изображений (только для осей высоты и ширины)
    images_padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant', constant_values=0)

    # Инициализируем выходной массив той же размерности, что и входные картинки
    convolved = np.zeros((m, h, w))

    # Проходим циклами по высоте и ширине выходного изображения
    for i in range(h):
        for j in range(w):
            # Извлекаем область из дополненного изображения
            slice_images = images_padded[:, i:i + kh, j:j + kw]
            # Выполняем поэлементное умножение и суммируем по осям ядра
            convolved[:, i, j] = np.sum(slice_images * kernel, axis=(1, 2))

    return convolved
