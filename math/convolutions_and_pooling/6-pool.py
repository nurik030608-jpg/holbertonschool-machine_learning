#!/usr/bin/env python3
"""
Функция для выполнения пулинга (Max или Average) на изображениях.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Выполняет пулинг над набором многоканальных изображений.

    Аргументы:
        images: numpy.ndarray формы (m, h, w, c).
        kernel_shape: кортеж (kh, kw) с размерами окна пулинга.
        stride: кортеж (sh, sw) с шагом.
        mode: строка 'max' или 'avg'.

    Возвращает:
        numpy.ndarray с результатом пулинга.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Вычисляем размеры выходного изображения
    # В пулинге обычно не используется padding, поэтому формула простая:
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Инициализируем массив для результата
    pooled = np.zeros((m, output_h, output_w, c))

    # Проходим циклами по пространственным координатам выхода
    for i in range(output_h):
        for j in range(output_w):
            h_start = i * sh
            w_start = j * sw
            
            # Извлекаем фрагмент (m, kh, kw, c) для всех фото и каналов
            slice_images = images[:,
                                  h_start:h_start + kh,
                                  w_start:w_start + kw,
                                  :]
            
            # Выполняем операцию в зависимости от режима
            if mode == 'max':
                # Максимум по осям высоты (1) и ширины (2) окна
                pooled[:, i, j, :] = np.max(slice_images, axis=(1, 2))
            elif mode == 'avg':
                # Среднее по осям высоты (1) и ширины (2) окна
                pooled[:, i, j, :] = np.mean(slice_images, axis=(1, 2))

    return pooled
