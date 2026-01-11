#!/usr/bin/env python3
"""Модуль для сравнения распада двух радиоактивных элементов"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """Строит графики распада C-14 и Ra-226 на одном полотне"""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Рисуем графики с указанием цвета, стиля линии и метки для легенды
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Настройка осей и заголовка
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    # Установка диапазонов осей
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Добавление легенды в верхний правый угол
    plt.legend(loc='upper right')

    plt.show()
