#!/usr/bin/env python3
"""
Модуль, содержащий класс Neuron для бинарной классификации.
"""
import numpy as np


class Neuron:
    """
    Класс, описывающий один нейрон, выполняющий бинарную классификацию.
    """

    def __init__(self, nx):
        """
        Конструктор класса.

        Args:
            nx (int): Количество входных признаков (features) нейрона.

        Raises:
            TypeError: Если nx не является целым числом.
            ValueError: Если nx меньше 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # W: Вектор весов, инициализированный случайным нормальным распределением.
        # Используем размерность (1, nx).
        self.W = np.random.randn(1, nx)

        # b: Смещение (bias), инициализированное в 0.
        self.b = 0

        # A: Активированный выход (предсказание), инициализированный в 0.
        self.A = 0
