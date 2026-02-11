#!/usr/bin/env python3
"""Модуль содержит класс Poisson для работы с распределением Пуассона."""


class Poisson:
    """Представляет распределение Пуассона."""

    def __init__(self, data=None, lambtha=1.):
        """
        Инициализация распределения.
        
        Args:
            data (list): Список данных для оценки распределения.
            lambtha (float): Ожидаемое количество событий.
        """
        if data is None:
            # Случай, когда данные не предоставлены
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Случай, когда предоставлены данные для расчета
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Рассчитываем лямбду как среднее арифметическое данных
            self.lambtha = float(sum(data) / len(data))
