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
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Рассчитывает значение PMF для заданного количества "успехов".
        
        Args:
            k (int): Количество успехов.
            
        Returns:
            float: Значение PMF для k.
        """
        # Превращаем k в целое число
        k = int(k)
        
        # Если k отрицательное, вероятность такого события равна 0
        if k < 0:
            return 0
        
        # Число e (основание натурального логарифма)
        e = 2.7182818285
        
        # Считаем факториал k!
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
            
        # Формула: (lambtha^k * e^-lambtha) / k!
        pmf_value = (self.lambtha ** k * (e ** -self.lambtha)) / factorial
        
        return pmf_value
