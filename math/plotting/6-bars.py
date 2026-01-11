#!/usr/bin/env python3
"""Модуль для построения стекового бар-графика количества фруктов"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Строит стековый график фруктов для Фарры, Фреда и Фелиции"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    width = 0.5

    # Инициализируем массив для хранения нижней границы текущего слоя
    bottom_val = np.zeros(3)

    # Строим слои фруктов в цикле
    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=width, bottom=bottom_val,
                color=colors[i], label=fruit_names[i])
        # Прибавляем текущий слой к 'фундаменту' для следующего слоя
        bottom_val += fruit[i]

    # Настройка осей и заголовка
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()

    plt.show()
