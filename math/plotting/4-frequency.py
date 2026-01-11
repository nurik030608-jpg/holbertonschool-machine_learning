#!/usr/bin/env python3
"""Модуль для построения гистограммы оценок"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Строит гистограмму распределения оценок студентов"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Определяем корзины каждые 10 единиц: [0, 10, 20, ..., 100]
    bins = np.arange(0, 101, 10)

    # Строим гистограмму с черной обводкой (edgecolor)
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Настраиваем подписи и заголовок
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Устанавливаем деления на оси X согласно корзинам
    plt.xticks(bins)

    plt.show()
