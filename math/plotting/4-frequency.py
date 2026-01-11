#!/usr/bin/env python3
"""Модуль для построения гистограммы оценок"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Строит гистограмму распределения оценок студентов"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Корзины от 0 до 100 с шагом 10
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Явно задаем границы оси X и деления
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))

    plt.show()
    plt.show()
