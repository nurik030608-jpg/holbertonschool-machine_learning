#!/usr/bin/env python3
"""Модуль для объединения пяти графиков в одну фигуру"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """Строит 5 графиков в сетке 3x2"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(6.4, 4.8))
    plt.suptitle('All in One')

    # 1. Line Graph
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(y0, 'r-')
    ax1.set_xlim(0, 10)

    # 2. Scatter Plot
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(x1, y1, c='m', s=5)
    ax2.set_title("Men's Height vs Weight", fontsize='x-small')
    ax2.set_xlabel('
