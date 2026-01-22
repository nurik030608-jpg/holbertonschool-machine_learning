#!/usr/bin/env python3
"""Модуль для транспонирования 2D матрицы"""


def matrix_transpose(matrix):

    rows = len(matrix)
    cols = len(matrix[0])
    transpose = [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    
    return transpose
