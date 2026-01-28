#!/usr/bin/env python3
"""
Модуль для поэлементного сложения 2D матриц.
"""

def add_matrices2D(mat1, mat2):
  """
    Складывает две матрицы 2D.
    Возвращает новую матрицу или None, если размеры не совпадают.
    """
  if len(mat1) != len(mat2):
    return None
  if len(mat1[0]) != len(mat2[0]):
    return None
  return [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
        for i in range(len(mat1))
    ]
