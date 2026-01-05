#!/usr/bin/env python3
"""
Module to create a DataFrame from a numpy ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels.
    """
    # Получаем количество колонок
    num_cols = array.shape[1]

    # Генерируем список букв A, B, C... используя ASCII коды
    # 65 — это код заглавной 'A'. 65 + 0 = 'A', 65 + 1 = 'B' и т.д.
    col_names = [chr(65 + i) for i in range(num_cols)]

    # Создаем и возвращаем DataFrame
    return pd.DataFrame(array, columns=col_names)
