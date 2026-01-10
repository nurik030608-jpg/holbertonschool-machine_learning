#!/usr/bin/env python3
"""Module to compute descriptive statistics for a DataFrame"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df: pd.DataFrame containing the data.

    Returns:
        A new pd.DataFrame containing the descriptive statistics.
    """
    # Удаляем столбец Timestamp и вычисляем статистику для остальных
    # Метод describe() считает count, mean, std, min, 25%, 50%, 75%, max
    stats = df.drop(columns=['Timestamp']).describe()

    return stats
