#!/usr/bin/env python3
"""Module to sort a DataFrame by the High column"""


def high(df):
    """
    Sorts the DataFrame by the High price in descending order.

    Args:
        df: pd.DataFrame to be sorted.

    Returns:
        The sorted pd.DataFrame.
    """
    # Сортируем по столбцу 'High' в порядке убывания
    df = df.sort_values(by='High', ascending=False)

    return df
