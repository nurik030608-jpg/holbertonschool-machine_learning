#!/usr/bin/env python3
"""Module to concatenate two DataFrames with specific indexing and keys"""
import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes dataframes and concatenates specific rows from df2 to df1.

    Args:
        df1: pd.DataFrame (coinbase)
        df2: pd.DataFrame (bitstamp)

    Returns:
        The concatenated pd.DataFrame with keys 'bitstamp' and 'coinbase'.
    """
    # 1. Устанавливаем Timestamp как индекс для обоих DataFrame
    df1 = index(df1)
    df2 = index(df2)

    # 2. Выбираем строки из df2 до метки 1417411920 включительно
    # Используем .loc, так как мы работаем с именованными индексами
    df2_sub = df2.loc[:1417411920]

    # 3. Объединяем: df2_sub идет первым (наверх), df1 вторым
    # Параметр keys добавит мультииндекс для идентификации источника строк
    result = pd.concat([df2_sub, df1], keys=['bitstamp', 'coinbase'])

    return result
