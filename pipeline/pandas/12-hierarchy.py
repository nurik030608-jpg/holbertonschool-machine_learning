#!/usr/bin/env python3
"""Module to create a MultiIndex hierarchy with Timestamp as the first level"""
import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two dataframes and rearranges the MultiIndex.

    Args:
        df1: pd.DataFrame (coinbase)
        df2: pd.DataFrame (bitstamp)

    Returns:
        The concatenated pd.DataFrame with Timestamp as the first index level.
    """
    # 1. Индексируем оба DataFrame по колонке Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # 2. Выбираем временной диапазон [1417411980, 1417417980] для обоих
    df1_sub = df1.loc[1417411980:1417417980]
    df2_sub = df2.loc[1417411980:1417417980]

    # 3. Конкатенируем с ключами.
    # По умолчанию ключи становятся уровнем 0, а Timestamp уровнем 1.
    df = pd.concat([df2_sub, df1_sub], keys=['bitstamp', 'coinbase'])

    # 4. Переставляем уровни индекса (Timestamp становится уровнем 0)
    df = df.swaplevel(0, 1, axis=0)

    # 5. Сортируем по индексу для соблюдения хронологического порядка
    df = df.sort_index()

    return df
