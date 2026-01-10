#!/usr/bin/env python3
"""Module to remove rows with NaN values in the Close column"""


def prune(df):
    """
    Removes any entries where Close has NaN values.

    Args:
        df: pd.DataFrame to be pruned.

    Returns:
        The modified pd.DataFrame.
    """
    # Удаляем строки, где в столбце 'Close' есть пропуски (NaN)
    df = df.dropna(subset=['Close'])

    return df
