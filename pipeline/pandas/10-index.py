#!/usr/bin/env python3
"""Module to set the index of a DataFrame"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.

    Args:
        df: pd.DataFrame containing a Timestamp column.

    Returns:
        The modified pd.DataFrame.
    """
    # Устанавливаем столбец 'Timestamp' как индекс
    df = df.set_index('Timestamp')

    return df
