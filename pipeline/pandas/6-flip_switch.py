#!/usr/bin/env python3
"""Module to sort and transpose a DataFrame"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and transposes it.

    Args:
        df: pd.DataFrame to be transformed.

    Returns:
        The transformed pd.DataFrame.
    """
    # 1. Sort the dataframe by index in descending order
    # 2. Transpose the result using .T
    df = df.sort_index(ascending=False).T

    return df


