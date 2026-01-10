#!/usr/bin/env python3
"""Module to slice a DataFrame by columns and rows"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row.

    Args:
        df: pd.DataFrame to be sliced.

    Returns:
        The sliced pd.DataFrame containing High, Low, Close, and Volume_(BTC).
    """
    # 1. Select the specific columns
    # 2. Use [::60] to take every 60th row from the beginning
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]

    return df
