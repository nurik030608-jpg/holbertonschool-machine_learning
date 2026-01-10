#!/usr/bin/env python3
"""Module to convert specific DataFrame columns to a numpy array"""


def array(df):
    """
    Converts the last 10 rows of the High and Close columns to a numpy array.

    Args:
        df: pd.DataFrame containing columns High and Close.

    Returns:
        numpy.ndarray: The last 10 rows of High and Close.
    """
    # Select the High and Close columns and take the last 10 rows
    selected_data = df[['High', 'Close']].tail(10)

    # Convert the selected data to a numpy array and return
    return selected_data.to_numpy()
