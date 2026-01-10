#!/usr/bin/env python3
"""Module to rename a column and convert to datetime"""
import pandas as pd


def rename(df):
    # 1. Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # 2. Convert to datetime specifying the unit as seconds ('s')
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # 3. Filter to display only the Datetime and Close columns
    df = df[['Datetime', 'Close']]

    return df
