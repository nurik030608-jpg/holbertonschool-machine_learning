#!/usr/bin/env python3
"""Module to fill missing values in a cryptocurrency DataFrame"""


def fill(df):
    """
    Fills missing values in the DataFrame according to specific logic.

    Args:
        df: pd.DataFrame to be modified.

    Returns:
        The modified pd.DataFrame.
    """
    # 1. Удаляем столбец Weighted_Price
    df = df.drop(columns=['Weighted_Price'])

    # 2. Заполняем пропуски в Close значениями из предыдущей строки (Forward Fill)
    df['Close'] = df['Close'].ffill()

    # 3. Заполняем пропуски в High, Low, Open значениями из Close той же строки
    # fillna() сработает только там, где значения еще отсутствуют (NaN)
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Устанавливаем пропуски в объемах в 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
