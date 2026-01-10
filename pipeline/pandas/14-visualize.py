#!/usr/bin/env python3
"""Module to clean, process and visualize cryptocurrency data"""
import pandas as pd
import matplotlib.pyplot as plt


def visualize(df):
    """
    Processes a DataFrame and plots daily cryptocurrency data from 2017 onwards.

    Args:
        df: pd.DataFrame containing OHLCV data.

    Returns:
        The transformed pd.DataFrame.
    """
    # 1. Удаление Weighted_Price
    df = df.drop(columns=['Weighted_Price'])

    # 2. Переименование Timestamp в Date и конвертация в datetime
    df = df.rename(columns={'Timestamp': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    # 3. Установка Date в качестве индекса
    df = df.set_index('Date')

    # 4. Заполнение Close предыдущими значениями
    df['Close'] = df['Close'].ffill()

    # 5. Заполнение High, Low, Open значениями из Close
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 6. Заполнение объемов нулями
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # 7. Фильтрация данных: с 2017 года и далее
    df = df.loc['2017
