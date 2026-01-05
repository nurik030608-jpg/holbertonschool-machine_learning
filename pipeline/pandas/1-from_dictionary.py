#!/usr/bin/env python3
"""
Script that creates a pd.DataFrame from a dictionary
"""
import pandas as pd


# Создаем данные
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Создаем индекс (строки A, B, C, D)
index = ['A', 'B', 'C', 'D']

# Создаем DataFrame и сохраняем в переменную df
df = pd.DataFrame(data, index=index)
