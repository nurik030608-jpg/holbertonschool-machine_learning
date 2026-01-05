import pandas as pd

# Данные для столбцов
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Метки строк
row_labels = ['A', 'B', 'C', 'D']

# Создание DataFrame
df = pd.DataFrame(data, index=row_labels)
