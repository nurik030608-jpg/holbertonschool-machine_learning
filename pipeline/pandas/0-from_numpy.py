import pandas as pd
import string

def from_numpy(array):
    """
    Создает pd.DataFrame из np.ndarray с заголовками колонок в алфавитном порядке.
    """
    # Определяем количество колонок в массиве
    num_columns = array.shape[1]
    
    # Генерируем список букв (A, B, C...) в зависимости от количества колонок
    column_names = list(string.ascii_uppercase[:num_columns])
    
    # Создаем DataFrame
    df = pd.DataFrame(array, columns=column_names)
    
    return df
