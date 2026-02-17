#!/usr/bin/env python3
import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Создает матрицу ошибок.
    labels: реальные метки (one-hot)
    logits: предсказания (one-hot)
    """
    # 1. Превращаем one-hot в обычные индексы классов (0, 1, 2...)
    # axis=1 означает, что мы ищем индекс максимального значения в каждой строке
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)

    # 2. Узнаем количество классов
    num_classes = labels.shape[1]

    # 3. Создаем матрицу, заполненную нулями
    confusion = np.zeros((num_classes, num_classes))

    # 4. Итерируемся по всем точкам данных и заполняем ячейки
    for i in range(len(y_true)):
        # y_true[i] - индекс строки (реальность)
        # y_pred[i] - индекс столбца (предсказание)
        confusion[y_true[i], y_pred[i]] += 1

    return confusion
