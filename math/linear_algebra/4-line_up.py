#!/usr/bin/env python3
"""
Этот блок называется docstring. В некоторых проверках он обязателен
для описания модуля.
"""


def add_arrays(arr1, arr2):
    """
    Складывает два списка поэлементно.
    Если формы не совпадают, возвращает None.
    """
    if len(arr1) != len(arr2):
        return None
    
    # Создаем новый список, не меняя старые
    return [x + y for x, y in zip(arr1, arr2)]
