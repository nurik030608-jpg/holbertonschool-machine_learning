#!/usr/bin/env python3
"""Модуль, содержащий функцию word2vec_model."""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Создаёт, строит и обучают модель gensim Word2Vec.

    Args:
        sentences (list): Список предложений для обучения.
        vector_size (int): Размерность пространства эмбеддингов.
        min_count (int): Минимальная частота слова для включения в словарь.
        window (int): Максимальное расстояние между текущим и контекстным словом.
        negative (int): Количество отрицательных примеров (negative sampling).
        cbow (bool): True для архитектуры CBOW, False для Skip-gram.
        epochs (int): Количество эпох (итераций обучения).
        seed (int): Сид для генератора случайных чисел.
        workers (int): Количество рабочих потоков.

    Returns:
        Word2Vec: Обученная модель gensim Word2Vec.
    """
    # sg = 0 обозначает CBOW, sg = 1 обозначает Skip-gram
    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
