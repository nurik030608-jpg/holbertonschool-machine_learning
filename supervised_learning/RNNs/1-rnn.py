#!/usr/bin/env python3
"""Module that contains the rnn function."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.

    rnn_cell is an instance of RNNCell
    X is a numpy.ndarray of shape (t, m, i) containing data
    h_0 is a numpy.ndarray of shape (m, h) containing initial hidden state

    Returns: H, Y
    H is a numpy.ndarray containing all hidden states
    Y is a numpy.ndarray containing all outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    # Инициализируем массивы для всех шагов во времени
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Записываем начальное состояние h_0 в самый первый слой (индекс 0)
    H[0] = h_0

    # Прогоняем цикл по каждому шагу во времени
    for step in range(t):
        x_t = X[step]
        h_prev = H[step]

        # Вызываем forward для одного шага у переданного rnn_cell
        h_next, y = rnn_cell.forward(h_prev, x_t)

        # Сохраняем полученные состояния
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
