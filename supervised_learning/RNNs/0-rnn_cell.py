#!/usr/bin/env python3
import numpy as np

class RNNCell:
    """Представляет одну ячейку простой RNN"""
    
    def __init__(self, i, h, o):
        """
        i - размерность входных данных
        h - размерность скрытого состояния (hidden state)
        o - размерность выходов
        """
        # Веса инициализируем нормальным распределением (Random Normal)
        # Так как умножение справа (X @ W), размерность Wh = (h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        
        # Смещения (Biases) инициализируем нулями
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Прямой проход (Forward Propagation) для одного шага во времени.
        x_t: (m, i) - входной батч
        h_prev: (m, h) - предыдущее скрытое состояние
        """
        # 1. Конкатенируем h_prev и x_t по столбцам (axis=1)
        # Размерность станет: (m, h + i)
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        
        # 2. Считаем h_next с активацией tanh
        # (m, h + i) @ (h + i, h) + (1, h) -> (m, h)
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)
        
        # 3. Считаем линейную часть для выхода
        # (m, h) @ (h, o) + (1, o) -> (m, o)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        
        # 4. Применяем Softmax для получения вероятностей
        # e^x / sum(e^x) вдоль оси 1 (по строкам батча)
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))  # Защита от overflow
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)
        
        return h_next, y
