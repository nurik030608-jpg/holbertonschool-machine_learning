#!/usr/bin/env python3
"""Policy gradient: policy function"""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix"""
    z = matrix @ weight
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)
