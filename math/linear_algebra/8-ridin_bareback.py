#!/usr/bin/env python3
"""
Module for matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices and returns the result
    """
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])

    if cols1 != rows2:
        return None

    # Initialize result matrix with zeros
    res = [[0 for _ in range(cols2)] for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                res[i][j] += mat1[i][k] * mat2[k][j]

    return res
