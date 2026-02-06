#!/usr/bin/env python3
"""
This module provides a function to calculate the adjugate matrix of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix (helper function).
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix.

    Args:
        matrix: A list of lists whose adjugate matrix should be calculated.

    Returns:
        The adjugate matrix of the input matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case for 1x1 matrix: adjugate is defined as [[1]]
    if n == 1:
        return [[1]]

    # 1. Create the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in
                          (matrix[:i] + matrix[i+1:])]
            minor_val = determinant(sub_matrix)
            row_cofactors.append(((-1) ** (i + j)) * minor_val)
        cofactor_matrix.append(row_cofactors)

    # 2. Transpose the cofactor matrix to get the adjugate
    adj_matrix = []
    for i in range(n):
        adj_row = []
        for j in range(n):
            adj_row.append(cofactor_matrix[j][i])
        adj_matrix.append(adj_row)

    return adj_matrix
