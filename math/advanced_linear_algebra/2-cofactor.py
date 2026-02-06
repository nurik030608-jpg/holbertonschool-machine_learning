#!/usr/bin/env python3
"""
This module provides a function to calculate the cofactor matrix of a matrix.
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


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.

    Args:
        matrix: A list of lists whose cofactor matrix should be calculated.

    Returns:
        The cofactor matrix of the input matrix.

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

    # Special case for 1x1 matrix: cofactor is defined as 1
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            # Create sub-matrix for minor by removing row i and column j
            sub_matrix = [row[:j] + row[j+1:] for row in
                          (matrix[:i] + matrix[i+1:])]
            # Cofactor formula: (-1)^(i+j) * minor
            minor_val = determinant(sub_matrix)
            row_cofactors.append(((-1) ** (i + j)) * minor_val)
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix
