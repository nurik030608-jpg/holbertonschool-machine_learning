#!/usr/bin/env python3
"""Module for transposing a 2D matrix."""


def matrix_transpose(matrix):
    """
    Returns a new transposed matrix.

    Args:
        matrix: A 2D list (matrix).

    Returns:
        A new 2D list where rows and columns are swapped.
    """
    # Calculate dimensions
    rows = len(matrix)
    cols = len(matrix[0])

    # Create new matrix by iterating over columns then rows
    transpose = [[matrix[i][j] for i in range(rows)] for j in range(cols)]

    return transpose
