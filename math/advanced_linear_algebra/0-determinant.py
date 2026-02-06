#!/usr/bin/env python3
"""
This module provides a function to calculate the determinant of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix: A list of lists whose determinant should be calculated.

    Returns:
        The determinant of the matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix.
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Handle the 0x0 case [[]] or []
    if matrix == [[]] or matrix == []:
        return 1

    # Check if every element is a list
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Check if square matrix
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix for performance
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    # Recursive step: Laplace expansion along the first row
    det = 0
    for j in range(n):
        # Create a sub-matrix by removing the 0th row and j-th column
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        # Calculate cofactor: (-1)^i+j * element * minor
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)

    return det
