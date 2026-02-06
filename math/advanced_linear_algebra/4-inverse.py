#!/usr/bin/env python3
"""Module to calculate the inverse of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix"""
    n = len(matrix)
    if n == 1:
        return [[1]]
    
    cofactor_mat = []
    for i in range(n):
        row_list = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            row_list.append(((-1) ** (i + j)) * determinant(sub_matrix))
        cofactor_mat.append(row_list)
    return cofactor_mat


def inverse(matrix):
    """Calculates the inverse of a square matrix"""
    # Validation
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case for 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None
        return [[1 / matrix[0][0]]]

    # Calculate determinant
    det = determinant(matrix)
    if det == 0:
        return None

    # Calculate adjugate matrix (transpose of cofactor matrix)
    cofactors = cofactor(matrix)
    adjugate = []
    for j in range(n):
        new_row = []
        for i in range(n):
            new_row.append(cofactors[i][j] / det)
        adjugate.append(new_row)

    return adjugate
