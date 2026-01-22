#!/usr/bin/env python3
"""Defines a function to calculate the shape of a matrix"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix and returns it as a list of integers.
    
    Args:
        matrix: A nested list representing a matrix.
        
    Returns:
        A list of integers representing the dimensions.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
