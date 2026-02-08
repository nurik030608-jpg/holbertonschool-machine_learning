#!/usr/bin/env python3
"""Function for element-wise operations using NumPy"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    Returns a tuple containing the sum, difference, product, and quotient
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
