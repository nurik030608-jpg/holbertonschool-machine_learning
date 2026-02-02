#!/usr/bin/env python3 
def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication on two 2D matrices.
    """
    # Get dimensions
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])

    # Check if multiplication is possible
    if cols1 != rows2:
        return None

    # Initialize the result matrix with zeros (m x p)
    # Using nested list comprehension for a clean, new matrix
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    # Perform multiplication
    for i in range(rows1):          # Iterate through rows of mat1
        for j in range(cols2):      # Iterate through columns of mat2
            for k in range(cols1):  # Iterate through rows of mat2/cols of mat1
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
