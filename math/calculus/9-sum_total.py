#!/usr/bin/env python3

def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n using the formula:
    n(n + 1)(2n + 1) / 6
    """
    # Validation: n must be an integer and n >= 1 
    # (If n=0, the sum is technically 0, but usually 'n' is a positive limit)
    if not isinstance(n, int) or n < 0:
        return None
    
    # Formula for the sum of the first n squares
    # We use integer division // to ensure the return value is an int
    return (n * (n + 1) * (2 * n + 1)) // 6
