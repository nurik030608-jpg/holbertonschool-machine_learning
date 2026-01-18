#!/usr/bin/env python3

def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n without loops.
    """
    if not isinstance(n, (int, float)) or n < 0:
        return None
    
    n = int(n)
    
    result = (n * (n + 1) * (2 * n + 1)) // 6
    
    return result
