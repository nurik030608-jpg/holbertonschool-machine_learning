#!/usr/bin/env python3
"""
This module provides a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.
    Returns None if shapes are not equal.
    """
    if len(arr1) != len(arr2):
        return None

    return [x + y for x, y in zip(arr1, arr2)]
