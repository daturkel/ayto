import numpy as np
from math import factorial
from numpy.typing import NDArray


def faster_permutations(n: int) -> NDArray:
    """Generate a numpy array of the permutations 0 to n-1.

    Source: Daniel Giger's StackOverflow answer: https://stackoverflow.com/a/71231033

    """
    # empty() is fast because it does not initialize the values of the array
    # order='F' uses Fortran ordering, which makes accessing elements in the same column fast
    perms = np.empty((factorial(n), n), dtype=np.uint8, order="F")
    perms[0, 0] = 0

    rows_to_copy = 1
    for i in range(1, n):
        perms[:rows_to_copy, i] = i
        for j in range(1, i + 1):
            start_row = rows_to_copy * j
            end_row = rows_to_copy * (j + 1)
            splitter = i - j
            perms[start_row:end_row, splitter] = i
            perms[start_row:end_row, :splitter] = perms[
                :rows_to_copy, :splitter
            ]  # left side
            perms[start_row:end_row, splitter + 1 : i + 1] = perms[
                :rows_to_copy, splitter:i
            ]  # right side

        rows_to_copy *= i + 1

    return perms
