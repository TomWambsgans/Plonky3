#!/usr/bin/env python3
"""
Check whether the 8x8 circulant matrix with first row
    [-1, -2, 1, -2, 8, 8, 4, 1]
is MDS over the Goldilocks prime field p = 2^64 - 2^32 + 1.

An n x n matrix is MDS iff every square submatrix is non-singular.
We enumerate all k x k minors for k = 1..8 and check det != 0 mod p.
"""

from itertools import combinations

P = (1 << 64) - (1 << 32) + 1  # Goldilocks prime
N = 8

first_row = [-1, -2, 1, -2, 8, 8, 4, 1]
# Build circulant: row i is first_row shifted by i (row i, col j) = first_row[(j - i) mod N]
M = [[first_row[(j - i) % N] % P for j in range(N)] for i in range(N)]

def det_mod(mat, p):
    """Compute determinant mod p using fraction-free / modular Gauss elimination."""
    n = len(mat)
    A = [row[:] for row in mat]
    sign = 1
    det = 1
    for i in range(n):
        # find pivot
        pivot = -1
        for r in range(i, n):
            if A[r][i] % p != 0:
                pivot = r
                break
        if pivot == -1:
            return 0
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            sign = -sign
        inv = pow(A[i][i], -1, p)
        det = (det * A[i][i]) % p
        for r in range(i + 1, n):
            if A[r][i] % p == 0:
                continue
            factor = (A[r][i] * inv) % p
            for c in range(i, n):
                A[r][c] = (A[r][c] - factor * A[i][c]) % p
    if sign == -1:
        det = (-det) % p
    return det

def submatrix(M, rows, cols):
    return [[M[r][c] for c in cols] for r in rows]

failures = []
total = 0
for k in range(1, N + 1):
    count_k = 0
    fail_k = 0
    for rows in combinations(range(N), k):
        for cols in combinations(range(N), k):
            sub = submatrix(M, rows, cols)
            d = det_mod(sub, P)
            count_k += 1
            total += 1
            if d == 0:
                fail_k += 1
                if len(failures) < 20:
                    failures.append((k, rows, cols))
    print(f"k={k}: checked {count_k} minors, singular = {fail_k}")

print(f"\nTotal minors checked: {total}")
if not failures:
    print("RESULT: All minors non-zero -> matrix IS MDS over Goldilocks.")
else:
    print(f"RESULT: NOT MDS. {len(failures)} (showing up to 20) singular minors:")
    for k, rows, cols in failures:
        print(f"  k={k} rows={rows} cols={cols}")
