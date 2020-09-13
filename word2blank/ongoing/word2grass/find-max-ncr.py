#!/usr/bin/env python3
import numpy as np

MAX_PRODUCT = 100
N = 20
C = np.zeros((N, N))
for n in range(N):
    C[n][0] = 1
    for r in range(1, n+1):
        C[n][r] = C[n-1][r] + C[n-1][r-1]

print("finding numbers (n, r) such that n*r < %d while maximizing nCr" % (MAX_PRODUCT, ))

maxi = 0; maxj = 0;
for i in range(N):
    for j in range(N):
        print("C(%d, %d) = %d" % (i, i, C[i][j]))
        if i * j > MAX_PRODUCT: continue
        if C[i][j] > C[maxi][maxj]:
            (maxi, maxj) = (i, j)
            print("***MAX: (%d, %d)***" % (maxi, maxj))

print("i: %d | j: %d | C(%d, %d) = %d)" % (maxi, maxj, maxi, maxj, C[maxi][maxj]))

