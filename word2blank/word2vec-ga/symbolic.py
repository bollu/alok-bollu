#!/usr/bin/env ipython3 
# symbolic represenatation of GA operations
# I suspect that the A matrix has nice properties.
import sympy
import sympy as sp


# degree that x is contained in y
def LContainedInR(x, y):
    dot = 0
    for i in range(len(x)):
        for j in range(len(y)):
            # i subset j => i & j == i
            if i & j != i:
                continue
            dot += x[i] * y[j]
    return dot

def AContainment(dim):
    A = [[0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            # i subset j => i & j == i
            if i & j == i:
                if i | j != j: raise Exception("ERROR")
            A[i][j] = 1
    return A


def matmul(x, A, y):
    z = 0
    for i  in range(len(x)):
        for j in range(len(y)):
                z += x[i] * A[i][j] * y[j]
    return z

def gavec(name, coeffs, ndims):
    assert(len(coeffs) == (1 << ndims))
    vec = []
    for i in range(1 << ndims):
        coeffname = name
        for j in range(ndims):
            coeffname += "@" if i & (1 << j) > 0 else "-"
        vec.append(coeffs[i] * sympy.Symbol(coeffname))
    return vec

for D in range(1, 4):
    print("D: %s" % D)
    x = gavec("x", [1 for _ in range(1<<D)], D)
    y = gavec("y", [1 for _ in range(1<<D)], D)

    sp.pprint(sympy.simplify(LContainedInR(x, y)))
    A = AContainment(1 << D)
    sp.pprint(LContainedInR(x, y) == matmul(x, A, y))

    A = sp.Matrix(A)
    print("A.rank: %s" % A.rank())

    for (val, mul, vecs) in A.eigenvects():
        for vec in vecs:
            sp.pprint("λ = %s | mul = %s | vec = %s" % (val, mul, vec.T))


    # P, D = A.diagonalize()
    # print("Change of basis:")
    # print(P)
    # 
    # print("Diagonal:")
    # print(D)
            
    lam = sp.symbols("l")
    cp = sp.det(A - lam * sp.eye(1 << D))
    eigs = sp.roots(sp.Poly(cp, lam))

    print("cp: %s" % sp.factor(cp))
    print("eigs: %s" % eigs)

