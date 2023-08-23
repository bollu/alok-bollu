#!/usr/bin/env python3
import picos
from picos.solvers import *
import sys
import mosek
from random import *
import cvxpy as cp
import numpy as np


VECSIZE = 10
WINDOWSIZE = 2
EPSILON = 1e-1
NEGSAMPLES = 3

# TODO: figure out why the solver does not accept the kind of constraints
# we want. May need to read SDP lecture notes / lecture video

# TODO: solve a _mock_ problem (forget our problem, but something like
# the one we have commented in test_solver)

# Currently, we are setting the dot products to 0 or 1 depending on if
# the word is focus or context. If we read GLoVe, they have a value
# they set the dot product to based on co-occurence
# v . w = log (co-occurence of v & w)
# we should finally load the GLoVe co-occurence file, and set our solver
# dot product values to this log (co occurence)

# |target -  x . y | < ε
def mk_dot_product_constraint(problem, target, x, y, eps):
    problem.add_constraint(target - eps <= x | y)
    problem.add_constraint(x | y <= target + eps)

# |1 -  focus . ctx | < ε
def mk_positive_constraint(problem, focus, posctx, eps):
    mk_dot_product_constraint(problem, 1, focus, posctx, eps)


# |0 -  focus . ctx | < ε
def mk_negative_constraint(problem, focus, posctx, eps):
    mk_dot_product_constraint(problem, 0, focus, posctx, eps)

# |1 - v.v| < ε
def unit_vector_constraint(problem, v, eps):
    mk_dot_product_constraint(problem, v, v, eps)


def test_solver_cvxpy():
# Generate a random SDP.
    n = 2
    np.random.seed(1)
    C = np.zeros([2, 2], dtype=float)
    A = []
    b = []
    # x . x <= 1
    # x . y <= 0
    # y . x <= 0
    # y . y <= 1
    A.append(np.array([[1, 0.2], [0.2, 1]]))
    b.append(np.array([1, 1]))

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((2, 2), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
            cp.trace(A[i]@X) == b[i] for i in range(len(b))
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C@X)),
                                        constraints)
    print(prob)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)


def test_solver():
    P = picos.Problem()
    x = P.add_variable("x", 10, vtype="continuous")
    # y = P.add_variable("y", 10, vtype="continuous")
    z = P.add_variable("z", 1, vtype="continuous")

    print(x)
    
    # xT I x <= EPS
    # (x^T -I x) <= EPS
    EPS = 1.0

    P.add_constraint(z == 1.0)
    P.add_constraint(x | x <= 1)
    P.add_constraint(-1 <= x | x)
    P.set_objective("min", z)
    # P.set_objective("find", 0)
    print("problem:")
    print(P)
    print("solution:")
    P.solve(solver='cvxopt')
    print("z: ", z.value)
    print("x: ", x.value)

if __name__ == "__main__":
    test_solver_cvxpy()

if False and __name__ == "__main__":
    P = picos.Problem()

    input_file_path = sys.argv[1]

    # read file and load corpus into file.
    with open(input_file_path, "r") as f:
        corpus = []
        for line in f.read().split("\n"):
            for word in line.split(" "):
                if not word: continue
                corpus.append(word)

    # vocabulary: set of words in the corpus
    vocab = set(corpus)
    # mapping from vocabulary to picos vectors (symbolic)
    vocab2vec = { word : P.add_variable(word, VECSIZE, vtype="continuous") for word in vocab }

    print("building problem...")

    # total number of iterations
    total = len(corpus) * (WINDOWSIZE * 2 + NEGSAMPLES)
    curiter = 0

    # fi = index of focus word
    # ci = index of context word
    for fi in range(len(corpus)):
        # positive context words
        for ci in range(fi - WINDOWSIZE, fi + WINDOWSIZE):
            # skip this index
            if ci < 0 or ci >= len(corpus) - 1:
                continue
            
            curiter += 1

            # otherwise, add a positive context constraint
            mk_positive_constraint(P, vocab2vec[corpus[fi]], vocab2vec[corpus[ci]], EPSILON)


        # negative context words
        nneg_used = 0
        while nneg_used < NEGSAMPLES:
            ci = randint(0, len(corpus) - 1)
            # negative sample is inside positive window: reject
            if fi - WINDOWSIZE <= ci <= fi + WINDOWSIZE: continue

            nneg_used += 1
            curiter += 1
            # index is legal negative sample
            mk_negative_constraint(P, vocab2vec[corpus[fi]], vocab2vec[corpus[ci]], EPSILON)
        print("\rPercentage: %f" % (curiter / float(total) * 100), end='')

    # [TODO]
    #  Need to fix choices of hyperparameters, and also read GLOVE to use the
    #  GLOVE objective function (currently using word2vec objective function)
    # print("problem formulated. Problem: %s" % P)
    print("problem:")
    print(P)

    print("solving problem...")
    P.solve()

