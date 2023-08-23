#!/usr/bin/env python3
import picos
from picos.solvers import *
import sys
import mosek
from random import *

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

def test_solver():
    P = picos.Problem()
    x = P.add_variable("x", (10, 10), vtype="symmetric")
    y = P.add_variable("y", (10,10) , vtype="symmetric")
    
    # xT I x <= EPS
    # (x^T -I x) <= EPS
    EPS = 1.0
    one = picos.new_param("two", 2)
    # c2= P.add_constraint(x | y  <= -1)
    c3 = P.add_constraint (x >> 0)
    c3 = P.add_constraint (y >> 0)
    
    # c2= P.add_constraint(-EPS <= x | x )
    # c2= P.add_constraint(EPS >= -(x | x ))
    # c2= P.add_constraint(-(x | x ) <= EPS)

    P.set_objective("min", x[0]y)
    # P.set_objective("find", 0)
    print("problem:")
    print(P)
    print("solution:")
    P.solve(solver='mosek')
    #P.solve()
    print("x: ", x.value)

if __name__ == "__main__":
    test_solver()

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



