#!/usr/bin/env python3.6
# https://www.cvxpy.org/examples/applications/nonneg_matrix_fact.html
# Try to magically solve for co-occurence using quickmaf
import cvxpy as cp
import numpy as np
import argparse
import random
import collections
from array import array

# helper to create sparse matrices of 1% sparseness.
def sparse_vocab_x_vocab(m, n):
    return np.random.binomial(1, 0.1, (m, n)) * np.abs(np.random.randn(m, n))

# Ensure repeatably random problem data.
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("-inputpath", type=str, default='text8')
parser.add_argument("-dimsize", type=int, default=10)
parser.add_argument("-vocabsize", type=int, default=2000)
parser.add_argument("-windowsize", type=int, default=20)
parser.add_argument("-numiters", type=int, default=30000)
parser.add_argument("-batchrows", type=int, default=50)
args = parser.parse_args()


# Generate random data matrix A.
DIMSIZE = args.dimsize
VOCABSIZE = args.vocabsize
WINDOWSIZE = args.windowsize

print("reading corpus")
with open(args.inputpath, "r") as f:
    CORPUS = f.read()

assert CORPUS is not None
CORPUS = [w.strip() for w in CORPUS.split(' ') if w.strip()]
FREQ = collections.Counter(CORPUS)

print("building most common words")
# Zipf: fuck the top 10%
max_freq = 0
for w in FREQ: max_freq = max(max_freq, FREQ[w])

WORDS = set([w for (w, freq) in FREQ.most_common(VOCABSIZE) if freq >= 2])
VOCABSIZE = len(WORDS) # we may have less words than VOCABSIZE! adjust
IX2WORD = dict(enumerate(WORDS))
WORD2IX = { w: ix for (ix, w) in IX2WORD.items() }
print("filtering corpus")
# filter out corpus to be the words we want 
CORPUS = np.array([WORD2IX[w] for w in CORPUS if w in WORD2IX])


print("building word |-> index")
print("\n".join([str(k) for (k, v) in WORD2IX.items()]))

# sparse
A = np.zeros((VOCABSIZE, VOCABSIZE), dtype=np.float)

# (focus, ctx)
print("\n")
for fix in range(len(CORPUS)):
    print("\rfilling up coocurrence [%8d/%8d]"% (fix, len(CORPUS)), end='')
    f = CORPUS[fix]
    window = range(max(f - WINDOWSIZE, 0), min(f+WINDOWSIZE, len(CORPUS)))
    for cix in window:
        c = CORPUS[cix]
        A[f][c] += 1; A[c][f] += 1
A /= np.max(A)
print("\n")

# A = np.eye(VOCABSIZE)
print("A shape: [%s] | sum: [%5.4f] | A filled percent: [%4.2f %%]" % \
        (A.shape, np.sum(A), 100.0 * np.sum(A) /VOCABSIZE / VOCABSIZE))





print("generating random Y...")

# Pick random x, y values
# YVAL = XVAL = sparse_vocab_x_vocab(VOCABSIZE, DIMSIZE)
XVAL = np.eye(VOCABSIZE, DIMSIZE)

print("initial state: [✗ full: %8.3f]" % (np.linalg.norm(A - XVAL @ XVAL.T)))

def write_output_file(): # really a macro, not a function
    with open("x.bin", "wb") as f:
        f.write(bytes("%s %s\n" % (VOCABSIZE, DIMSIZE), 'utf-8'))
        for ix in range(VOCABSIZE):
            f.write(bytes("%s " % (IX2WORD[ix]), 'utf-8'))
            array('f', XVAL[ix]).tofile(f); f.write(bytes("\n", 'utf-8'))


# Perform alternating minimization.
NUM_ITERS = args.numiters
residual = np.zeros(NUM_ITERS)
print("\n\n=========")
for cur_iter in range(1, 1+NUM_ITERS):
    output_str = ""
    output_str += "[iter %5d/%5d]" % (cur_iter, NUM_ITERS)


    NROWS = min(args.batchrows, VOCABSIZE)
    # TODO: sample based on frequency?
    ROWS = random.sample(range(0, VOCABSIZE), NROWS)

    X = cp.Variable(shape=(NROWS, DIMSIZE))
    constraint = [X >= -2, X <= 2]
    XVAL_CUR = XVAL[ROWS] # HACK!


    # Solve the problem.
    # increase max iters otherwise, a few iterations are "OPTIMAL_INACCURATE"
    for _ in range(5):
        obj = cp.Minimize(cp.norm(A[ROWS, :][:, ROWS] - X@XVAL_CUR.T, 'fro'))
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.GUROBI)

        if prob.status != cp.OPTIMAL:
            output_str += "[no converge]"
            continue # skip iteration

        # Convert variable to NumPy array constant for next iteration.
        XVAL[ROWS,:] = X.value

    # ✗ = error
    output_str += "[✗ solver %8.2f]" %(prob.value) # error reported by solver
    residual[cur_iter-1] = prob.value
    output_str += "[✗ batch: %8.3f]" \
            % (np.linalg.norm(A[ROWS, :][:, ROWS] - XVAL[ROWS] @ XVAL[ROWS].T)) # error calculated by us
    output_str += "[✗ corpus: %8.3f]" \
        % (np.linalg.norm(A - XVAL @ XVAL.T))
    print("\r" + output_str, end='')


    if cur_iter % 10 == 0:
        write_output_file()
        print(" [wrote output]")

# write out matrix
print("\n".join(["%8s: %20s" % (ix, w) for (ix, w) in IX2WORD.items()]))
write_output_file()


