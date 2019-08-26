#!/usr/bin/env python3
import sys
import struct
import numpy as np
# import pycuda
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import skcuda.linalg as linalg
# import sklearn


def read_occur_np(p):
    ty = np.dtype([('word1', np.int32), ('word2', np.int32), 
                   ('val', np.double)])
    with open(p, "rb") as f:
        return np.frombuffer(f.read(), ty)

def GPU(): return False
def RAND(): return False

if __name__ == "__main__":
    cooccur_path = sys.argv[1]
    words_path = sys.argv[2]

    with open(words_path, "r") as f:
        words = [line.split(' ')[0] for line in f.readlines()]
    print("words: ", words[:10])


    structs = read_occur_np(cooccur_path)
    print("===NUMPY====")
    print("length of structs: ", len(structs))
    nwords = np.max(structs['word1'])
    print("number of words: ", nwords)
    dots = np.ndarray(shape=(nwords+1, nwords+1), dtype=np.float)

    assert(nwords == len(words))


    print("setting up dots...")
    i = 0
    nrecords = len(structs)
    dots[structs['word1'], structs['word2']] = structs['val']


    print("done setting up dots.")
    print("dots: ")
    print(dots[1,0:100])
    print("====")
    print("structs of word1: ")
    print(structs[0:100])
    print("====")

    print("dots setup, it looks like. Remember, dots[0][0] is meaningless")
    dots = dots[1:, 1:]

    print("calculating ||dots - dots.T||")
    # Check that the matrix is symmetric. Only then will SVD work...
    for i in range(100):
        for j in range(100):
            if np.abs(dots[i][j] - dots[j][i]) > 1e-1:
                print("delta:", dots[i][j] - dots[j][i])
        print("no delta for: ", i)
    print("shape of dots: ", dots.shape)
    print("calculating SVD...")

    if GPU():
        print("sending u to GPU...")
        dots_gpu = gpuarray.to_gpu(np.asarray(dots, dtype=np.float))
        print("sent. calculating SVD of: ", dots_gpu)
        u_gpu, s_gpu, vt_gpu = linalg.svd(dots_gpu)
        print("u.shape: ", u_gpu.shape)
    else:
        if RAND():
            (u, s, vt) = sklearn.utils.extmath.randomized_svd(dots)
        else:
            (u, s, vt) = np.linalg.svd(dots)

        # can directly use SVD since it is in descending order
        EMBEDSIZE = 300
        X = np.array([np.sqrt(s[i]) * u[i] for i in range(EMBEDSIZE)])
        print("X.shape: ", X.shape)

        with open("vectors-exact.txt", "w") as f:
            for i in range(len(words)):
                f.write("%s " % words[i])
                for j in range(EMBEDSIZE):
                    f.write("%f " % X[j][i])
                f.write("\n")
