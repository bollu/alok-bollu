import tensorflow as tf
import tensorflow.logging
import tensorflow.random
import numpy as np
import numpy.linalg
from tensorflow import keras
from collections import OrderedDict
import os
import random
import numba


tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)

SAVEFOLDER='models'
SAVEPATH='text0-transformer.bin'
INPUTPATH='text0'
SENTENCELEN = 30
LEARNING_RATE=0.1
NEPOCHS=10
BATCHSIZE=100000

with open(INPUTPATH, "r") as f:
  corpus = f.read()
  corpus = [w for w in corpus.split() if w]
  CORPUSLEN = len(corpus)
  # print("corpus:\n|%s|" % corpus)
  # stable sort
  vocab = list(OrderedDict.fromkeys(corpus))
  print("vocab:\n|%s|" % vocab[:5])
  
  # map words to their index in the embedding array
  VOCAB2IX = {w: i for (i, w) in enumerate(vocab)}
  # print("VOCAB2IX:\n%s" % VOCAB2IX)
  VOCABSIZE = len(vocab)


  corpusixed = np.empty(CORPUSLEN, dtype=np.int32)
  for i in range(CORPUSLEN):
      corpusixed[i] = VOCAB2IX[corpus[i]]

assert VOCABSIZE is not None
assert CORPUSLEN is not None

EMBEDSIZES = [VOCABSIZE, 32, 16]



# numba python -> LLVM
@numba.jit(nopython=True, parallel=True)
def mkdata():
  fixs = np.empty(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.int32)
  cixs = np.empty(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.int32)
  labels = np.empty(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.int32)

  n = 0
  r = np.uint32(1)
  for ixf in np.arange(CORPUSLEN):
    l = max(0, ixf - WINDOWSIZE)
    r = min(CORPUSLEN - 1, ixf + WINDOWSIZE)
    
    # the fox [vc=|jumps| *vf=over* the] dog (vc.vf=1)
    for ixc in np.arange(l, r):
      # variable[placeholder]
      fixs[n] = corpusixed[ixf]
      cixs[n] = corpusixed[ixc]
      labels[n] = 1
      n += 1
  
    # vc=|the| fox [jumps *vf=over* the] dog (vc.vf = 0)
    for _ in np.arange(NEGSAMPLES):
      r = r * 25214903917 + 11
      ixrand = r % (CORPUSLEN - 1)
      # if l <= ixrand <= r: continue # reject words inside window
      fixs[n] = corpusixed[ixf]
      cixs[n] = corpusixed[ixrand]
      labels[n] = 0
      n += 1
  
    print((100.0 * n / (CORPUSLEN * (2 * WINDOWSIZE + NEGSAMPLES))))

  return fixs, cixs, labels, n



# learning rate

# input sentence, one-hot encoding of sentence
ph_sentence = tf.placeholder(tf.float32, (SENTENCELEN, VOCABSIZE), name="ph_sentence")

# embedding matrices per layer
embedM = []
for l in range(len(EMBEDSIZES-1)):
  v = 1.0 / EMBEDSIZES[i]
  init = tf.random.uniform([EMBEDSIZES[i], EMBEDSIZES[i+1]], 
                           minval=-v,
                           maxval=v)
  embedM.append(tf.Variable(init, name="emedding-%s" % (i, )))


# atttention computation
# computes weight of each embedding for input
attnMs = []

# embedded vectors per layer:
# ph_sentence: SENTENCELEN x VOCABSIZE
# embedM[0]: VOCABSIZE x EMBEDSIZE[1]
# product: SENTENCELEN x EMBEDSIZE[1]
embeds = [tf.matmul(ph_sentence, embedM[0])]

for l in range(len(EMBEDSIZES-1)):

def mk_attended_layer(attnM, sentence, slen, dim):
  # sentence: slen x dim
  # attenM: slen x slen x slen

  # embeds: [w][i]: SENTENCELEN x EMBEDSIZE[l+1]
  # gaussians: [i][j]: e^-|i - j|: exponentially decaying distance from i to j
  #      EMBEDSIZE[l+1] x EMBEDSIZE[l+1]
  # distributedS: [w][k] # for each word, what is my distributional semantics?
  #                        it's a weighted sum of all the words around me.
  #        SENTENCELEN x EMBEDSIZE[l+1]
  # attentionS: based on the current distributedS, force each word to pick
  # over all other words
  distributedS = embeds[l]# tf.matmul(embeds[l], ph_gaussians)

  attnMs.append(tf.Variable(tf.random.normal([EMBEDSIZE[l+1], EMBEDSIZE[l+1],
                                             EMBEDSIZE[l+1]]), name="attnMs-" + str(l))

  # (SENTENCELENxEMBEDSIZE) x (DISTRLENxEMBEDSIZE) -> (SENTENCELENxEMBEDSIZE)
  # d[w][i] x d[w][j] -> d[w][k]
  # d[w][k] := attn[k][i][j] d[w][i] d[w][j]
  attnS = tf.einsum("wk,kij,wi->wj", distributedS, attnMs[l], distributedS)



ph_lr = tf.placeholder(tf.float32, name="ph_lr")
optimizer = tf.train.AdamOptimizer(learning_rate=ph_lr).minimize(var_loss)

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


def epoch(curepoch, sess, n, data_fixs, data_cixs, data_labels, data_lr):
    i = 0
    while (i + 2) * BATCHSIZE < n:
      i += 1
      loss, _ = sess.run([var_loss, optimizer], 
                         feed_dict={ph_fixs:data_fixs[i*BATCHSIZE:(i+1)*BATCHSIZE], 
                                    ph_cixs: data_cixs[i*BATCHSIZE:(i+1)*BATCHSIZE],
                                    ph_labels: data_labels[i*BATCHSIZE:(i+1)*BATCHSIZE],
                                    ph_lr: data_lr})

      print("epoch: %10s | loss: %20.5f | lr: %20.8f | %10.2f %%" % (curepoch, 
                          loss,
                          data_lr,
                          (100 * (curepoch + (i * BATCHSIZE/ n)) / NEPOCHS)))
      # data_lr = data_lr * (1.0 - 1e-6)
      # data_lr = max(1e-7, max(LEARNING_RATE * 1e-5, data_lr))



def train():
  saver = tf.train.Saver()
  with tf.Session() as sess:
    global_init = tf.global_variables_initializer()
    sess.run(global_init)
    data_fixs = []
    data_cixs = []
    data_labels = []
    data_lr = LEARNING_RATE

    print("making data...")
    fixs, cixs, labels, n = mkdata()
    print("done. n: %10s" % (n, ))

    # print("\n***LLVM of mkdata:***")
    # for v, k in mkdata.inspect_llvm().items():
    #     print(v, k)
    # print("***end LLVM of mkdata:***\n")
    # raise RuntimeError("inspection")

    for i in range(NEPOCHS):
      print("===epoch: %s===" % i)
      epoch(i, sess, n, fixs, cixs, labels, data_lr) 
  
    data_syn0 = sess.run([var_syn0])
    data_syn1neg = sess.run([var_syn1neg])
  
    if not os.path.exists(SAVEFOLDER):
      os.makedirs(SAVEFOLDER)
  
    saver.save(sess, SAVEPATH)

    # TASK 1. Pull out the data from the session, and _print it_. Maybe try and
    # implement distance()
    
    # print distance of fox from all other words, ordered by ascending order (Dot
    # product / cosine distance)
    # distance('fox', data_syn0)
  
    # quick - fox + dog == ? print best candidates for this
    # Fox :  quick :: fox : ? == (quick - fox) + fox = quick
    # analogy('fox', 'quick', 'dog', data_syn0)
    
    # TASK 2. copy and understand (plz plz plz) the data saving/loading code, and
    # save the learnt word vectors.
  
    # TASK 3. make this batced: use multiple indeces and
    # multipl labels _in batch mode_. I presume this is requires one to
    # change the code to "store" the (fix, cix, and labels) and then
    # pass them as arrays to sess.run(...)

def distance():
  saver = tf.train.Saver()
  IX2VOCAB = {VOCAB2IX[w]:w for w in VOCAB2IX}
  with tf.Session() as sess:
    saver.restore(sess, SAVEPATH)
    [syn0] = sess.run([var_syn0])
    print("syn0 shape: %s" % (syn0.shape, ))
    print(type(syn0))

    for i in range(VOCABSIZE):
        syn0[i, :] = syn0[i, :] / np.linalg.norm(syn0[i, :])

    while True:
        w = input(">")
        if w == "exit": break
        if w not in VOCAB2IX: continue
        wix = VOCAB2IX[w]
        wv = syn0[wix, :]
        dots = np.dot(syn0, wv)
        print ("syn0:\n%s\nwv:\n%s\ndots:\n%s" % (syn0, wv, dots))
        print("wv len: %s" % (np.dot(wv, wv), ))
        ixs = np.argsort(dots)
        ixs = np.flip(ixs)
        for ix in ixs[:30]:
            print("%20s %10.5f" % (IX2VOCAB[ix], dots[ix]))

if __name__ == "__main__":
    test_positional_encoding()
    train()
    distance()
