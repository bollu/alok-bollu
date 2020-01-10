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

# TODO: make data loading from corpus faster (Siddharth)
# TODO: run on text8 and generate a vector.bin file according to word2vec
#       convention so we can run compute-accuracy and distance (Siddharth)
# TODO: include all poplar tricks such as frequency based discarding (Souvik)
# TODO: adapt the code to GA (Souvik)



tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)

SAVEFOLDER='models'
SAVEPATH='text8.bin'
INPUTPATH='text8'
EMBEDSIZE = 50
WINDOWSIZE = 8
NEGSAMPLES = 15
LEARNINGRATE=1e-2
NEPOCHS=15
BATCHSIZE=10000
DISTANCE_DOT_DECAY = 1.0 / 10.0
REGULARIZATION_COEFF = 0

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


  # corpus: location -> str
  # corpusixed: location -> int (ix = index)
  corpusixed = np.empty(CORPUSLEN, dtype=np.int32)
  for i in range(CORPUSLEN):
      corpusixed[i] = VOCAB2IX[corpus[i]]

assert VOCABSIZE is not None
assert CORPUSLEN is not None


# syn0_init = np.random.random((VOCABSIZE, EMBEDSIZE)) - 0.5
# for i in range(VOCABSIZE):
#   syn0_init[i, :] = syn0_init[i, :] / np.linalg.norm(syn0_init[i, :])
# var_syn0 = tf.Variable(tf.constant(syn0_init, dtype=tf.float32,
#                                    shape=(VOCABSIZE, EMBEDSIZE)), name="syn0")

VAL = 1.0 / EMBEDSIZE
var_syn0 = tf.Variable(tf.random.uniform([VOCABSIZE, EMBEDSIZE],
                                         minval=-VAL,
                                        maxval=VAL), name="syn0")

var_syn1neg = tf.Variable(tf.zeros([VOCABSIZE, EMBEDSIZE]), 
                          name="syn1neg")

ph_fixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_fixs")
ph_cixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_cixs")
ph_labels = tf.placeholder(tf.float32, (BATCHSIZE, ), name="ph_labels")

# loss = (label - (focus[fix] . ctx[cix])^2
# BATCHSIZE x EMBEDSIZE
var_syn0_cur = tf.gather(var_syn0, ph_fixs, name="syn0_cur")
var_syn1neg_cur = tf.gather(var_syn1neg, ph_cixs, name="syn1neg_cur")

# syn0       syn1neg
# 1 2        1 0
# 3 4        0 1
# 5 6        2 3

# syn0 * syn1neg
# BATCHSIZE x EMBEDSIZE
# 1  0
# 0  4
# 10 18

# tf.reduce_sum(..., axis=1)
# 1
# 4
# 28
var_dots = tf.reduce_sum(tf.multiply(var_syn0_cur, var_syn1neg_cur), axis=1, name="dots")

var_clipped_dots = tf.math.sigmoid(var_dots, "clipped_dots")
# var_clipped_dots = var_dots

# loss = tf.norm(tf.math.sub(ph_label, d), name="loss")
var_losses_dot = tf.squared_difference(ph_labels, var_clipped_dots, name="losses_dot")
# add up all the losses
var_loss = tf.reduce_sum(var_losses_dot, name="loss_dot")
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNINGRATE).minimize(var_loss)

# var_lensq = tf.reduce_sum(tf.multiply(var_syn0_cur, var_syn0_cur), axis=1, name="lensq")
# 
# var_regularize = tf.squared_difference(1.0, var_lensq, name="regularizations")
# var_regularize = tf.multiply(var_regularize, REGULARIZATION_COEFF,
#                              name="regularizations_scaled")
# var_regularize = tf.reduce_sum(var_regularize, name="regularization_final")
# 
# 
# # var_loss = tf.add(var_regularize, var_loss_dot, name="loss")

# learning rate
# ph_lr = tf.placeholder(tf.float32, name="ph_lr")


print("\n***NETWORK:***")
print("syn0: %s | syn1neg: %s" % (var_syn0, var_syn1neg))
print("syn0 cur: %s" % var_syn0_cur)
print("syn1neg cur: %s" % var_syn1neg_cur)
print("dots: %s" % var_dots)
print("clipped dots: %s" % var_clipped_dots)
print("losses_dot: %s" % var_losses_dot)
# print("loss_dot: %s" % var_loss_dot)
# print("lensq: %s" % var_lensq)
# print("regularize: %s" % var_regularize)
print("loss: %s" % var_loss)
print("***END NETWORK:***\n")

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


# @numba.jit(nopython=True, parallel=True)
# WTF, how is _this_ the bottleneck?
# TODO: make this faster
def mkdata():
  fixs = np.empty(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.int32)
  cixs = np.empty(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.int32)
  labels = np.empty(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.float32)

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
      fixs[n] = corpusixed[ixf]
      cixs[n] = corpusixed[ixrand]
      labels[n] = 0
      n += 1
  
    print((100.0 * n / (CORPUSLEN * (2 * WINDOWSIZE + NEGSAMPLES))))


  return fixs, cixs, labels, n

def shuffledata(fixs, cixs, labels):
  print("shape: |%s|" % fixs.shape)
  randixs = np.arange(fixs.shape)
  randixs = np.random.shuffle(randixs)
  fixs = fixs[randixs]
  cixs = cixs[randixs]
  labels = labels[randixs]

  return fixs, cixs, labels


def epoch(curepoch, sess, n, data_fixs, data_cixs, data_labels, data_lr):
    i = 0
    while (i + 1) * BATCHSIZE < n:
      i += 1
      loss, _ = sess.run([var_loss, optimizer], 
                         feed_dict={ph_fixs:data_fixs[i*BATCHSIZE:(i+1)*BATCHSIZE], 
                                    ph_cixs: data_cixs[i*BATCHSIZE:(i+1)*BATCHSIZE],
                                    ph_labels: data_labels[i*BATCHSIZE:(i+1)*BATCHSIZE]
                                   })

      print("\repoch: %10s | loss: %20.5f | lr: %20.8f | %10.2f %%" % (curepoch, 
                          loss,
                          data_lr,
                          (100 * (curepoch + (i * BATCHSIZE/ n)) / NEPOCHS)),
            end="")

def train():
  saver = tf.train.Saver()
  with tf.Session() as sess:
    global_init = tf.global_variables_initializer()
    sess.run(global_init)
    data_lr = LEARNINGRATE

    print("load data...\n")
    fixs, cixs, labels, n = mkdata()
    # fixs, cixs, labels = shuffledata(fixs, cixs, labels, n)
    print("done. n: %10s" % (n, ))

    print("\n***LLVM of mkdata:***")
    for v, k in mkdata.inspect_llvm().items():
        print(v, k)
    print("***end LLVM of mkdata:***\n")

    for i in range(NEPOCHS):
      print("\n===epoch: %s===" % i)
      epoch(i, sess, n, fixs, cixs, labels, data_lr) 
      # data_lr = data_lr * 0.5
      # data_lr = max(1e-7, data_lr)
  
    data_syn0 = sess.run([var_syn0])
    data_syn1neg = sess.run([var_syn1neg])
  
    if not os.path.exists(SAVEFOLDER):
      os.makedirs(SAVEFOLDER)
  
    saver.save(sess, SAVEPATH)

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
    train()
    distance()
