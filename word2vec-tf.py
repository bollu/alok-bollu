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
SAVEPATH='text0.bin'
INPUTPATH='text0'
EMBEDSIZE = 50
WINDOWSIZE = 8
NEGSAMPLES = 15
LEARNING_RATE=1e-2
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


  corpusixed = np.empty(CORPUSLEN, dtype=np.int32)
  for i in range(CORPUSLEN):
      corpusixed[i] = VOCAB2IX[corpus[i]]

assert VOCABSIZE is not None
assert CORPUSLEN is not None


VAL = 1.0 / EMBEDSIZE
syn0_init = np.random.random((VOCABSIZE, EMBEDSIZE)) - 0.5
for i in range(VOCABSIZE):
  syn0_init[i, :] = syn0_init[i, :] / np.linalg.norm(syn0_init[i, :])

# var_syn0 = tf.Variable(tf.random.uniform([VOCABSIZE, EMBEDSIZE],
#                                          minval=-VAL,
#                                         maxval=VAL), name="syn0")
# 
var_syn0 = tf.Variable(tf.constant(syn0_init, dtype=tf.float32,
                                   shape=(VOCABSIZE, EMBEDSIZE)), name="syn0")
# var_syn0 = tf.Variable(tf.random.normal([VOCABSIZE, EMBEDSIZE], mean=0, stddev=0.3  / EMBEDSIZE), name="syn0")
var_syn1neg = tf.Variable(tf.zeros([VOCABSIZE, EMBEDSIZE]), name="syn1neg")

ph_fixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_fixs")
ph_cixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_cixs")
ph_labels = tf.placeholder(tf.float32, (BATCHSIZE, ), name="ph_labels")

# loss = (label - (focus[fix] . ctx[cix])^2
var_syn0_cur = tf.gather(var_syn0, ph_fixs, name="syn0_cur")
var_syn1neg_cur = tf.gather(var_syn1neg, ph_cixs, name="syn1neg_cur")

var_dots = tf.reduce_sum(tf.multiply(var_syn0_cur, var_syn1neg_cur), axis=1, name="dots")

var_clipped_dots = tf.math.sigmoid(var_dots, "clipped_dots")
# var_clipped_dots = var_dots

# loss = tf.norm(tf.math.sub(ph_label, d), name="loss")
var_losses_dot = tf.squared_difference(ph_labels, var_clipped_dots, name="losses_dot")
# add up all the losses
var_loss_dot = tf.reduce_sum(var_losses_dot, name="loss_dot")

var_lensq = tf.reduce_sum(tf.multiply(var_syn0_cur, var_syn0_cur), axis=1, name="lensq")

var_regularize = tf.squared_difference(1.0, var_lensq, name="regularizations")
var_regularize = tf.multiply(var_regularize, REGULARIZATION_COEFF,
                             name="regularizations_scaled")
var_regularize = tf.reduce_sum(var_regularize, name="regularization_final")


# var_loss = tf.add(var_regularize, var_loss_dot, name="loss")
var_loss = var_loss_dot

# learning rate
ph_lr = tf.placeholder(tf.float32, name="ph_lr")

optimizer = tf.train.AdamOptimizer(learning_rate=ph_lr).minimize(var_loss)


print("\n***NETWORK:***")
print("syn0: %s | syn1neg: %s" % (var_syn0, var_syn1neg))
print("syn0 cur: %s" % var_syn0_cur)
print("syn1neg cur: %s" % var_syn1neg_cur)
print("dots: %s" % var_dots)
print("clipped dots: %s" % var_clipped_dots)
print("losses_dot: %s" % var_losses_dot)
print("loss_dot: %s" % var_loss_dot)
print("lensq: %s" % var_lensq)
print("regularize: %s" % var_regularize)
print("loss: %s" % var_loss)
print("***END NETWORK:***\n")

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


@numba.jit(nopython=True, parallel=True)
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
      # labels[n] = np.exp(-1.0 * DISTANCE_DOT_DECAY * np.fabs(ixc - ixf))
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
  
    # print((100.0 * n / (CORPUSLEN * (2 * WINDOWSIZE + NEGSAMPLES))))


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
                                    ph_labels: data_labels[i*BATCHSIZE:(i+1)*BATCHSIZE],
                                    ph_lr: data_lr})

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
    data_lr = LEARNING_RATE

    print("load data...\n")
    fixs, cixs, labels, n = mkdata()
    # fixs, cixs, labels = shuffledata(fixs, cixs, labels, n)
    print("done. n: %10s" % (n, ))

    print("\n***LLVM of mkdata:***")
    for v, k in mkdata.inspect_llvm().items():
        print(v, k)
    print("***end LLVM of mkdata:***\n")
    # raise RuntimeError("inspection")

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
    train()
    distance()
