import tensorflow as tf
import tensorflow.logging
import tensorflow.random
import numpy as np
import numpy.linalg
from tensorflow import keras
from collections import OrderedDict
import os
import random


tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)

SAVEFOLDER='models'
SAVEPATH='text0.bin'
INPUTPATH='text0'
EMBEDSIZE = 50
WINDOWSIZE = 8
NEGSAMPLES = 15
LEARNING_RATE=1e-3
NEPOCHS=10
BATCHSIZE=1000

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

assert VOCABSIZE is not None
assert CORPUSLEN is not None


VAL = 1.0 / EMBEDSIZE
var_syn0 = tf.Variable(tf.random.uniform([VOCABSIZE, EMBEDSIZE],
                                         minval=-VAL,
                                        maxval=VAL), name="syn0")
var_syn1neg = tf.Variable(tf.zeros([VOCABSIZE, EMBEDSIZE]), name="syn1neg")

ph_fixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_fixs")
ph_cixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_cixs")
ph_labels = tf.placeholder(tf.float32, (BATCHSIZE, ), name="ph_labels")

# loss = (label - (focus[fix] . ctx[cix])^2
var_syn0_cur = tf.gather(var_syn0, ph_fixs, name="syn0_cur")
var_syn1neg_cur = tf.gather(var_syn1neg, ph_cixs, name="syn1neg_cur")

var_dots = tf.reduce_sum(tf.multiply(var_syn0_cur, var_syn1neg_cur), axis=1, name="dots")

var_clipped_dots = tf.math.tanh(var_dots, "clipped_dots")
# loss = tf.norm(tf.math.sub(ph_label, d), name="loss")
var_losses = tf.squared_difference(ph_labels, var_clipped_dots, name="losses")
# add up all the losses
var_loss = tf.reduce_sum(var_losses, name="loss")

# learning rate
ph_lr = tf.placeholder(tf.float32, name="ph_lr")

optimizer = tf.train.AdamOptimizer(learning_rate=ph_lr).minimize(var_loss)


print("\n***NETWORK:***")
print("syn0: %s | syn1neg: %s" % (var_syn0, var_syn1neg))
print("syn0 cur: %s" % var_syn0_cur)
print("syn1neg cur: %s" % var_syn1neg_cur)
print("dots: %s" % var_dots)
print("clipped dots: %s" % var_clipped_dots)
print("losses: %s" % var_losses)
print("loss: %s" % var_loss)
print("***END NETWORK:***\n")

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


def epoch(curepoch, sess, data_fixs, data_cixs, data_labels, data_lr):
  for ixf in range(CORPUSLEN):
    l = max(0, ixf - WINDOWSIZE)
    r = min(CORPUSLEN - 1, ixf + WINDOWSIZE)
    
    # the fox [vc=|jumps| *vf=over* the] dog (vc.vf=1)
    for ixc in range(l, r):
      # variable[placeholder]
      data_fixs.append(VOCAB2IX[corpus[ixf]])
      data_cixs.append(VOCAB2IX[corpus[ixc]])
      data_labels.append(1)
  
    # vc=|the| fox [jumps *vf=over* the] dog (vc.vf = 0)
    for _ in range(NEGSAMPLES):
      ixrand = random.randint(0, CORPUSLEN-1)
      if l <= ixrand <= r: continue # reject words inside window
      data_fixs.append(VOCAB2IX[corpus[ixf]])
      data_cixs.append(VOCAB2IX[corpus[ixrand]])
      data_labels.append(0)


    while len(data_labels) >= BATCHSIZE:
        assert len(data_labels) == len(data_cixs)
        assert len(data_labels) == len(data_fixs)
        # print("fix: %s | cix: %s | label: %s" % (data_fix, data_cix, data_label))
        loss, _ = sess.run([var_loss, optimizer], 
                           feed_dict={ph_fixs:data_fixs[:BATCHSIZE], 
                                      ph_cixs: data_cixs[:BATCHSIZE],
                                      ph_labels: data_labels[:BATCHSIZE],
                                      ph_lr: data_lr})

        print("epoch: %10s | loss: %20.5f | lr: %20.8f | %10.2f %%" % (curepoch, 
                            loss,
                            data_lr,
                            ((curepoch + (ixf / CORPUSLEN)) / NEPOCHS)*100))
        # data_lr = data_lr * (1.0 - 1e-6)
        # data_lr = max(1e-7, max(LEARNING_RATE * 1e-5, data_lr))

        data_labels = data_labels[BATCHSIZE:]
        data_fixs = data_fixs[BATCHSIZE:]
        data_cixs = data_cixs[BATCHSIZE:]


def train():
  saver = tf.train.Saver()
  with tf.Session() as sess:
    global_init = tf.global_variables_initializer()
    sess.run(global_init)
    data_fixs = []
    data_cixs = []
    data_labels = []
    data_lr = LEARNING_RATE
    for i in range(NEPOCHS):
      print("===epoch: %s===" % i)
      epoch(i, sess, data_fixs, data_cixs, data_labels, data_lr) 
  
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
