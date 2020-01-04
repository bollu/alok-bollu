import tensorflow as tf
from tensorflow import keras
import os
import random
from collections import OrderedDict
import numpy as np
import numpy.linalg


SAVEFOLDER='models'
SAVEPATH='text0.bin'
INPUTPATH='text0'
EMBEDSIZE = 10
WINDOWSIZE = 8
NEGSAMPLES = 15
LEARNING_RATE=1e-5
NEPOCHS=15
BATCHSIZE=10000

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


var_syn0 = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn0")
var_syn1neg = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn1neg")

ph_fixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_fixs")
ph_cixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_cixs")
ph_labels = tf.placeholder(tf.float32, (BATCHSIZE, ), name="ph_labels")

# loss = (label - (focus[fix] . ctx[cix])^2
var_syn0_cur = tf.gather(var_syn0, ph_fixs)
var_syn1neg_cur = tf.gather(var_syn1neg, ph_cixs)

var_dots = tf.reduce_sum(tf.multiply(var_syn0_cur, var_syn1neg_cur))
# loss = tf.norm(tf.math.sub(ph_label, d), name="loss")
var_losses = tf.squared_difference(ph_labels, var_dots, name="losses")
# add up all the losses
var_loss = tf.reduce_sum(var_losses)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(var_loss)

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


def epoch(curepoch, sess, data_fixs, data_cixs, data_labels):
  for ixf in range(CORPUSLEN):
    l = max(0, ixf - WINDOWSIZE)
    r = min(CORPUSLEN - 1, ixf + WINDOWSIZE)
  
  
    
    # the fox [vc=jumps *vf=over* the] dog (vc.vf=1)
    for ixc in range(l, r):
      # variable[placeholder]
      data_fixs.append(VOCAB2IX[corpus[ixf]])
      data_cixs.append(VOCAB2IX[corpus[ixc]])
      data_labels.append(1)
  
    # vc=the fox [jumps *vf=over* the] dog (vc.vf = 0)
    for _ in range(NEGSAMPLES):
      data_fixs.append(VOCAB2IX[corpus[ixf]])
      data_cixs.append(random.randint(0, VOCABSIZE-1))
      data_labels.append(0)


    while len(data_labels) >= BATCHSIZE:
        assert len(data_labels) == len(data_cixs)
        assert len(data_labels) == len(data_fixs)
        # print("fix: %s | cix: %s | label: %s" % (data_fix, data_cix, data_label))
        loss, _ = sess.run([var_loss, optimizer], 
                           feed_dict={ph_fixs:data_fixs[:BATCHSIZE], 
                                      ph_cixs: data_cixs[:BATCHSIZE],
                                      ph_labels: data_labels[:BATCHSIZE]})

        print("epoch: %10s | loss: %20.2f | ixf: %10s | corpuslen: %10s | \
              %10.2f %%" % (curepoch, 
                            loss, 
                            ixf, 
                            CORPUSLEN, 
                            ixf / CORPUSLEN * 100.0))
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
    for i in range(NEPOCHS):
      print("===epoch: %s===" % i)
      epoch(i, sess, data_fixs, data_cixs, data_labels) 
  
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

    while True:
        w = input(">")
        if w == "exit": break
        if w not in VOCAB2IX: continue
        wix = VOCAB2IX[w]
        wv = syn0[wix, :]
        wv = wv / np.linalg.norm(wv)
        dots = np.dot(syn0, wv)
        ixs = np.argsort(dots)
        ixs = np.flip(ixs)
        for ix in ixs[:30]:
            print("%20s %4.2f" % (IX2VOCAB[ix], dots[ix]))

if __name__ == "__main__":
    train()
    distance()
