%tensorflow_version 1.x
import tensorflow as tf
import tensorflow.logging
import tensorflow.random
import numpy as np
import numpy.linalg
from tensorflow import keras
from collections import OrderedDict, Counter
import os
import sys
import random
import numba

# TODO: make data loading from corpus faster (Siddharth)
# TODO: run on text8 and generate a vector.bin file according to word2vec
#       convention so we can run compute-accuracy and repl (Siddharth)
# TODO: include all poplar tricks such as frequency based discarding (Souvik)(Done)
# TODO: adapt the code to GA (Souvik)

# Features not added - Binary Huffman tree(If we use hierarchial softmax) and hash
# Need help with intialisation(Given how they are intialised as random vectors, this would mean the inner product (A,B) will only give
# 0 if they share basis otherwise the magnitude of the vector B)

tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)

SAVEFOLDER='models'
SAVEPATH='text0.bin'
INPUTPATH='/content/drive/My Drive/text0'
EMBEDSIZE = 10
WINDOWSIZE = 4
NEGSAMPLES = 0
LEARNINGRATE=1e-3
NEPOCHS=4
BATCHSIZE=100
READ_VOCAB_FILE = 0
SAVE_VOCAB_FILE = 1
MIN_COUNT = 1
VOCAB_HASH_SIZE = 3000000
BINARY = True 

#Aim to create a array of instances of vocab class, should make handling of vocab removal easier
class VOCAB(self, cn, word, code, codelen):
  def __init__():
    self.cn = cn
    self.word = word
    self.code = code
    self.codelen = codelen


with open(INPUTPATH, "r") as f:
  corpus = f.read()
  corpus = [w for w in corpus.split() if w]
  
  # Count of each word in the corpus 
  VOCABCOUNT = Counter(corpus)
  # print("Count of each word:\n|%s|" % VOCABCOUNT)
  
  VOCABSIZE = len(list(set(corpus)))
  
  #Infrequent words are removed from corpus in case vocab is too big
  min_reduce = 1
  while VOCABSIZE > VOCAB_HASH_SIZE*0.7:
    corpus = [word for word in corpus if VOCABCOUNT[word] > min_reduce]
    VOCABSIZE = len(list(set(corpus)))
    min_reduce += 1

  #Sorting the vocab in ascending order
  sorted_corpus = sorted(corpus, key=Counter(corpus).get, reverse=False)
  vocab = list(OrderedDict.fromkeys(sorted_corpus))
  print("vocab:\n|%s|" % vocab)
  VOCABSIZE = len(vocab)

  #Freq based discarding
  for word in vocab:
    if VOCABCOUNT[word] < MIN_COUNT:
      vocab.remove(word)

  # map words to their index in the embedding array
  VOCAB2IX = {w: i for (i, w) in enumerate(vocab)}
  # print("VOCAB2IX:\n%s" % VOCAB2IX)

  CORPUSLEN = len(corpus)
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
var_syn0 = tf.Variable(tf.random.uniform([VOCABSIZE, EMBEDSIZE], minval=-VAL, maxval=VAL), name="syn0")

#var_syn1neg = tf.Variable(tf.random.uniform([VOCABSIZE, EMBEDSIZE], minval=-VAL, maxval=VAL), name="syn1neg")
var_syn1neg = tf.Variable(tf.zeros([VOCABSIZE, EMBEDSIZE]), name="syn1neg")

ph_fixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_fixs")
ph_cixs = tf.placeholder(tf.int32, (BATCHSIZE, ), name="ph_cixs")
ph_labels = tf.placeholder(tf.float32, (BATCHSIZE, ), name="ph_labels")
var_syn0_cur = tf.gather(var_syn0, ph_fixs, name="syn0_cur")
var_syn1neg_cur = tf.gather(var_syn1neg, ph_cixs, name="syn1neg_cur")
var_dots = tf.reduce_sum(tf.multiply(var_syn0_cur, var_syn1neg_cur), axis=1, name="dots")
var_losses_dot = tf.squared_difference(ph_labels, var_dots, name="losses_dot")
var_loss = tf.reduce_sum(var_losses_dot, name="loss_dot")
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNINGRATE).minimize(var_loss)


print("\n***NETWORK:***")
print("syn0: %s | syn1neg: %s" % (var_syn0, var_syn1neg))
print("syn0 cur: %s" % var_syn0_cur)
print("syn1neg cur: %s" % var_syn1neg_cur)
print("dots: %s" % var_dots)
print("losses_dot: %s" % var_losses_dot)
print("loss: %s" % var_loss)
print("***END NETWORK:***\n")

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program

@numba.jit(nopython=True, parallel=True)
def mkdata():
  fixs = np.empty(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.int32)
  cixs = np.random.randint(low=0,
                           high=CORPUSLEN-1,
                           size=CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1))
  labels = np.zeros(CORPUSLEN * (2*WINDOWSIZE + NEGSAMPLES + 1), dtype=np.float32)

  n = 0
  ixf = 0
  while ixf < CORPUSLEN:
    l = max(0, ixf - WINDOWSIZE)
    r = min(CORPUSLEN - 1, ixf + WINDOWSIZE)
    windowsize = r-l

    wix = l
    while wix < r:
        fixs[n] = corpusixed[ixf]
        cixs[n] = corpusixed[wix]
        labels[n] = 1
        wix += 1
        n += 1

    rix = 0
    while rix < NEGSAMPLES:
        fixs[rix] = corpusixed[ixf]
        rix += 1
        n += 1

    print((100.0 * n / (CORPUSLEN * (2 * WINDOWSIZE + NEGSAMPLES))))
    ixf += 1

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
    print("done. n: %10s" % (n, ))

    # print("\n***LLVM of mkdata:***")
    # for v, k in mkdata.inspect_llvm().items():
    #     print(v, k)
    # print("***end LLVM of mkdata:***\n")

    for i in range(NEPOCHS):
      print("\n===epoch: %s===" % i)
      epoch(i, sess, n, fixs, cixs, labels, data_lr) 

      # save file per epoch
      if not os.path.exists(SAVEFOLDER):
        os.makedirs(SAVEFOLDER)
      saver.save(sess, SAVEPATH)
  
    [data_syn0] = sess.run([var_syn0])
    [data_syn1neg] = sess.run([var_syn1neg])
    f = open("vector"+str(EMBEDSIZE)+".bin", "wb+")
    info_text = "{0} {1}\n".format(VOCABSIZE, EMBEDSIZE)
    f.write(info_text.encode('utf-8'))
    if BINARY:
      for word in vocab:
        text = word+" "
        f.write(text.encode('utf-8'))
        for x in data_syn0[VOCAB2IX[word],:]:
          f.write(bytes(x))
        f.write("\n".encode('utf-8'))
    else:
      for word in vocab:
        text = word + " " + " ".join(str(x) for x in data_syn0[VOCAB2IX[word],:]) +"\n"
        f.write(text.encode('utf-8'))
    f.close()
    
    f = open("negvector"+str(EMBEDSIZE)+".bin", "wb+")
    info_text = "{0} {1}\n".format(VOCABSIZE, EMBEDSIZE)
    f.write(info_text.encode('utf-8'))
    if BINARY:
      for word in vocab:
        text = word+" "
        f.write(text.encode('utf-8'))
        for x in data_syn0[VOCAB2IX[word],:]:
          f.write(bytes(x))
        f.write("\n".encode('utf-8'))
    else:
      for word in vocab:
        text = word + " " + " ".join(str(x) for x in data_syn1neg[VOCAB2IX[word],:]) +"\n"
        f.write(text.encode('utf-8'))
    
    f.close()

    if not os.path.exists(SAVEFOLDER):
      os.makedirs(SAVEFOLDER)
  
    saver.save(sess, SAVEPATH)

def repl():
  saver = tf.train.Saver()
  IX2VOCAB = {VOCAB2IX[w]:w for w in VOCAB2IX}
  with tf.Session() as sess:
    saver.restore(sess, SAVEPATH)
    [syn0] = sess.run([var_syn0])
    print("\nsyn0 shape: %s" % (syn0.shape, ))
    print(type(syn0))

    for i in range(VOCABSIZE):
        syn0[i, :] = syn0[i, :] / np.linalg.norm(syn0[i, :])

    while True:
        s = input(">")
        if s == "exit": break
        if len(s.split()) == 0: continue
        c = s.split()[0]
        if c == "dist":
            if len(s.split()) != 2: continue
            w = s.split()[1]
            if w not in VOCAB2IX: continue
            wix = VOCAB2IX[w]

            v = syn0[wix, :]
        elif c == "analogy":
            if len(s.split()) != 4: continue
            wa = s.split()[1]
            wb = s.split()[2]
            wx = s.split()[3]

            if wa not in VOCAB2IX: continue
            if wb not in VOCAB2IX: continue
            if wx not in VOCAB2IX: continue

            wa = VOCAB2IX[wa]
            wb = VOCAB2IX[wb]
            wx = VOCAB2IX[wx]

            v = syn0[wb, :] - syn0[wa, :]  + syn0[wx, :]
        else: continue

        dots = np.tanh(np.dot(syn0, v))
        print ("syn0:\n%s\nwv:\n%s\ndots:\n%s" % (syn0, v, dots))
        print("v len: %s" % (np.dot(v, v), ))
        ixs = np.argsort(dots)
        ixs = np.flip(ixs)
        for ix in ixs[:30]:
            print("%20s %10.5f" % (IX2VOCAB[ix], dots[ix]))

if __name__ == "__main__":
  train()
  repl()
