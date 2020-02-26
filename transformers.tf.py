import tensorflow as tf
import tensorflow.logging
import tensorflow.random
import numpy as np
import numpy.linalg
# from tensorflow import keras
from collections import OrderedDict
import os
import random
import numba


tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)

SAVEFOLDER='models'
SAVEPATH='text0-transformer.bin'
INPUTPATH='text0'
IN_SENTENCE_LEN = 10
OUT_SENTENCE_LEN = 10
LEARNING_RATE=1e-2
NEPOCHS=10

with open(INPUTPATH, "r") as f:
  CORPUS = f.read()
  CORPUS = [w for w in CORPUS.split() if w]
  CORPUSLEN = len(CORPUS)
  # print("corpus:\n|%s|" % corpus)
  # stable sort
  vocab = list(OrderedDict.fromkeys(CORPUS))
  print("vocab:\n|%s|" % vocab[:5])
  
  # map words to their index in the embedding array
  VOCAB2IX = {w: i for (i, w) in enumerate(vocab)}
  IX2VOCAB = dict(enumerate(vocab))
  # print("VOCAB2IX:\n%s" % VOCAB2IX)
  VOCABSIZE = len(vocab)


  CORPUSIXED = np.empty(CORPUSLEN, dtype=np.int32)
  for i in range(CORPUSLEN):
      CORPUSIXED[i] = VOCAB2IX[CORPUS[i]]

assert CORPUS is not None
assert VOCAB2IX is not None
assert IX2VOCAB is not None
assert VOCABSIZE is not None
assert CORPUSLEN is not None
assert CORPUSIXED is not None



@numba.jit(nopython=True, parallel=True)
def mkdata(): pass

# return linear interpolation of [val0 --- val1] as specified by t
def lerp(t, val0, val1):
    val0 = tf.cast(val0, dtype=tf.float32);
    val1 = tf.cast(val1, dtype=tf.float32)
    one_minus_t = tf.math.subtract(1.0, t)
    return tf.math.add(tf.math.multiply(val0, one_minus_t), 
                       tf.math.multiply(val1, t))

def differentiable_min(x):
    xfloor_NOT_DIFF = tf.round(x)
    xfloor_diff = (x - (tf.stop_gradient(x) - xfloor_NOT_DIFF))
    return tf.cast(xfloor_diff, dtype=tf.int32)


# BUILD THE NEXTWORK
# ===================================

# input sentence, one-hot encoding of sentence
PH_INPUT_SENTENCE = tf.placeholder(tf.int32, (IN_SENTENCE_LEN), name="ph_input_sentence")
PH_OUTPUT_SENTENCE = tf.placeholder(tf.int32, (OUT_SENTENCE_LEN), name="ph_output_sentence")

# embedding matrices per layer
EMBEDMS = []
EMBEDSIZES = [VOCABSIZE, 32, 16, VOCABSIZE]

for i in range(len(EMBEDSIZES)-1):
  v = 1.0 / EMBEDSIZES[i] # initialization value
  init = tf.random.uniform([EMBEDSIZES[i], EMBEDSIZES[i+1]], 
                           minval=-v,
                           maxval=v)
  EMBEDMS.append(tf.Variable(init, dtype=tf.float32, name="embedm-%s" % (i, )))

print("EMBEDMS: %s" % EMBEDMS)


index_ixs = PH_INPUT_SENTENCE
for i in range(0, len(EMBEDSIZES)-2):
    # embed_ixs: SLEN x 1
    # embed: SLEN x EMBEDSIZES[i+1]
    embed = tf.gather(EMBEDMS[i], index_ixs, 
                         name="embed-%s" % i)
    # QKV^T : EMBEDSIZES[i+1] x EMBEDSIZES[i+1]
    mixer = tf.Variable(tf.random.normal([EMBEDSIZES[i+1], EMBEDSIZES[i+1]],
                                         name="mix-%s" % i))

    # get indexes into next layer
    # (SLEN X EMBEDSIZES[i+1]) x (EMBEDSIZES[i+1] x EMBEDSIZES[i+1]) = (SLEN x EMBEDSIZES[i+1])
    # ensure that entires of out are in the range [0..EMBEDSIZES[i+1]-1]
    print("embed: %s | mixer: %s" % (embed, mixer))
    embed = tf.nn.relu(tf.matmul(embed, mixer))

    # now reduce to get SLENx1
    # (SLEN X EMBEDSIZES[i+1]) x (EMBEDSIZES[i+1] x 1) = SLEN x 1
    reducer = tf.Variable(tf.random.normal([EMBEDSIZES[i+1], 1], 
                                           name="red-%s" % i))

    # embed = SLEN x 1
    embed = tf.math.sigmoid(tf.matmul(embed, reducer))
    print("===embed: %s=====" % embed)
    # reshape (SLENx1) to (SLEN)
    embed = tf.reshape(embed, [IN_SENTENCE_LEN])
    index_ixs = differentiable_min(lerp(embed, 0, EMBEDSIZES[i+2]))
    print("index_ixs: %s" % index_ixs)

# final output empedding is the final embed
embed_out = index_ixs
# we now have embed_ixs as the final embedding. We need to extract an
output_predicted = tf.cast(embed_out, dtype=tf.float32)

loss = \
    tf.math.squared_difference(output_predicted, 
                               tf.cast(PH_OUTPUT_SENTENCE, dtype=tf.float32))
loss = tf.math.reduce_sum(loss)
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# tf.keras.utils.plot_model(model, show_shapes=True)

def try_sample_sentence(sess):
    i = 0
    in_end = i + IN_SENTENCE_LEN
    out_start = in_end + 1; out_end = out_start + OUT_SENTENCE_LEN
    feed = {
        PH_INPUT_SENTENCE: CORPUSIXED[i:in_end],
        PH_OUTPUT_SENTENCE: CORPUSIXED[out_start:out_end]
    }
    lossval, out = sess.run([loss, embed_out], feed_dict=feed)

    print("out: %50s" % out)
    truth = " ".join(CORPUS[i:in_end]) + "|" + " ".join(CORPUS[out_start:out_end])
    print("ground truth: %50s" % truth)
    predict = " ".join([IX2VOCAB[out[i]] for i in range(OUT_SENTENCE_LEN)])
    print("prediction: %50s" % predict)


def epoch(curepoch, sess):
    i = 0
    N = len(CORPUSIXED) - max(IN_SENTENCE_LEN, OUT_SENTENCE_LEN)
    while i < N:
      in_end = i + IN_SENTENCE_LEN
      out_start = in_end + 1; out_end = out_start + OUT_SENTENCE_LEN

      feed = {
          PH_INPUT_SENTENCE: CORPUSIXED[i:in_end],
          PH_OUTPUT_SENTENCE: CORPUSIXED[out_start:out_end]
      }

      lossval, _ = sess.run([loss, OPTIMIZER], feed_dict=feed)

      try_sample_sentence(sess)


      i += 1
      print("epoch: %10s | i: %4s/%4s | loss: %20.5f" % \
            (curepoch, i, N, lossval))

def train():
  saver = tf.train.Saver()
  with tf.Session() as sess:
    global_init = tf.global_variables_initializer()
    sess.run(global_init)
    print("making data...")
    print("done.")
    for i in range(NEPOCHS):
        print("===epoch: %s===" % i)
        epoch(i, sess)
  
    if not os.path.exists(SAVEFOLDER):
      os.makedirs(SAVEFOLDER)
  
    saver.save(sess, SAVEPATH)


train()

