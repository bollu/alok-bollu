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
import math

# TODO: note that currently, we always have [SENTENCELEN] keys, so our
# embeddings will always be [SENTENCELEN]. This is weird. We want
# this to be dynamic.
# make every _f a FF network?

tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)

SAVEFOLDER='models'
SAVEPATH='text0-transformer.bin'
INPUTPATH='text0'
IN_SENTENCE_LEN = 5
OUT_SENTENCE_LEN = 5
LEARNING_RATE=1e-4
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

# the red boy was shot furiously

# the red boy        | VOCAB--E1--->32 --attn ---*-->E2--->16--attn----*
#                                                |                     |
#                                                ff                    ff
#                                                |                     |
#                                                v                     v
# was shot furiously | VOCAB--E1--->32 --attn-->[L2]-E2-->16----attn--->L


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
    xfloor_NOT_DIFF = tf.floor(x)
    xfloor_diff = (x - (tf.stop_gradient(x) - xfloor_NOT_DIFF))
    return tf.cast(xfloor_diff, dtype=tf.int32)

def cross_entropy(pred, true):
    eps = 1e-4
    clipped_pred = tf.clip_by_value(pred, eps, 1-eps)
    return -tf.reduce_sum(true * tf.log(clipped_pred))

def l2(pred, true):
    return tf.reduce_sum(tf.math.squared_difference(pred, true))


# BUILD THE NEXTWORK
# ===================================

# input sentence, one-hot encoding of sentence
PH_INPUT_SENTENCE = tf.placeholder(tf.int32, (IN_SENTENCE_LEN), name="ph_input_sentence")
PH_OUTPUT_SENTENCE = tf.placeholder(tf.int32, (OUT_SENTENCE_LEN), name="ph_output_sentence")

# embedding matrices per layer
EMBEDMS = []
EMBEDSIZES = [VOCABSIZE, VOCABSIZE//2, 512]

for i in range(len(EMBEDSIZES)-1):
  v = 1.0 / EMBEDSIZES[i] # initialization value
  init = tf.random.uniform([EMBEDSIZES[i], EMBEDSIZES[i+1]], 
                           minval=-v,
                           maxval=v)
  EMBEDMS.append(tf.Variable(init, dtype=tf.float32, name="embedm-%s" % (i, )))

print("EMBEDMS: %s" % EMBEDMS)

key = PH_INPUT_SENTENCE
mix_fns = []; reduce_fns = []
input_embeds = []
for i in range(0, len(EMBEDSIZES)-2):
    # embed_ixs: SLEN x 1
    # embed: SLEN x EMBEDSIZES[i+1]
    embed = tf.gather(EMBEDMS[i], key, 
                         name="embed-%s" % i)
    # QKV^T : EMBEDSIZES[i+1] x EMBEDSIZES[i+1]
    mixer = tf.Variable(tf.random.normal([EMBEDSIZES[i+1], EMBEDSIZES[i+1]],
                                         name="mix-%s" % i))
    mixer = 0.5 * (mixer + tf.transpose(mixer))
    mix_fns.append(mixer)

    # get indexes into next layer
    # (SLEN X EMBEDSIZES[i+1]) x (EMBEDSIZES[i+1] x EMBEDSIZES[i+1]) = (SLEN x EMBEDSIZES[i+1])
    # ensure that entires of out are in the range [0..EMBEDSIZES[i+1]-1]
    print("embed: %s | mixer: %s" % (embed, mixer))
    embed = tf.nn.relu(tf.matmul(embed, mixer))
    input_embeds.append(embed)

    # now reduce to get SLENx1
    # (SLEN X EMBEDSIZES[i+1]) x (EMBEDSIZES[i+1] x 1) = SLEN x 1
    reducer = tf.Variable(tf.random.normal([EMBEDSIZES[i+1], 1], 
                                           name="red-%s" % i))
    reduce_fns.append(reducer)
    # embed = SLEN x 1
    embed = lerp(tf.math.sigmoid(tf.matmul(embed, reducer)), 
                 0, EMBEDSIZES[i+2]-1)
    # reshape (SLENx1) to (SLEN)
    embed = tf.reshape(embed, [IN_SENTENCE_LEN])
    print("===embed: %s=====" % embed)
    key = tf.cast(tf.stop_gradient(tf.floor(embed)), dtype=tf.int32) # differentiable_min(lerp(embed, 0, EMBEDSIZES[i+2]))

output_embeds = []
output_keys = []
for i in range(0, len(EMBEDSIZES)-2):
    # embed_ixs: SLEN x 1
    # embed: SLEN x EMBEDSIZES[i+1]
    embed = tf.gather(EMBEDMS[i], key, 
                         name="embed-%s" % i)
    # get indexes into next layer
    # (SLEN X EMBEDSIZES[i+1]) x (EMBEDSIZES[i+1] x EMBEDSIZES[i+1]) = (SLEN x EMBEDSIZES[i+1])
    # ensure that entires of out are in the range [0..EMBEDSIZES[i+1]-1]
    embed = tf.nn.relu(tf.matmul(embed, mix_fns[i]))
    output_embeds.append(embed)

    # now reduce to get SLENx1
    # (SLEN X EMBEDSIZES[i+1]) x (EMBEDSIZES[i+1] x 1) = SLEN x 1
    # embed = SLEN x 1
    embed = lerp(tf.math.sigmoid(tf.matmul(embed, reduce_fns[i])), 
                 0, EMBEDSIZES[i+2]-1)

    # reshape (SLENx1) to (SLEN)
    embed = tf.reshape(embed, [IN_SENTENCE_LEN])
    # TODO: do I need the stop_gradient?
    key = tf.cast(tf.stop_gradient(tf.floor(embed)),dtype=tf.int32) # differentiable_min(lerp(embed, 0, EMBEDSIZES[i+2]))
    output_keys.append(key)
    

# go from input embedding at level i -> output embedding at level i
predicted_out_states = []

loss = tf.constant(0.)
for i in range(len(EMBEDSIZES)-2):
    in2out_embed_fn = \
            tf.Variable(tf.random.normal([EMBEDSIZES[i+1], EMBEDSIZES[i+1]],
                                         dtype=tf.float32,
                                         name="in2out_embed_fn-%s" % i))
    # make this [1 x EMBEDSIZES[i+1]] so I can multiply it
    out_predict = tf.matmul(input_embeds[i], in2out_embed_fn, name="out-predict-%s"%i)
    predicted_out_states.append(out_predict)
    curloss = l2(out_predict, output_embeds[i])
    # TODO: squared difference is retarded as fuck, use cross entropy + epislon
    loss = tf.add(loss, curloss)


# decoder that uncompresses the latent space to create a key that
# matches where I came from.
in2out_keys = []

# predicted output keys for the SAME EMBED SIZE.
# ie, predict the key generated for output at layer "i" from input
# at layer "i"
predicted_out_keys = []
for i in range(len(EMBEDSIZES)-2):
    # input_embeds[i]: SLEN x EMBEDSIZES[i+1]
    # key: SLEN x 1
    # probably need a g to hadamard product with first, otherwise
    # no mixing is going to happen :/
    # alternatively, attn works.
    # f : (EMBEDSIZES[i+1] x 1)
    in2out_key_fn = \
            tf.Variable(tf.random.normal([EMBEDSIZES[i+1], 1],
                                          name="in2out-key-f-%s"%i))
    # TODO: non-linearity
    # (SLEN x EMBEDSIZES[i+1]) @ (EMBEDSIZES[i+1] x 1) = SLEN x 1
    # out_key_predict = tf.matmul(input_embeds[i], in2out_key_fn)
    out_key_predict = tf.matmul(predicted_out_states[i], in2out_key_fn)

    out_key_predict = tf.sigmoid(out_key_predict)
    # lerp it to fit in the other key space
    out_key_predict = lerp(out_key_predict, 0, EMBEDSIZES[i]-1)
    predicted_out_keys.append(out_key_predict)
    curloss = cross_entropy(out_key_predict, 
                            tf.cast(output_keys[i], dtype=tf.float32))
    loss = tf.add(loss, curloss)


OPTIMIZER = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# need beam decoding here. 
def try_sample_sentence(sess):
    i = 0
    in_end = i + IN_SENTENCE_LEN
    out_start = in_end + 1; out_end = out_start + OUT_SENTENCE_LEN
    feed = {
        PH_INPUT_SENTENCE: CORPUSIXED[i:in_end],
        PH_OUTPUT_SENTENCE: CORPUSIXED[out_start:out_end]
    }
    lossval, out_key = sess.run([loss] + [predicted_out_keys[0]], feed_dict=feed)

    out_text = [IX2VOCAB[math.floor(out_key[i])]  \
                for i in range(OUT_SENTENCE_LEN)]

    truth = " ".join(CORPUS[i:in_end]) + "|" + \
            " ".join(CORPUS[out_start:out_end])
    print("ground truth: %50s ;" % truth, end='')
    print("out: %40s ;" % out_text)
    print("out: %40s ;" % out_key)
    print("loss: %s" % float(lossval))


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


      print("HACK: not incrementing!")
      #i += 1
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

