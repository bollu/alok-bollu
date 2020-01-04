import tensorflow as tf
from tensorflow import keras

import random
INPUTPATH='text0'
EMBEDSIZE = 300
WINDOWSIZE = 8
NEGSAMPLES = 25
LEARNING_RATE=1e-3

with open(INPUTPATH, "r") as f:
  corpus = f.read()
  corpus = [w for w in corpus.split() if w]
  vocab = set(corpus)
  
  # map words to their index in the embedding array
  VOCAB2IX = {w: i for (i, w) in enumerate(vocab)}
  VOCABSIZE = len(vocab)

assert VOCABSIZE is not None


var_syn0 = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn0")
var_syn1neg = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn1neg")

ph_fix = tf.placeholder(tf.int32, name="ph_fix")
ph_cix = tf.placeholder(tf.int32, name="ph_cix")
ph_label = tf.placeholder(tf.int32, name="ph_label")

# loss = (label - (focus[fix] . ctx[cix])^2
d = tf.dot(tf.slice(var_syn0, ph_fix), tf.slice(var_syn1neg, ph_cix))
loss = tf.norm(tf.sub(ph_label, d)), name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


with tf.Session() as sess:
  for ixf in range(len(corpus)):
    l = max(0, ixf - WINDOWSIZE)
    r = min(len(corpus) - 1, ixf + WINDOWSIZE)
  
    # word(fox) -> index (10) -> vector [1; -1; 42; -42]
    data_fix = VOCAB2IX[corpus[ixf]]
  
    
    # the fox [vc=jumps *vf=over* the] dog (vc.vf=1)
    for ixc in range(l, r+1):
      # variable[placeholder]
      data_cix = VOCAB2IX[corpus[ixc]]
      data_label = 1
  
    # vc=the fox [jumps *vf=over* the] dog (vc.vf = 0)
    for _ in range(NEGSAMPLES):
      data_cix = syn1neg[random.randint(0, len(corpus)]
      data_label = 0
    
    loss, _ = sess.run([loss, optimizer], 
	    feed_dict={ph_fix:data_fix, ph_cix: data_cix, ph_label: data_label})

    if i % 200 == 0: print("loss: %4.2f" % loss)
