import tensorflow as tf
from tensorflow import keras

import random
INPUTPATH='text0'
EMBEDSIZE = 300
WINDOWSIZE = 2
NEGSAMPLES = 5
LEARNING_RATE=1e-3
NEPOCHS=100

with open(INPUTPATH, "r") as f:
  corpus = f.read()
  corpus = [w for w in corpus.split() if w]
  print("corpus:\n|%s|" % corpus)
  vocab = set(corpus)
  
  # map words to their index in the embedding array
  VOCAB2IX = {w: i for (i, w) in enumerate(vocab)}
  print("VOCAB2IX:\n%s" % VOCAB2IX)
  VOCABSIZE = len(vocab)

assert VOCABSIZE is not None


var_syn0 = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn0")
var_syn1neg = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn1neg")

ph_fix = tf.placeholder(tf.int32, name="ph_fix")
ph_cix = tf.placeholder(tf.int32, name="ph_cix")
ph_label = tf.placeholder(tf.float32, name="ph_label")

# loss = (label - (focus[fix] . ctx[cix])^2
var_d = tf.tensordot(var_syn0[ph_fix, :], var_syn1neg[ph_cix, :], 1)
# loss = tf.norm(tf.math.sub(ph_label, d), name="loss")
var_loss = tf.norm(ph_label - var_d, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(var_loss)

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


def epoch(sess):
  i = 0
  for ixf in range(len(corpus)):
    l = max(0, ixf - WINDOWSIZE)
    r = min(len(corpus) - 1, ixf + WINDOWSIZE)
  
    # word(fox) -> index (10) -> vector [1; -1; 42; -42]
    data_fix = VOCAB2IX[corpus[ixf]]
  
    
    # the fox [vc=jumps *vf=over* the] dog (vc.vf=1)
    for ixc in range(l, r):
      # variable[placeholder]
      data_cix = VOCAB2IX[corpus[ixc]]
      data_label = 1
  
    # vc=the fox [jumps *vf=over* the] dog (vc.vf = 0)
    for _ in range(NEGSAMPLES):
      data_cix = random.randint(0, VOCABSIZE-1)
      data_label = 0
    
    # print("fix: %s | cix: %s | label: %s" % (data_fix, data_cix, data_label))
    loss, _ = sess.run([var_loss, optimizer], 
	    feed_dict={ph_fix:data_fix, ph_cix: data_cix, ph_label: data_label})

    if i % 4 == 0: print("loss: %4.2f" % (loss, ))
    i += 1

with tf.Session() as sess:
  global_init = tf.global_variables_initializer()
  sess.run(global_init)
  for i in range(NEPOCHS):
    print("===epoch: %s===" % i)
    epoch(sess) 

  data_syn0 = sess.run([var_syn0])
  data_syn1neg = sess.run([var_syn1neg])

  # TASK 1. Pull out the data from the session, and _print it_. Maybe try and
  # implement distance()
  
  # print distance of fox from all other words, ordered by ascending order (Dot
  # product / cosine distance)
  distance('fox', data_syn0)

  # quick - fox + dog == ? print best candidates for this
  # Fox :  quick :: fox : ? == (quick - fox) + fox = quick
  analogy('fox', 'quick', 'dog' data_syn0)
  
  # TASK 2. copy and understand (plz plz plz) the data saving/loading code, and
  # save the learnt word vectors.

  # TASK 3. make this parallel: use multiple indeces and
  # multipl labels _in parallel_
