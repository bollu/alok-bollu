import tensorflow as tf
import numpy as np
import sys
import os

LEARNINGRATE=1e-1
SAVEPATH='linreg.mdl'
NDATAPOINTS=5000

# y = mx + c
# (m, c) -> model params that need to be learnt: tf.Variable
# (x, y) -> input data that is fed during training: tf.Placeholder


m = tf.Variable(tf.random_normal(()), name="m")
c = tf.Variable(tf.random_normal(()), name="c")

ph_x = tf.placeholder(tf.float32, name="x")
ph_y = tf.placeholder(tf.float32, name="y")

y_predict = tf.add(tf.multiply(m, ph_x), c)

loss = tf.math.squared_difference(y_predict, ph_y)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNINGRATE).minimize(loss)


xs = np.random.rand(NDATAPOINTS)
ys = 10 * xs + 20

# what GPUS/CPUs/TPUS am I connected to, where does all the data of
# all my variables live.
with tf.Session() as sess:
  # first run
  global_init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  # re-loading data from a checkpoint

  saver.restore(sess, SAVEPATH)

  for i in range(NDATAPOINTS):
      sess.run([optimizer], feed_dict={ph_x: xs[i], ph_y: ys[i]})
      loss_val = sess.run([loss], feed_dict={ph_x: xs[i], ph_y: ys[i]})

      print("current loss: %s" % loss_val)

  final_m, final_c = sess.run([m, c])

  print("learnt m: %s | c: %s" % (final_m, final_c))

  saver.save(sess, SAVEPATH)
