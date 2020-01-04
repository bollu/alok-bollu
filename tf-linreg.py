import tensorflow as tf
import numpy as np
import sys
import os

# saves
SAVEFOLDER = 'saves/'
SAVENAME = 'linalg-model'
SAVEFILEPATH = os.path.join(SAVEFOLDER, SAVENAME)

# meta variables / hyperparameters
LEARNING_RATE = 0.5
NUM_TRANING_STEPS = 10000


# model variables
M = 500
C = 42

NINPUT = 1
NOUTPUT = NINPUT


def mk_input():
    xs = np.random.normal(size=(NINPUT, 1))
    ys = xs * M + C

    return (xs, ys)


def mk_vars():
    var_weights = tf.Variable(tf.random_normal([NOUTPUT, NINPUT]), name="xvar")
    var_biases = tf.Variable(tf.random_normal([NOUTPUT, 1]), name="yvar")

    return (var_weights, var_biases)


def mk_placeholders():
    x = tf.placeholder(tf.float32, (NINPUT, 1), name="xinput")
    y = tf.placeholder(tf.float32, (NOUTPUT, 1), name="yinput")

    return (x, y)


def mk_nn(ph_x, var_weights, var_biases):
    out = tf.add(tf.matmul(var_weights, ph_x, name="weightedx"),
                 var_biases, name="biasedx")
    # out = tf.nn.relu(layer)

    return out


def mk_cost(ph_y, var_y):
    errs = tf.losses.absolute_difference(labels=ph_y,
                                         predictions=var_y)
    return errs


def mk_optimiser(var_cost, learning_rate):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(var_cost)
    return optimizer


def run_save_vars(saver, sess, savefolder, savepath):
    print("*** saving data...")

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    saver.save(sess, savepath)


# if the savefolder directory exists, pull data from it
def run_restore_vars(saver, sess, savefolder, savepath):
    if os.path.exists(savefolder):
        should_restore = input("restore?>")

        if should_restore.lower() == "y":
            saver.restore(sess, savepath)

            for var in tf.global_variables():
                varval = sess.run([var])
                print("%s:\n%s\n%s\n" %
                      (var.name, "-" * len(var.name), varval))

            should_use_restore = input("use restored?>")
            if should_use_restore.lower() == "y":
                return

    print("*** unable to restore!")
    global_init = tf.global_variables_initializer()
    sess.run(global_init)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        (var_weights, var_biases) = mk_vars()
        (ph_x, ph_y) = mk_placeholders()

        var_y = mk_nn(ph_x, var_weights, var_biases)

        var_cost = mk_cost(ph_y, var_y)
        optimizer = mk_optimiser(var_cost, LEARNING_RATE)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            run_restore_vars(saver, sess, SAVEFOLDER, SAVEFILEPATH)

            for i in range(NUM_TRANING_STEPS):
                (xs, ys) = mk_input()
                nn_out_y, weights, biases, cost, _ = \
                    sess.run([var_y, var_weights,
                              var_biases, var_cost, optimizer],
                             feed_dict={ph_x: xs, ph_y: ys})

                if i % 200 == 0:
                    run_save_vars(saver, sess, SAVEFOLDER, SAVEFILEPATH)

                if i % 50 == 0:
                    print("%s * %s + %s = %s :: %s" %
                          (xs, weights, biases, nn_out_y, ys))
                    print("cost: %s" % cost)

            run_save_vars(saver, sess, SAVEFOLDER, SAVEFILEPATH)

            weights = sess.run(var_weights)
            biases = sess.run(var_biases)

            print("final output\n%s" % ("-" * 10))
            print("weights: %s\nbiases: %s" % (weights, biases))
