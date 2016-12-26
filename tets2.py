from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.mul(a, b)


print("%f" % tf.Session().run(tf.pow(a,b), feed_dict={a:2, b:2}))

with tf.Session() as sess:
    print("%f " % sess.run(y, feed_dict={a: 1, b: 2}))



################################

X = tf.placeholder("float")
Y = tf.placeholder("float")
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33
w = tf.Variable(0.0, name="weights")
y_model = tf.mul(X, w)

cost = tf.square(Y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))


    plt.plot(trX, trY, 'ro', label='Test data')
    plt.plot(trX, sess.run(w) * trX, label='Fitted line')
    plt.legend()
    plt.show()
    plt.savefig("test_2_test.png")