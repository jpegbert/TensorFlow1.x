import tensorflow as tf


"""
TensorFlow实现单层感知机详解
http://c.biancheng.net/view/1914.html
"""


def threshold(x):
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype=x.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


eta = 0.4 # learning rate
epsilon = 1e-3 # minimum accepted error
max_epoches = 100

T, F = 1., 0.
X_in = [
    [T, T, T, T],
    [T, T, F, T],
    [T, F, T, T],
    [T, F, F, T],
    [F, T, T, T],
    [F, T, F, T],
    [F, F, T, T],
    [F, F, F, T],
]
Y = [
    [T],
    [T],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F],
]

W = tf.Variable(tf.random_normal([4, 1], stddev=2, seed=0))
h = tf.matmul(X_in, W)
y_hat = threshold(h)
error = Y - y_hat
mean_error = tf.reduce_mean(tf.square(error))
dW = eta * tf.matmul(X_in, error, transpose_a=True)
train = tf.assign(W, W + dW)

init = tf.global_variables_initializer()
err = 1
epoch = 0
with tf.Session() as sess:
    sess.run(init)
    while err > epsilon and epoch < max_epoches:
        epoch += 1
        err, _ = sess.run([mean_error, train])
        print("epoch: {0} mean error: {1}".format(epoch, err))
    print("Training complete")
