import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data

"""
TensorFlow逻辑回归处理MNIST数据集
http://c.biancheng.net/view/1909.html
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name="X")
y = tf.placeholder(tf.float32, [None, 10], name="Y")

W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10], name="b"))

with tf.name_scope("wx_b") as scope:
    y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

with tf.name_scope("cross-entropy") as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    tf.summary.scalar("cross-entropy", loss)

with tf.name_scope("Train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


init = tf.global_variables_initializer()
merged_summary_op = tf.summary.merge_all()

max_epochs = 10
batch_size = 128

with tf.Session() as sess:
    sess.run(init)
    summery_writer = tf.summary.FileWriter("graphs", sess.graph)
    # training
    for epoch in range(max_epochs):
        loss_avg = 0
        num_of_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_of_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, l, summary_str = sess.run([optimizer, loss, merged_summary_op], feed_dict={x: batch_xs, y: batch_ys})
            loss_avg += 1
            summery_writer.add_summary(summary_str, epoch * num_of_batch + 1) # add all summaries per batch
        loss_avg = loss_avg / num_of_batch
        print("Epoch {0}: Loss {1}".format(epoch, loss_avg))
    print("Done")
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))




