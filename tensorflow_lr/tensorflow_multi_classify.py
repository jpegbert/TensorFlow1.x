
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 加载mnist数据集
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
print("Mnist数据集加载成功！")


# 构建模型计算图
x = tf.placeholder(dtype=float, shape=[None,784])
y = tf.placeholder(dtype=float, shape=[None,10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
actv = tf.nn.softmax(tf.matmul(x, w)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))
learning_ratio = 0.01
optm = tf.train.GradientDescentOptimizer(learning_ratio).minimize(cost)

# 预测值
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# 精确值
accr = tf.reduce_mean(tf.cast(pred, "float"))
# 初始化
init = tf.global_variables_initializer()

# 开始训练模型
training_epochs = 50
batch_size = 100
display_step = 5

with tf.Session() as sess:
    sess.run(init)
    # 小批量梯度下降算法优化
    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            print("batch_ys:", batch_ys)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            feeds = {x: batch_xs, y: batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds)
        # display
        if epoch % display_step == 0:
            feeds_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: mnist.test.images, y: mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"%(epoch,training_epochs,avg_cost,train_acc,test_acc))
print("DONE")



