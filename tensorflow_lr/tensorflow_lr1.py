# https://mp.weixin.qq.com/s/SbpmbhNXEUm3uzIjt23ITw
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist.npz",one_hot=True)
mnist_data = mnist.train.images
mnist_target = mnist.train.labels
mnist_test_data = mnist.test.images
mnist_test_target = mnist.test.labels
print("训练数据的大小：",mnist_data.shape,"训练标签的大小：",mnist_target.shape,"测试数据的大小：",mnist_test_data.shape,"测试标签的大小：",mnist_test_target.shape)
def random_data(data,target):
    all_data = []
    all_target = []
    for item,target in zip(data,target):
        all_data.append(np.array(item))
        all_target.append(np.array(target))
    all_data = np.array(all_data)
    all_target = np.array(all_target)
    p = np.random.permutation(all_data.shape[0])
    all_data = all_data[p]
    all_target = all_target[p]
    train_data=all_data
    train_target=all_target
    return train_data,train_target
#同样的先给出我们的占位符
X = tf.compat.v1.placeholder(tf.float32,shape=[None,784])
Y = tf.compat.v1.placeholder(tf.float32,shape=[None,10])
Weight = tf.compat.v1.get_variable(name='weight',shape=[X.get_shape()[-1],10],initializer=tf.random_normal_initializer(0,1))
bias = tf.compat.v1.get_variable(name='bais',shape=[10],initializer=tf.constant_initializer(0.0))
y_pre = tf.matmul(X,Weight)+bias
#使用softmax激活函数
activation = tf.nn.softmax(y_pre)
#定义损失函数
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(activation),reduction_indices=1))
#采用梯度下降的方法
with tf.name_scope("train_op"):
    train_op = tf.compat.v1.train.GradientDescentOptimizer(0.05).minimize(loss)
#求1位置索引值 对比预测值索引与label索引是否一样，一样返回True
pred = tf.equal(tf.argmax(activation,1),tf.argmax(Y,1))
#tf.cast把True和false转换为float类型 0，1
#把所有预测结果加在一起求精度
accurate = tf.reduce_mean(tf.cast(pred,"float"))
#初始化所有参数
init = tf.compat.v1.global_variables_initializer()
batch = 200#批次
steps =  1000#迭代100次
indicator = 0
lis = []
los = []
accu = []
with tf.compat.v1.Session() as sess:
    #初始化
    sess.run(init)
    for i in range(steps):
        train_data,train_target = random_data(mnist_data,mnist_target)
        end_indicator = indicator + batch
        batch_x = train_data[indicator:end_indicator]
        batch_y = train_target[indicator:end_indicator]
        indicator = end_indicator
        if indicator>len(train_data):
            indicator = 0
        acc,l,_= sess.run(
            [accurate,loss,train_op]
            , feed_dict={X:batch_x, Y: batch_y})
        print("损失；",l)
        print("精度：",acc)
        print(i)
        lis.append(i)
        los.append(l)
        accu.append(acc)
    test_data, test_target = random_data(mnist_test_data,mnist_test_target)
    loss_val_test, acc_val_test = sess.run(
        [loss, accurate]
        , feed_dict={X: test_data, Y: test_target})
    print("测试集损失")
    print(loss_val_test)
    print("测试集精度")
    print(acc_val_test)
plt.plot(lis,los)
plt.show()
plt.plot(lis,accu)
plt.show()