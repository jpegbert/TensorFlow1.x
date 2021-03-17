# https://www.jianshu.com/p/cdabe06d3894

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    sample_per_class = int(sample_size / 2)
    X0 = np.random.multivariate_normal(mean, cov, sample_per_class)
    Y0 = np.zeros(sample_per_class)
    # print("X0:",X0)
    # print("Y0:",Y0)
    for ci, d in enumerate(diff):
        # print("-------------------ci,d---------------------:")
        # print("ci:",ci)
        # print("d:",d)

        # print("--------------------X1:y1------------------------:")
        X1 = np.random.multivariate_normal(mean + d, cov, sample_per_class)
        Y1 = (ci + 1) * np.ones(sample_per_class)
        # print("X1:",X1)
        # print("Y1:",Y1)
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
        # print("---------------------X0，Y0-------------------------:")

        # print("X0:",X0)
        # print("Y0:",Y0)
        # print("--------------------X，Y--------------------------:")

    if regression == False:
        class_id = [Y1 == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_id), dtype=np.float32)
        X, Y = shuffle(X0, Y)
    else:

        X, Y = shuffle(X0, Y0)
    # print("X:",X)
    # print("Y:",Y)
    # print("------------------------------------------------:")
    return X, Y


np.random.seed(10)
num_classes=2

mean=np.random.randn(num_classes)
cov=np.eye(num_classes)
X,Y=generate(1000,mean,cov,[3.0],True)

#print("mean:",mean)
#print("cov:",cov)
# print("X:", X)
# print("Y:", Y)
print("type(X):", type(X))

colors = ['r' if l==0 else 'b' for l in Y[:]]
plt.scatter(X[:,0],X[:,1],c=colors)
plt.show()



# 定义维度
lab_dim = 1
input_dim = 2
# print(input_dim)

# 定义占位符数据
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])
# 定义变量
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim], name="bias"))
# 输出数据
output = tf.nn.sigmoid(tf.matmul(input_features, W) + b)
# 交叉熵
coross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output))
# 误差
ser = tf.square(input_labels - output)
# 损失函数
loss = tf.reduce_mean(coross_entropy)
# 误差均值
err = tf.reduce_mean(ser)
# 优化器
optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)

maxEpochs = 50
minibatchSize = 25

with tf.Session() as sess:
    # 初始化所有变量与占位符
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr = 0
        # 对于每一个batch
        for i in range(np.int32(len(Y) / minibatchSize)):
            # 取出X值
            x1 = X[i * minibatchSize:(i + 1) * minibatchSize, :]
            # 取出Y值
            y1 = np.reshape(Y[i * minibatchSize:(i + 1) * minibatchSize], [-1, 1])
            # 改变y的数据结构，变成tensor数据
            tf.reshape(y1, [-1, 1])

            # 对相关结果进行计算
            _, lossval, outputval, errval = sess.run([train, loss, output, err],
                                                     feed_dict={input_features: x1, input_labels: y1})

            # 计算误差和
            sumerr = sumerr + errval

        print("epoch:", epoch)
        print("cost=", lossval, "err=", sumerr)

    # 结果可视化
    train_X, train_Y = generate(100, mean, cov, [3.0], True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)
    # plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    # plt.colorbar()

    #    x1w1+x2*w2+b=0
    #    x2=-x1* w1/w2-b/w2
    x = np.linspace(-1, 8, 200)
    y = -x * (sess.run(W)[0] / sess.run(W)[1]) - sess.run(b) / sess.run(W)[1]
    plt.plot(x, y, label='Fitted line')
    plt.legend()
    plt.show()

