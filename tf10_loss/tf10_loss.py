import tensorflow as tf


"""
损失函数
"""


def standard_regression():
    """
    标准线性回归
    只有一个输入变量和一个输出变量
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")

    w0 = tf.Variable(0.0)
    w1 = tf.Variable(0.0)

    Y_hat = X * w1 + w0

    # loss function
    loss = tf.square(Y - Y_hat, name="loss")


def duoyuan_regression():
    """
    多元线性回归
    有多个输入变量和一个输出变量
    """
    m = 3
    n = 4
    X = tf.placeholder(tf.float32, name="X", shape=[m, n])
    Y = tf.placeholder(tf.float32, name="Y")

    w0 = tf.Variable(0.0)
    w1 = tf.Variable(tf.random_normal([n, 1]))

    Y_hat = tf.matmul(X, w1) + w0

    # loss function
    loss = tf.reduce_mean(tf.square(Y - Y_hat, name="loss"))


def logistic_regression():
    """
    逻辑回归
    """
    m = 3
    n = 4
    P = 3
    # 逻辑回归的情况下，损失函数定义为交叉熵。输出 Y 的维数等于训练数据集中类别的数量，其中 P 为类别数量
    X = tf.placeholder(tf.float32, name="X", shape=[m, n])
    Y = tf.placeholder(tf.float32, name="Y", shape=[m, P])

    w0 = tf.Variable(tf.zeros([1, P]), name="bias")
    w1 = tf.Variable(tf.random_normal([n, 1]), name="weights")

    Y_hat = tf.matmul(X, w1) + w0

    entropy = tf.nn.softmax_cross_entropy_with_logits(Y_hat, Y)
    loss = tf.reduce_mean(entropy)

    # 如果要加L1正则化
    lambda_ = tf.constant(0.8)
    regularization_param = lambda_ * tf.reduce_mean(tf.abs(w1))
    loss += regularization_param

    # 如果要加L2正则化
    lambda_ = tf.constant(0.8)
    regularization_param = lambda_ * tf.nn.l2_loss(w1)
    loss += regularization_param




def main():
    standard_regression() # 标准线性回归
    duoyuan_regression() # 多元线性回归
    logistic_regression() # 逻辑回归


if __name__ == '__main__':
    main()

