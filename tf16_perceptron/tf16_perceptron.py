import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""
常用激活函数及其特点和用法（6种）详解
http://c.biancheng.net/view/1911.html
"""


def threshold(x):
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype=x.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


def threshold_activation():
    """
    阈值激活函数：这是最简单的激活函数。在这里，如果神经元的激活值大于零，那么神经元就会被激活；否则，它还是处于抑制状态。
    """
    h = np.linspace(-1, 1, 50)
    out = threshold(h)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)
        plt.xlabel("Activity of Neuron")
        plt.ylabel("out of Neuron")
        plt.title("Threshold Activation Function")
        plt.plot(h, y)
        plt.show()


def sigmoid_activation():
    """
    Sigmoid 激活函数：在这种情况下，神经元的输出由函数 g(x)=1/(1+exp(-x)) 确定
    """
    h = np.linspace(-10, 10, 50)
    out = tf.sigmoid(h)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)
        plt.xlabel("Activity of Neuron")
        plt.ylabel("out of Neuron")
        plt.title("Sigmoid Activation Function")
        plt.plot(h, y)
        plt.show()


def tanh_activation():
    """
    双曲正切激活函数：在数学上，它表示为 (1-exp(-2x)/(1+exp(-2x)))。在形状上，
    它类似于 Sigmoid 函数，但是它的中心位置是 0，其范围是从 -1 到 1。
    """
    h = np.linspace(-10, 10, 50)
    out = tf.tanh(h)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)
        plt.xlabel("Activity of Neuron")
        plt.ylabel("out of Neuron")
        plt.title("Hyperblic Tangent Activation Function")
        plt.plot(h, y)
        plt.show()


def linear_activation():
    """
    线性激活函数：在这种情况下，神经元的输出与神经元的输入值相同。这个函数的任何一边都不受限制
    """
    X_in = tf.placeholder(tf.float32, shape=[None, 3], name="X_in")
    b = tf.Variable(tf.random_normal([1, 1], stddev=2))
    w = tf.Variable(tf.random_normal([3, 1], stddev=2))
    linear_out = tf.matmul(X_in, w) + b
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = sess.run(linear_out)
    print(out)


def relu_activation():
    """
    整流线性单元（ReLU）激活函数也被内置在 TensorFlow 库中。这个激活函数类似于线性激活函数，但有一个大的改变：
    对于负的输入值，神经元不会激活（输出为零），对于正的输入值，神经元的输出与输入值相同
    """
    h = np.linspace(-10, 10, 50)
    out = tf.nn.relu(h)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)
    plt.xlabel("Activity of Neuron")
    plt.ylabel("out of Neuron")
    plt.title("ReLU Activation Function")
    plt.plot(h, y)
    plt.show()


def softmax_activation():
    """
    Softmax 激活函数是一个归一化的指数函数。一个神经元的输出不仅取决于其自身的输入值，
    还取决于该层中存在的所有其他神经元的输入的总和。这样做的一个优点是使得神经元的输出小，因此梯度不会过大。
    """
    h = np.linspace(-5, 5, 50)
    out = tf.nn.softmax(h)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)
        plt.xlabel("Activity of Neuron")
        plt.ylabel("out of Neuron")
        plt.title("Softmax Activation Function")
        plt.plot(h, y)
        plt.show()


def main():
    threshold_activation() # 阈值激活函数
    sigmoid_activation() # Sigmoid激活函数
    tanh_activation() # 双曲正切激活函数
    linear_activation() # 线性激活函数
    relu_activation() # 整流线性单元（ReLU）激活函数
    softmax_activation() # Softmax 激活函数是一个归一化的指数函数



if __name__ == '__main__':
    main()
