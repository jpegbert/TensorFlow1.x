import tensorflow as tf


"""
http://c.biancheng.net/view/1885.html
"""


def constant_test():
    """
    常量
    """
    # 标量
    t1 = tf.constant(4)
    # 向量
    t1 = tf.constant([4, 3, 2])

    M = 3
    N = 4
    # 创建全0的张量
    zeros_t = tf.zeros([M, N], tf.int32)

    # 创建全1的张量
    ones_t = tf.ones([2, 3], tf.int32)

    # 在一定范围内生成一个从初值到终值等差排布的序列
    range_t = tf.linspace(2.0, 5.0, 5)
    range_t = tf.range(10)

    # 创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组
    t_random = tf.random_normal([2, 3], mean=2.0, stddev=4, seed=12)
    # 创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的截尾正态分布随机数组
    t_random = tf.truncated_normal([2, 3], mean=2.0, stddev=4, seed=12)
    # 要在种子的[minval（default = 0），maxval] 范围内创建形状为[M，N] 的给定伽马分布随机数组
    t_random = tf.random_uniform([2, 3], maxval=4, seed=12)
    # 将给定的张量随机裁剪为指定的大小
    cropped_data = tf.random_crop(t_random, [2, 5], seed=12)
    # 沿着它的第一维随机排列张量
    tf.random_shuffle(t_random)
    # 随机生成的张量受初始种子值的影响。要在多次运行或会话中获得相同的随机数，应该将种子设置为一个常数值。
    # 当使用大量的随机张量时，可以使用 tf.set_random_seed() 来为所有随机产生的张量设置种子。
    tf.set_random_seed(54)

    with tf.Session() as sess:
        res = sess.run(t_random)
        print(t_random)
        print(res)


def variable_test():
    """
    变量
    变量通常在神经网络中表示权重和偏置
    """
    t_random = tf.random_uniform([50, 50], minval=0, maxval=10, seed=0)
    ta = tf.Variable(t_random)

    # 定义权重和偏置。权重变量使用正态分布随机初始化，均值为 0，标准差为 2，权重大小为 100×100。
    # 偏置由 100 个元素组成，每个元素初始化为 0。在这里也使用了可选参数名以给计算图中定义的变量命名
    weights = tf.Variable(tf.random_normal([100, 100], stddev=2))
    bias = tf.Variable(tf.zeros([100], name="biases"))
    # 也可以指定一个变量来初始化另一个变量。下面的语句将利用前面定义的权重来初始化 weight2
    weight2 = tf.Variable(weights.initialized_value(), name="w2")

    # 变量的定义将指定变量如何被初始化，但是必须显式初始化所有的声明变量。在计算图的定义中通过声明初始化操作对象来实现
    init_op = tf.global_variables_initializer()
    """
    每个变量也可以在运行图中单独使用 tf.Variable.initializer 来初始化
    with tf.Session() as sess:
        sess.run(bias.initializer)
    """
    # 保存变量：使用 Saver 类来保存变量，定义一个 Saver 操作对象
    saver = tf.train.Saver()


def placeholder_test():
    """
    占位符
    用于将数据提供给计算图
    tf.placeholder(dtype,shape=None,name=None)
    """
    x = tf.placeholder("float")
    y = 2 * x
    data = tf.random_uniform([4, 5], 10)
    with tf.Session() as sess:
        x_data = sess.run(data)
        print(sess.run(y, feed_dict={x: x_data}))


def tuozhan():
    """
    拓展
    """
    large_array = tf.random_uniform([2, 3], minval=0, maxval=10, seed=0)
    # 很多时候需要大规模的常量张量对象；在这种情况下，为了优化内存，最好将它们声明为一个可训练标志设置为 False 的变量
    t_large = tf.Varible(large_array, trainable=False)
    # tf.convert_to_tensor()可以将给定的值转换为张量类型，并将其与 TensorFlow1.x 函数和运算符一起使用


def main():
    constant_test()
    variable_test()
    placeholder_test()
    tuozhan()


if __name__ == '__main__':
    main()
