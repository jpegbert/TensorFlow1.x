import tensorflow as tf
import numpy as np


"""
https://www.jianshu.com/p/52e7ae04cecf
保存模型和加载模型
这种方法不方便的在于，在使用模型的时候，必须把模型的结构重新定义一遍，然后载入对应名字的变量的值。
"""


def save():
    W = tf.Variable([[1, 1, 1], [2, 2, 2]], dtype=tf.float32, name="w")
    b = tf.Variable([0, 1, 2], dtype=tf.float32, name="b")
    print(W.shape)
    print(b.shape)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # 保存模型
        save_path = saver.save(sess, "save/model.ckpt")
        # 加载模型
        # saver.restore(sess, "save/model.ckpt")


def load():
    W = tf.Variable(tf.truncated_normal(shape=(2, 3)), dtype=tf.float32, name='w')
    b = tf.Variable(tf.truncated_normal(shape=(3,)), dtype=tf.float32, name='b')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "save/model.ckpt")


def main():
    save()
    # load()


if __name__ == '__main__':
    main()

