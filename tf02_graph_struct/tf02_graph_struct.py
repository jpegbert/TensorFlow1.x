import tensorflow as tf


"""
http://c.biancheng.net/view/1883.html
"""


v1 = tf.constant([1, 2, 3])
v2 = tf.constant([4, 5, 6])
v_add = tf.add(v1, v2)


def method1():
    # InteractiveSession比tf.Session。InteractiveSession使自己成为默认会话，可以使用eval()直接调用运行张量对象而不用显式调用会话
    sess = tf.InteractiveSession()
    print(v_add.eval())
    sess.close()


def method2():
    with tf.Session() as sess:
        print(sess.run(v_add))


def method3():
    sess = tf.Session()
    print(sess.run(v_add))
    sess.close()


def main():
    method1()
    method2()
    method3()


if __name__ == '__main__':
    main()
