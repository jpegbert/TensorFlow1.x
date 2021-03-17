import tensorflow as tf


"""
TensorFlow指定CPU和GPU设备操作详解
http://c.biancheng.net/view/1893.html
"""

# TensorFlow 将支持的 CPU 设备命名为“/device：CPU：0”（或“/cpu：0”），第 i 个 GPU 设备命名为“/device：GPU：I”（或“/gpu：I”）。


# 设置GPU的方法如下
# 要验证 TensorFlow 是否确实在使用指定的设备（CPU 或 GPU），可以创建会话，并将 log_device_placement 标志设置为 True，即：
config = tf.ConfigProto(log_device_placement=True)
# 如果你不确定设备，并希望 TensorFlow 选择现有和受支持的设备，则可以将 allow_soft_placement 标志设置为 True
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# 手动选择CPU进行操作
with tf.device("/cpu:0"):
    rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
    a = tf.Variable(rand_t)
    b = tf.Variable(rand_t)
    c = tf.matmul(a, b)
    init = tf.global_variables_initializer()
sess = tf.Session(config)
sess.run(init)
print(sess.run(c))


# 手动选择一个GPU
with tf.device("/gpu:0"):
    rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
    a = tf.Variable(rand_t)
    b = tf.Variable(rand_t)
    c = tf.matmul(a, b)
    init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
print(sess.run(c))


# 手动选择多个GPU
c = []
for d in ["/gpu:1", "/gpu:2"]:
    with tf.device("/gpu:0"):
        rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
        a = tf.Variable(rand_t)
        b = tf.Variable(rand_t)
        c.append(tf.matmul(a, b))
        init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(init)
print(sess.run(c))
sess.close()
# 在这种情况下，如果系统有 3 个 GPU 设备，那么第一组乘法将由'/：gpu：1'执行，第二组乘以'/gpu：2'执行。
