import tensorflow as tf


"""
优化器
"""


loss = ...

# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(train_step, feed_dict={})


# momentum动量
optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.5)


# 自适应单调递减学习率
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.95)


#
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.8, momentum=0.5)


optimizer = tf.train.AdamOptimizer()


# 指数衰减学习率
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.2
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=1000, decay_rate=0.95, staircase=True)


