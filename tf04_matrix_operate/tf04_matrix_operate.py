import tensorflow as tf


"""
http://c.biancheng.net/view/1886.html
TensorFlow矩阵基本操作及其实现
"""


# 开始一个交互式会话，以便得到计算结果
sess = tf.InteractiveSession()

# 定义一个5 * 5的单位矩阵
I_matrix = tf.eye(5)
print(I_matrix.eval())

# 定义一个长度为10的单位矩阵的变量
X = tf.Variable(tf.eye(10))
# 初始化
X.initializer.run()
print(X.eval())

# 定义一个 5 * 10 的矩阵
A = tf.Variable(tf.random_normal([5, 10]))
A.initializer.run()
print(A.eval())

# 两个矩阵相乘
product = tf.matmul(A, X)
print(product.eval())

b = tf.Variable(tf.random_uniform([5, 10], 0, 2, dtype=tf.int32))
b.initializer.run()
print(b.eval())
# 把b转换为float类型
b_new = tf.cast(b, dtype=tf.float32)

# 两个矩阵相加
t_sum = tf.add(product, b_new)
# 两个矩阵相减
t_sub = product - b_new
print(t_sum.eval())
print(t_sub.eval())

# 按元素相乘，element wise multiplication
a = tf.Variable(tf.random_normal([4, 5], stddev=2))
b = tf.Variable(tf.random_normal([4, 5], stddev=2))
A = a * b
a.initializer.run()
b.initializer.run()
print("---")
print(A.eval())

# 矩阵与标量相乘
B = tf.scalar_mul(2, A)
print("===")
print(B.eval())

# 按元素相除 element wise division
C = tf.div(a, b)
print("***")
print(C.eval())

# 按元素相除取余 element wise division
D = tf.mod(a, b)
print("---#")
print(D.eval())

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter("graphs", sess.graph)
    a, b, A_R, B_R, C_R, D_R = sess.run([a, b, A, B, C, D])
    print(a, b, A_R, B_R, C_R, D_R)
writer.close()



