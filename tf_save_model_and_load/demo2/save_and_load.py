import tensorflow as tf


"""
https://www.jianshu.com/p/52e7ae04cecf
不需重新定义网络结构的保存模型和加载模型的方法
"""


in_dim = 30
out_dim = 10
h1_dim = 20

### 定义模型
input_x = tf.placeholder(tf.float32, shape=(None, in_dim), name='input_x')
input_y = tf.placeholder(tf.float32, shape=(None, out_dim), name='input_y')

w1 = tf.Variable(tf.truncated_normal([in_dim, h1_dim], stddev=0.1), name='w1')
b1 = tf.Variable(tf.zeros([h1_dim]), name='b1')
w2 = tf.Variable(tf.zeros([h1_dim, out_dim]), name='w2')
b2 = tf.Variable(tf.zeros([out_dim]), name='b2')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
hidden1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
### 定义预测目标
y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)
# 创建saver
saver = tf.train.Saver()
# 假如需要保存y，以便在预测时使用
tf.add_to_collection('pred_network', y)
sess = tf.Session()

for step in range(10):
    # sess.run(train_op)
    if step % 1000 == 0:
        # 保存checkpoint, 同时也默认导出一个meta_graph
        # graph名为'my-model-{global_step}.meta'.
        saver.save(sess, 'my-model', global_step=step)


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    y = tf.get_collection('pred_network')[0]

    graph = tf.get_default_graph()

    # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
    input_x = graph.get_operation_by_name('input_x').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    # 使用y进行预测
    # sess.run(y, feed_dict={input_x:...., keep_prob:1.0})

