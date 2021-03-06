import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
# 本文件程序为配合教材及学习进度渐进进行，请按照注释分段执行
# 执行时要注意IDE的当前工作过路径，最好每段重启控制器一次，输出结果更准确

"""
https://blog.csdn.net/qq_40614981/article/details/81604020
"""

# 执行本段程序时注意当前的工作路径


def basic_save():
    """
    Part1: 通过tf.train.Saver类实现保存和载入神经网络模型
    """
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    add_ = tf.add(v1, v2, name="add_")
    # result = v1 + v2

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "Model/model.ckpt")


def basic_load():
    """
    # Part2: 加载TensorFlow模型的方法
    """
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./Model/model.ckpt")  # 注意此处路径前添加"./"
        print(sess.run(result))  # [ 3.]


def load_without_redefine_graph():
    """
    Part3: 若不希望重复定义计算图上的运算，可直接加载已经持久化的图
    """
    saver = tf.train.import_meta_graph("Model/model.ckpt.meta")

    with tf.Session() as sess:
        saver.restore(sess, "./Model/model.ckpt")  # 注意路径写法
        print(sess.run(tf.get_default_graph().get_tensor_by_name("add_:0")))  # [ 3.]

        # graph = tf.get_default_graph()
        # v1 = graph.get_operation_by_name('v1') # .outputs[0]
        # v2 = graph.get_operation_by_name('v2') # .outputs[0]
        # add_ = graph.get_operation_by_name('add_').outputs[0]
        # print(sess.run([add_], feed_dict={v1: 2.0, v2: 3.0}))


def load_without_redefine_graph_and_rename_variable():
    """
    Part4： tf.train.Saver类也支持在保存和加载时给变量重命名
    """
    # 声明的变量名称name与已保存的模型中的变量名称name不一致
    u1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
    u2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
    result = u1 + u2

    # 若直接生命Saver类对象，会报错变量找不到
    # 使用一个字典dict重命名变量即可，{"已保存的变量的名称name": 重命名变量名}
    # 原来名称name为v1的变量现在加载到变量u1（名称name为other-v1）中
    saver = tf.train.Saver({"v1": u1, "v2": u2})

    with tf.Session() as sess:
        saver.restore(sess, "./Model/model.ckpt")
        print(sess.run(result))  # [ 3.]


def save_huadongpingjun_model():
    """
    Part5: 保存滑动平均模型
    """
    v = tf.Variable(0, dtype=tf.float32, name="v")
    for variables in tf.global_variables():
        print(variables.name)  # v:0

    ema = tf.train.ExponentialMovingAverage(0.99)
    maintain_averages_op = ema.apply(tf.global_variables())
    for variables in tf.global_variables():
        print(variables.name)  # v:0
        # v/ExponentialMovingAverage:0

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(v, 10))
        sess.run(maintain_averages_op)
        saver.save(sess, "Model/model_ema.ckpt")
        print(sess.run([v, ema.average(v)]))  # [10.0, 0.099999905]


def load_huadongpingjun():
    """
    Part6: 通过变量重命名直接读取变量的滑动平均值
    """
    v = tf.Variable(0, dtype=tf.float32, name="v")
    saver = tf.train.Saver({"v/ExponentialMovingAverage": v})

    with tf.Session() as sess:
        saver.restore(sess, "./Model/model_ema.ckpt")
        print(sess.run(v))  # 0.0999999


def get_redefined_dict():
    """
    Part7: 通过tf.train.ExponentialMovingAverage的variables_to_restore()函数获取变量重命名字典
    """
    v = tf.Variable(0, dtype=tf.float32, name="v")
    # 注意此处的变量名称name一定要与已保存的变量名称一致
    ema = tf.train.ExponentialMovingAverage(0.99)
    print(ema.variables_to_restore())
    # {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
    # 此处的v取自上面变量v的名称name="v"

    saver = tf.train.Saver(ema.variables_to_restore())

    with tf.Session() as sess:
        saver.restore(sess, "./Model/model_ema.ckpt")
        print(sess.run(v))  # 0.0999999


def save_variable_and_value_to_file():
    """
    Part8: 通过convert_variables_to_constants函数将计算图中的变量及其取值通过常量的方式保存于一个文件中
    """
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 导出当前计算图的GraphDef部分，即从输入层到输出层的计算过程部分
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

        with tf.gfile.GFile("Model/combined_model.pb", 'wb') as f:
            f.write(output_graph_def.SerializeToString())


def load_variable_and_value():
    """
    Part9: 载入包含变量及其取值的模型
    """
    with tf.Session() as sess:
        model_filename = "Model/combined_model.pb"
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        result = tf.import_graph_def(graph_def, return_elements=["add:0"])


def main():
    # Part1: 通过tf.train.Saver类实现保存和载入神经网络模型
    # basic_save()
    # # Part2: 加载TensorFlow模型的方法
    # basic_load()
    # Part3: 若不希望重复定义计算图上的运算，可直接加载已经持久化的图
    load_without_redefine_graph()
    # Part4： tf.train.Saver类也支持在保存和加载时给变量重命名
    # load_without_redefine_graph_and_rename_variable()
    # Part5: 保存滑动平均模型
    # save_huadongpingjun_model()
    # Part6: 通过变量重命名直接读取变量的滑动平均值
    # load_huadongpingjun()
    # Part7: 通过tf.train.ExponentialMovingAverage的variables_to_restore()函数获取变量重命名字典
    # get_redefined_dict()
    # Part8: 通过convert_variables_to_constants函数将计算图中的变量及其取值通过常量的方式保存于一个文件中
    # save_variable_and_value_to_file()
    # Part9: 载入包含变量及其取值的模型
    # load_variable_and_value()


if __name__ == '__main__':
    main()
