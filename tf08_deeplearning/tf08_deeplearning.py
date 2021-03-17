import tensorflow as tf


"""
tensorflow深度学习基础
http://c.biancheng.net/view/1899.html
"""


def read_data_feed_dict():
    """
    feed_dict读取数据
    """
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    loss = tf.OptimizerOptions
    with tf.Session as sess:
        x_data = 0.3
        y_data = 1.2
        sess.run(loss, feed_dict={x: x_data, y: y_data})


def read_data_from_file(files):
    """
    从文件中读取数据
    """
    file_name_queue = tf.train.string_input_producer(files)
    reader = tf.TextLineReader()
    key, value = reader.read(file_name_queue)
    record_defaults = [[1], [1], [1]]
    col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)


def read_data_preload():
    """
    预加载数据
    """
    # preloaded data as constant
    training_data = ...
    training_labels = ...
    with tf.Session as sess:
        x_data = tf.Constant(training_data)
        y_data = tf.Constant(training_labels)

    # preloaded data as Variables
    training_data = ...
    training_labels = ...
    with tf.Session as sess:
        data_x = tf.placeholder(dtype=training_data.dtype, shape=training_data.shape)
        data_y = tf.placeholder(dtype=training_labels.dtype, shape=training_labels.shape)
        x_data = tf.Variable(data_x, trainable=False, collections=[])
        y_data = tf.Variable(data_y, trainable=False, collections=[])


def main():
    read_data_feed_dict() # feed_dict读取数据
    read_data_from_file() # 从文件中读取数据
    read_data_preload() # 预加载数据


if __name__ == '__main__':
    main()
