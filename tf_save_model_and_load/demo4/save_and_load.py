import tensorflow as tf


def save():
    # prepare to feed input, i.e. feed_dict and placeholders
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')  # name is very important in restoration
    w2 = tf.Variable(tf.random_normal(shape=[2]), name='w2')
    b1 = tf.Variable(2.0, name='bias1')
    feed_dict = {w1: [10, 3], w2: [5, 5]}

    # define a test operation that will be restored
    w3 = tf.add(w1, w2)  # without name, w3 will not be stored
    w4 = tf.multiply(w3, b1, name="op_to_restore")

    # saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(w4, feed_dict)) # [30. 16.]
    # saver.save(sess, 'my_test_model', global_step = 100)
    saver.save(sess, 'model/my_test_model')
    # saver.save(sess, 'my_test_model', global_step = 100, write_meta_graph = False)


def load():
    sess = tf.Session()

    # First, load meta graph and restore weights
    saver = tf.train.import_meta_graph('model/my_test_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))

    # Second, access and create placeholders variables and create feed_dict to feed new data
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name('w1:0')
    w2 = graph.get_tensor_by_name('w2:0')
    feed_dict = {w1: [-1, 1], w2: [4, 6]}
    # feed_dict = {w1: [10, 3], w2: [5, 5]}

    # Access the op that want to run
    op_to_restore = graph.get_tensor_by_name('op_to_restore:0')

    print(sess.run(op_to_restore, feed_dict))  # ouotput: [6. 14.]


def main():
    # save()
    load()


if __name__ == '__main__':
    main()

