import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns


"""
TensorFlow多层感知机函数逼近过程详解
http://c.biancheng.net/view/1924.html
"""


boston = datasets.load_boston
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["target"] = boston.target
print(df.head())

print(df.describe())

color_map_, ax = plt.subplots(figsize=(12, 10))
corr = df.corr(method="pearson")
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={"shrink": .9}, ax=ax, annot=True, annot_kws={"font_size": 12})

X_train, X_test, y_train, y_test = train_test_split(df[["RM", "LSTAT", "PTRATIO"]], df[["target"]], test_size=0.3, random_state=0)
X_train = MinMaxScaler().fit_transform(X_train)
y_train = MinMaxScaler().fit_transform(y_train)
X_test = MinMaxScaler().fit_transform(X_test)
y_test = MinMaxScaler().fit_transform(y_test)

m = len(X_train)
n = 3
n_hidden = 20
batch_size = 200
eta = 0.01
max_epoch = 1000


def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, n_hidden, activation_fn=tf.nn.relu, scope="fc1")
    out = layers.fully_connected(fc1, 1, activation_fn=tf.sigmoid, scope="out")
    return out


x = tf.placeholder(tf.float32, name="X", shape=[m, n])
y = tf.placeholder(tf.float32, name="Y")
y_hat = multilayer_perceptron(x)
correct_prediction = tf.square(y - y_hat)
mse = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train = tf.train.AdamOptimizer(learning_rate=eta).minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("graph", sess.graph)
    for i in range(max_epoch):
        _, l, p = sess.run([train, mse, y_hat], feed_dict={x: X_train, y: y_train})
        if i % 100 == 0:
            print("Epoch {0}, loss {1}".format(i, l))
    print("Train Done")
print("optimization done")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Mean Error: ", accuracy.eval({x: X_train, y: y_train}))
plt.scatter(y_train, p)
writer.close()





