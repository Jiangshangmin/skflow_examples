# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
# example: array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)

import tensorflow as tf
import skflow
from sklearn import metrics

# Parameters
learning_rate = 0.001
training_iters = 30000
batch_size = 50

def max_pool(tensor_in, k):
    return tf.nn.max_pool(tensor_in, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_model(X, y):
    X = tf.reshape(X, [-1, 28, 28, 1])

    with tf.variable_scope('conv_layer1'):
        conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        pool1 = max_pool(conv1, 2)

    with tf.variable_scope('conv_layer2'):
        conv2 = skflow.ops.conv2d(pool1, n_filters=64, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        pool2 = max_pool(conv2, 2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    fc1 = skflow.ops.dnn(pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.logistic_regression(fc1, y)

classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10, batch_size=batch_size,
                                        steps=training_iters, learning_rate=learning_rate)
classifier.fit(mnist.train.images, mnist.train.labels)

score = metrics.accuracy_score(classifier.predict(mnist.test.images), mnist.test.labels)
print 'Test Accuracy: %g' % score
