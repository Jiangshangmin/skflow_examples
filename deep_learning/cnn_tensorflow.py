# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# examples:
# array([[ 0.,  0.,  0., ...,  1.,  0.,  0.],
#        [ 0.,  0.,  1., ...,  0.,  0.,  0.],
#        [ 0.,  1.,  0., ...,  0.,  0.,  0.],
#        ...

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = 25000
batch_size = 50
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# variables initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# create model
def conv2d(x, W):
    # when padding=Same, which applying zero padding so that the output is the same size as input
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X):
    _X = tf.reshape(_X, [-1, 28, 28, 1])

    # First Convolution Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    conv1 = tf.nn.relu(conv2d(_X, W_conv1) + b_conv1)
    pool1 = max_pool(conv1, 2)

    # Second Convolution Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
    pool2 = max_pool(conv2, 2)

    # Fully Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)

    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # softmax layer/output layer
    W_fc2 = weight_variable([1024, n_classes])
    b_fc2 = bias_variable([n_classes])

    y_conv = tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) + b_fc2)
    return y_conv

sess = tf.InteractiveSession()
y_conv = conv_net(x)
# train and evaluate
cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(training_iters):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        print "Step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

print "Test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
