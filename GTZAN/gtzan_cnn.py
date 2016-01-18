__author__ = 'oriol'

import os
import preprocess_gtzan
import tensorflow as tf

if not os.path.isfile('features.json'):
    preprocess_gtzan.extract_spectrograms('gtzan_dataset')

train, _y_train, test, _y_test = preprocess_gtzan.pre_process_data()

# launch interactive session
sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder("float", shape=[None, 40])
y_ = tf.placeholder("float", shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([40, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialise
sess.run(tf.initialize_all_variables())

# Implement regression model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# the cost function is the cross_entropy between the target and the model's prediction
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# TRAINING
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

batch_size = len(train)/2018
for i in range(batch_size):
  train_step.run(feed_dict={x: train[i:i+batch_size], y_: _y_train[i:i+batch_size]})

# EVALUATION
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "Simple model accuracy:", accuracy.eval(feed_dict={x: test, y_: _y_test})

# # MULTILAYER CONVOLUTIONAL NETWORK
#
#
# # functions to initialise neurons with a slight positive bias
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# # convolution and pooling functions
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # first convolutional layer
# W_conv1 = weight_variable([5, 5, 1, 32])
# # bias vector with a component for each output channel
# b_conv1 = bias_variable([32])
#
# # reshape x to 4D. Second and third dimension correspond to the images width and height
# x_image = tf.reshape(x, [-1, 1, 1103, 1])
#
# # convolve x_image with the weight tensor, ad the bias, apply RelU function and max_pool
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # In order to build a deep network, we stack several layers of this type.
# #  The second layer will have 64 features for each 5x5 patch.
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # densely connected layer
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# # Dropout
# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# # Readout
# # add softmax layer
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# # EVALUATION
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# sess.run(tf.initialize_all_variables())
# batch_size = len(train)/1009
# for i in range(batch_size):
#     train_accuracy = accuracy.eval(feed_dict={
#         x: train[i:i+batch_size], y_: _y_train[i:i+batch_size], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#     train_step.run(feed_dict={x: train[i:i+batch_size], y_: _y_train[i:i+batch_size], keep_prob: 0.5})