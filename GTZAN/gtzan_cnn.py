import preprocess_gtzan
import tensorflow as tf

# if not os.path.isfile('features.json'):
preprocess_gtzan.extract_spectrograms('gtzan_dataset')

train, _y_train, test, _y_test = preprocess_gtzan.pre_process_data()

# launch interactive session
sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder("float", shape=[None, 800])
y_ = tf.placeholder("float", shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([800, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialise
sess.run(tf.initialize_all_variables())

# Implement regression model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# the cost function is the cross_entropy between the target and the model's prediction
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# TRAINING
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(155):
    if i == 0:
        a = 50
    else:
        a = 0
    train_step.run(feed_dict={x: train[i*50:2*i*50+a], y_: _y_train[i*50:2*i*50+a]})

# EVALUATION
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "Simple model accuracy:", accuracy.eval(feed_dict={x: test, y_: _y_test})


