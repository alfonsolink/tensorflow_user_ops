import tensorflow as tf
import numpy as np
import roi_warping_op
import roi_warping_op_grad

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(32, 100, 100, 3)
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois = tf.convert_to_tensor([[0, 0, 0, 10, 10], [0, 0, 0, 10, 10]], dtype=tf.float32)

W = weight_variable([3, 3, 3, 1])
h = conv2d(data, W)
h = tf.transpose(h, [0,3,1,2])

W1 = weight_variable([5,5])
rois2 = tf.matmul(rois,W1)

[y2, argmax_h, argmax_w] = roi_warping_op.roi_warping(h, rois2, 100, 100, 1)
y = tf.transpose(y2, [0,2,3,1])
y_data = np.random.rand(2, 100, 100, 1)
#y_data = tf.convert_to_tensor(np.ones((2, 7, 7, 1)), dtype=tf.float32)
#print y_data, y, argmax

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

for step in xrange(10000):
    sess.run(train)
    #print y2
    #print(step, sess.run(W))
    #print(sess.run(y))
    l = sess.run([loss])
    print l
#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
