import tensorflow as tf
import numpy as np
import mask_pooling_op
import mask_pooling_op_grad

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(4, 100, 100, 3)
data = tf.convert_to_tensor(array, dtype=tf.float32)

W = weight_variable([3, 3, 3, 1000])
h = conv2d(data, W)
W2 = weight_variable([1, 1, 1000, 1])
h2 = conv2d(h, W2)
h2 = tf.transpose(h2, [0,3,1,2])

array2 = np.random.rand(4, 100, 100, 3)
data_m = tf.convert_to_tensor(array2, dtype=tf.float32)
W_m = weight_variable([3, 3, 3, 1000])
h_m = conv2d(data_m, W_m)
W2_m = weight_variable([1, 1, 1000, 1])
h2_m = conv2d(h_m, W2_m)
h2_m = tf.transpose(h2_m, [0,3,1,2])
masks = h2_m

#masks = tf.convert_to_tensor(np.round(np.ones((4, 1, 100, 100))), dtype=tf.float32)

y2 = mask_pooling_op.mask_pooling(h2, masks, 100, 100)
y = tf.transpose(y2, [0,2,3,1])

#y_data = np.ones((2, 100, 100, 1))
y_data = tf.convert_to_tensor((np.round(np.random.rand(4, 100, 100, 1))), dtype=tf.float32)
#print y_data, y, argmax

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.MomentumOptimizer(0.001, 0.9, use_nesterov = True)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

for step in xrange(100000):
    sess.run(train)
    #print y2
    #print(step, sess.run(W))
    #print(sess.run(y))
    #print sess.run(loss)
    #r = np.sum(sess.run([y2]))
    l = sess.run([loss])
    print l
#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
