import tensorflow as tf
import numpy as np
import psroi_pooling_op
import psroi_pooling_op_grad

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(1, 100, 100, 21*7*7)
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois = tf.convert_to_tensor([[0, 10, 10, 20, 20], [0, 30, 30, 40, 40]], dtype=tf.float32)

W = weight_variable([3, 3, 21*7*7, 21*7*7])
h = conv2d(data, W)

#W = weight_variable([3, 3, 1024, 21])
#h = conv2d(h2, W)

h = tf.transpose(h, [0,3,1,2])
[y2, channels] = psroi_pooling_op.psroi_pool(h, rois, output_dim=21, group_size=3, spatial_scale=1.0/16)
y = tf.transpose(y2,[0,2,3,1])
y = tf.reduce_mean(y, [1,2])
#y = tf.nn.avg_pool(y, (1,7,7,1),(1,7,7,1),'VALID')
y = tf.reshape(y,(2,21))
#y_data = tf.convert_to_tensor(np.ones((2, 1, 1, 21)), dtype=tf.float32)
y_data = tf.convert_to_tensor((4,2),dtype=tf.int64)
#print y_data, y, argmax

# Minimize the mean squared errors.
#loss = tf.reduce_mean(tf.abs(y - y_data))
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_data)
optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
gvs = optimizer.compute_gradients(loss)
#capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
train = optimizer.apply_gradients(gvs)
#train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

for step in xrange(100000):
    sess.run(train)
    #print(step, sess.run(W))
    #print(sess.run(y))
    l = sess.run([loss])
    print l
    #print sess.run(y2)
#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
