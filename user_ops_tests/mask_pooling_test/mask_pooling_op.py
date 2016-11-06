import tensorflow as tf
import os.path as osp

filename = '/home/alfonso/tensorflow/bazel-bin/tensorflow/core/user_ops/maskpool/mask_pooling.so'
_mask_pooling_module = tf.load_op_library(filename)
mask_pooling = _mask_pooling_module.mask_pool
mask_pooling_grad = _mask_pooling_module.mask_pool_grad
