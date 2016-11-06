import tensorflow as tf
import os.path as osp

filename = '/home/alfonso/tensorflow/bazel-bin/tensorflow/core/user_ops/mask_resize/mask_resize.so'
_mask_resize_module = tf.load_op_library(filename)
mask_resize = _mask_resize_module.mask_resize
mask_resize_grad = _mask_resize_module.mask_resize_grad
