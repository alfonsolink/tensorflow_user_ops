import tensorflow as tf
import os.path as osp

filename = '/home/alfonso/tensorflow/bazel-bin/tensorflow/core/user_ops/girshick_roipool/roi_pooling_girshick.so'
_roi_pooling_module = tf.load_op_library(filename)
roi_pool = _roi_pooling_module.roi_pool
roi_pool_grad = _roi_pooling_module.roi_pool_grad
