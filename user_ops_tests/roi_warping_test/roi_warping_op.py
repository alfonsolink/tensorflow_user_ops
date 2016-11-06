import tensorflow as tf
import os.path as osp

filename = '/home/alfonso/tensorflow/bazel-bin/tensorflow/core/user_ops/roiwarp/roi_warping.so'
_roi_warping_module = tf.load_op_library(filename)
roi_warping = _roi_warping_module.roi_warp
roi_warping_grad = _roi_warping_module.roi_warp_grad
