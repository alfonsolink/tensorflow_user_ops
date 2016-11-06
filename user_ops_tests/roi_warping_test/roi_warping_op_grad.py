import tensorflow as tf
from tensorflow.python.framework import ops
import roi_warping_op

@tf.RegisterShape("RoiWarp")
def _roi_warping_shape(op):
  """Shape function for the RoiPool op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  channels = dims_data[1]

  dims_rois = op.inputs[1].get_shape().as_list()
  num_rois = dims_rois[0]

  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')

  output_shape = tf.TensorShape([num_rois, channels, pooled_height, pooled_width])
  return [output_shape, output_shape, output_shape]

@ops.RegisterGradient("RoiWarp")
def _roi_warping_grad(op, grad, h, w):
  """The gradients for `roi_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  rois = op.inputs[1]
  argmax_h = op.outputs[1]
  argmax_w = op.outputs[2]
  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')
  spatial_scale = op.get_attr('spatial_scale')

  # compute gradient
  data_grad, buffer_, data_grad_feature = roi_warping_op.roi_warping_grad(data, rois, argmax_h, argmax_w, grad, pooled_height, pooled_width, spatial_scale)

  return [data_grad_feature, data_grad]  
