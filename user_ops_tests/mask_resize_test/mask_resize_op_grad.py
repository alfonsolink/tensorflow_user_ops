import tensorflow as tf
from tensorflow.python.framework import ops
import mask_resize_op

@tf.RegisterShape("MaskResize")
def _maskresize_pool_shape(op):
  """Shape function for the RoiPool op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  channels = dims_data[1]
  num_masks = dims_data[0]

  output_width = op.get_attr('output_width')
  output_height = op.get_attr('output_height')


  output_shape = tf.TensorShape([num_masks, channels, output_height, output_width])
  return [output_shape]

@ops.RegisterGradient("MaskResize")
def _maskresize_pool_grad(op, grad):
  """The gradients for `roi_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  dims_data = op.inputs[0].get_shape().as_list()
  channels = dims_data[1]
  output_width = op.get_attr('output_width')
  output_height = op.get_attr('output_height')
 

  data_grad = mask_resize_op.mask_resize_grad(data, grad, output_width, output_height, channels)  

  return [data_grad]  # List of one Tensor, since we have one input
