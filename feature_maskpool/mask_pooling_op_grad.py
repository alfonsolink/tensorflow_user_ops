import tensorflow as tf
from tensorflow.python.framework import ops
import mask_pooling_op

@tf.RegisterShape("MaskPool")
def _mask_pool_shape(op):
  """Shape function for the MaskPool op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  channels = dims_data[1]

  dims_masks = op.inputs[1].get_shape().as_list()
  num_masks = dims_masks[0]

  height = op.get_attr('height')
  width = op.get_attr('width')

  output_shape = tf.TensorShape([num_masks, channels, height, width])
  return [output_shape]

@ops.RegisterGradient("MaskPool")
def _mask_pool_grad(op, grad, _):
  """The gradients for `mask_pool`.
  Args:
    op: The `mask_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `mask_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  masks = op.inputs[1]
  height = op.get_attr('height')
  width = op.get_attr('width')
  
  # compute gradient
  data_grad = mask_pooling_op.mask_pool_grad(data, masks, grad, height, width)

  return [data_grad, None]  # List of one Tensor, since we have one input
