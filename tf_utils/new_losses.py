from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

FLAGS = None
def convert_PN_enc(target):
  """ Convert label encoding from one-hot 0-1 encoding to +1, -1 encoding
  """
  all_ones = array_ops.ones_like(target)
  # convert labels into {1, -1} matrix
  labels = math_ops.sub(2 * target, all_ones)
  return labels

def hinge_loss_cap(logits, target, cap=10.0, scope=None):
  """Method that returns the loss tensor for capped hinge loss.

  Args:
    logits: The logits, a float tensor.
    target: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    cap: parameter for the hinge loss upper bound
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A `Tensor` of same shape as logits and target representing the loss values
      across the batch.

  Raises:
    ValueError: If the shapes of `logits` and `target` don't match.
  """
  with ops.name_scope(scope) as scope:
    logits.get_shape().assert_is_compatible_with(target.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    target = math_ops.to_float(target)
    all_ones = array_ops.ones_like(target)
    # convert labels into {1, -1} matrix
    labels = math_ops.sub(2 * target, all_ones)
    cross_prod = math_ops.mul(labels, logits)
    losses = nn_ops.relu(math_ops.sub(all_ones, cross_prod))
    #losses_cap = -nn_ops.relu(math_ops.sub(cap, losses))
    losses_cap = math_ops.minimum(cap, losses) 
    return losses_cap

    # sigmoid loss
    #losses_left = math_ops.sigmoid(-10*cross_prod)
    #losses_right = math_ops.sigmoid(-0.01*cross_prod)
    #losses_soft = math_ops.minimum(losses_left, losses_right)
