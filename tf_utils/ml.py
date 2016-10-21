from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

def least_square(x, y, reg=0.01, x_test=None, y_test=None):
  """ Least squre solver, with x as a n by d matrix, y as a n by c matrix
  """
  d = x.shape[1]
  xx = math_ops.matmul(tf.transpose(x), x) + reg * np.eye(d)
  xx_inv = tf.matrix_inverse(xx)
  import ipdb
  ipdb.set_trace()
  w = np.dot(xx_inv, np.dot(x.T, y))

  if x_test is not None and y_test is not None:
    y_c = np.dot(x_test, w)
    accu = tf.equal(tf.argmax(y_c, 1), tf.argmax(y_test, 1))
    accu_val = tf.reduce_mean(tf.cast(accu, tf.float32))
  return w, accu_val 

if __name__ == "__main__":
  from tensorflow.examples.tutorials.mnist import input_data
  mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
   
  batch = mnist.train.next_batch(50)
  least_square(batch[0], batch[1], 0.01, batch[0], batch[1])
