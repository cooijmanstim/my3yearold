import numpy as np, tensorflow as tf

FF_NORM_AXIS = 0
BOUND_WEIGHTS = False

def maybe_bound_weights(w):
  if BOUND_WEIGHTS:
    w = tf.nn.tanh(w)
  return w

def project_terms(xs, depth=None, normalize=True, bias=True, scope=None):
  xs = list(xs)
  with tf.variable_scope(scope or "project_terms", values=xs):
    if normalize:
      # batch-normalize each projection separately before summing
      projected_xs = [project(x, depth=depth, bias=False, scope=str(i))
                      for i, x in enumerate(xs)]
      normalized_xs = [batch_normalize(x, beta=0, axis=FF_NORM_AXIS, scope=str(i))
                       for i, x in enumerate(projected_xs)]
      y = sum(normalized_xs)
    else:
      # concatenate terms and do one big projection
      y = project(tf.concat(xs, axis=1), depth=depth, bias=False)
    if bias:
      y += tf.get_variable("b", shape=[depth], initializer=tf.constant_initializer(0))
    return y

def project(x, depth, bias=True, scope=None):
  with tf.variable_scope(scope or "project", [x]):
    input_depth = x.get_shape().as_list()[-1]
    w = tf.get_variable("w", shape=[input_depth, depth],
                        initializer=tf.uniform_unit_scaling_initializer())
    w = maybe_bound_weights(w)
    y = tf.matmul(x, w)
    if bias:
      y += tf.get_variable("b", shape=[depth], initializer=tf.constant_initializer(0))
    return y

def batch_normalize(x, beta=None, gamma=None, epsilon=1e-5, scope=None, axis=FF_NORM_AXIS):
  with tf.variable_scope(scope or "norm", [x, beta, gamma]):
    # tf.nn.moments doesn't deal well with dynamic shapes
    mean = tf.reduce_mean(x, axis=axis, keep_dims=True)
    variance = tf.reduce_mean((x - mean)**2, axis=axis, keep_dims=True)

    depth = x.get_shape().as_list()[-1]
    if gamma is None:
      gamma = tf.get_variable("gamma", shape=depth, initializer=tf.constant_initializer(0.1))
      gamma = maybe_bound_weights(gamma)
    if beta is None:
      beta = tf.get_variable("beta", shape=depth, initializer=tf.constant_initializer(0))
    return beta + gamma / tf.sqrt(variance + epsilon) * (x - mean)

def conv_layer(x, radius=None, stride=1, padding="SAME", depth=None, fn=tf.nn.relu, normalize=True, bias=True, scope=None):
  with tf.variable_scope(scope or "conv", []):
    input_depth = x.get_shape().as_list()[-1]
    w = tf.get_variable("w", shape=[radius, radius, input_depth, depth],
                        initializer=tf.uniform_unit_scaling_initializer())
    w = maybe_bound_weights(w)
    y = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
    if normalize:
      y = batch_normalize(y, axis=[FF_NORM_AXIS, 1, 2])
    else:
      b = tf.get_variable("b", shape=[depth],
                          initializer=tf.constant_initializer(0))
      y += b
    return fn(y)

def meanlogabs(x):
  return tf.reduce_mean(tf.log1p(tf.abs(x)))
