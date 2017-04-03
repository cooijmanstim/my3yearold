import numbers
import functools as ft, operator as op
import numpy as np, tensorflow as tf
import util, holster
from dynamite import D

def get_collection(key, scope=None):
  # https://github.com/tensorflow/tensorflow/issues/7719
  if isinstance(scope, tf.VariableScope):
    scope = scope.name + "/"
  return tf.get_collection(key, scope=scope)

def get_depth(x):
  return x.get_shape().as_list()[-1]

def collapse(x, axisgroups):
  # use static shape info where available
  old_shape = [x.shape.as_list()[i] if x.shape.as_list()[i] is not None else tf.shape(x)[i]
               for i in range(x.shape.ndims)]
  axisgroups = [[axes] if isinstance(axes, numbers.Integral) else axes for axes in axisgroups]
  order = [axis for axes in axisgroups for axis in axes]
  if order != list(range(len(order))):
    x = tf.transpose(x, order)
  shape = [ft.reduce(op.mul, [old_shape[axis] for axis in axes], 1) for axes in axisgroups]
  return tf.reshape(x, shape)

def layer(xs, fn=tf.nn.relu, **project_terms_kwargs):
  return fn(project_terms(xs, **project_terms_kwargs))

def project_terms(xs, depth=None, normalizer=util.DEFAULT, bias=True, scope=None):
  if normalizer is util.DEFAULT:
    normalizer = normalize

  xs = list(xs)
  with tf.variable_scope(scope or "project_terms", values=xs):
    if normalizer:
      # batch-normalize each projection separately before summing
      projected_xs = [project(x, depth=depth, bias=False, scope=str(i))
                      for i, x in enumerate(xs)]
      normalized_xs = [normalizer(x, beta=0., scope=str(i))
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
    y = tf.matmul(x, w)
    if bias:
      y += tf.get_variable("b", shape=[depth], initializer=tf.constant_initializer(0))
    return y

def conv_layer(x, radius=1, stride=1, padding="SAME", depth=None, fn=tf.nn.relu,
               normalizer=util.DEFAULT, bias=True, scope=None):
  if normalizer is util.DEFAULT:
    normalizer = normalize

  with tf.variable_scope(scope or "conv", []):
    input_depth = get_depth(x)
    w = tf.get_variable("w", shape=[radius, radius, input_depth, depth],
                        initializer=tf.uniform_unit_scaling_initializer())
    y = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
    if normalizer:
      y = normalizer(y)
    else:
      b = tf.get_variable("b", shape=[depth], initializer=tf.constant_initializer(0))
      y += b
    return fn(y)

def residual_block(h, scope=None, depth=None, fn=tf.nn.relu, **conv_layer_kwargs):
  input_depth = get_depth(h)
  if not depth:
    depth = input_depth
  conv_layer_kwargs["depth"] = depth
  conv_layer_kwargs["fn"] = lambda x: x
  with tf.variable_scope(scope or "res"):
    if depth == input_depth:
      h_thru = h
    else:
      thru_layer_kwargs = dict(conv_layer_kwargs)
      thru_layer_kwargs["radius"] = 1
      h_thru = conv_layer(h, scope="thru", bias=False, **thru_layer_kwargs)
    h = conv_layer(h, scope="pre",  **conv_layer_kwargs)
    h = fn(h)
    h = conv_layer(h, scope="post", **conv_layer_kwargs)
    h = fn(h + h_thru)
    return h

def toconv(x, depth, height, width, **project_terms_kwargs):
  return tf.reshape(project_terms([x], depth=depth * height * width,
                                  **project_terms_kwargs),
                    [tf.shape(x)[0], height, width, depth])

def fromconv(x, depth, **project_terms_kwargs):
  return project_terms([tf.reshape(x, [tf.shape(x)[0], int(np.prod(x.shape[1:]))])],
                       depth=depth, **project_terms_kwargs)

def meanlogabs(x):
  return tf.reduce_mean(tf.log1p(tf.abs(x)))

softmax_xent = tf.nn.softmax_cross_entropy_with_logits # geez

def normalize(x, *args, **kwargs):
  if x.shape.ndims == 2:
    return layer_normalize(x, *args, **kwargs)
  elif x.shape.ndims == 4:
    kwargs.setdefault("axes", [0, 1, 2])
    return batch_normalize(x, *args, **kwargs)
  else:
    raise ValueError()

def layer_normalize(x, beta=None, gamma=None, epsilon=1e-5, scope=None, axes=-1):
  with tf.variable_scope(scope or "ln"):
    axes = [axes] if isinstance(axes, numbers.Integral) else axes
    mean, variance = tf.nn.moments(x, axes=axes, keep_dims=True)
    if gamma is None:
      gamma = tf.get_variable("gamma", shape=x.get_shape()[-1:], initializer=tf.constant_initializer(0.1))
    if beta is None:
      beta = tf.get_variable("beta", shape=x.get_shape()[-1:], initializer=tf.constant_initializer(0))
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=epsilon)

def batch_normalize(x, beta=None, gamma=None, epsilon=1e-5, scope=None, axes=0):
  with tf.variable_scope(scope or "bn"):
    axes = [axes] if isinstance(axes, numbers.Integral) else axes
    batchmean, batchvariance = tf.nn.moments(x, axes=axes, keep_dims=True)

    popmean = tf.get_variable("popmean", shape=batchmean.shape, trainable=False,
                              collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
                              initializer=tf.constant_initializer(0.0))
    popvariance = tf.get_variable("popvariance", shape=batchvariance.shape, trainable=False,
                                  collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
                                  initializer=tf.constant_initializer(1.0))

    if D.train:
      decay = 0.05
      mean, variance = batchmean, batchvariance
      updates = [popmean.assign_sub(decay * (popmean - mean)),
                 popvariance.assign_sub(decay * (popvariance - variance))]
      # make update happen when mean/variance are used
      with tf.control_dependencies(updates):
        mean, variance = tf.identity(mean), tf.identity(variance)
    else:
      mean, variance = tf.convert_to_tensor(popmean), tf.convert_to_tensor(popvariance)

    if gamma is None:
      gamma = tf.get_variable("gamma", shape=variance.shape, initializer=tf.constant_initializer(0.1))
    if beta is None:
      beta = tf.get_variable("beta", shape=mean.shape, initializer=tf.constant_initializer(0))
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=epsilon)
