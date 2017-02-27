import numbers
import numpy as np, tensorflow as tf
import holster

FF_NORM_AXIS = 0
BOUND_WEIGHTS = False

# the current holster interface fails late on nonexistent keys (e.g. typos). this sucks, mitigate
# the damage by catching mistakes early.
def holster_guard(value, dtype=None, name=None, as_ref=False):
  assert isinstance(value, holster.BaseHolster)
  raise ValueError("holster passed into tensorflow function: %s" % value)
tf.register_tensor_conversion_function(holster.BaseHolster, holster_guard)

def get_collection(key, scope=None):
  # https://github.com/tensorflow/tensorflow/issues/7719
  if isinstance(scope, tf.VariableScope):
    scope = scope.name + "/"
  return tf.get_collection(key, scope=scope)

def maybe_bound_weights(w):
  if BOUND_WEIGHTS:
    w = tf.nn.tanh(w)
  return w

def layers(xs, sizes, scope=None, **layer_kwargs):
  xs = list(xs)
  with tf.variable_scope(scope or "layers", values=xs):
    for i, size in enumerate(sizes):
      with tf.variable_scope(str(i), values=xs):
        xs = [layer(xs, output_dim=size, **layer_kwargs)]
    return xs[0]

def layer(xs, fn=tf.nn.relu, normalize=True, bias=True, **project_terms_kwargs):
  project_terms_kwargs.setdefault("normalize", normalize)
  project_terms_kwargs.setdefault("bias", bias)
  return fn(project_terms(xs, **project_terms_kwargs))

def project_terms(xs, depth=None, normalize=True, bias=True, scope=None):
  xs = list(xs)
  with tf.variable_scope(scope or "project_terms", values=xs):
    if normalize:
      # batch-normalize each projection separately before summing
      projected_xs = [project(x, depth=depth, bias=False, scope=str(i))
                      for i, x in enumerate(xs)]
      normalized_xs = [batch_normalize(x, beta=0., axis=FF_NORM_AXIS, scope=str(i))
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
    depth = x.get_shape().as_list()[-1]
    if gamma is None:
      gamma = tf.get_variable("gamma", shape=depth, initializer=tf.constant_initializer(0.1))
      gamma = maybe_bound_weights(gamma)
    if beta is None:
      beta = tf.get_variable("beta", shape=depth, initializer=tf.constant_initializer(0))
    if False:
      # tf.nn.moments doesn't deal well with dynamic shapes
      mean = tf.reduce_mean(x, axis=axis, keep_dims=True)
      variance = tf.reduce_mean((x - mean)**2, axis=axis, keep_dims=True)
      return beta + gamma / tf.sqrt(variance + epsilon) * (x - mean)
    else:
      axis = [axis] if isinstance(axis, numbers.Integral) else axis
      mean, variance = tf.nn.moments(x, axes=axis)
      return tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=epsilon)

def conv_layer(x, radius=None, stride=1, padding="SAME", depth=None, fn=tf.nn.relu, normalize=True, bias=True, separable=True, separable_multiplier=3, scope=None):
  with tf.variable_scope(scope or "conv", []):
    input_depth = x.get_shape().as_list()[-1]
    if separable:
      dw = tf.get_variable("dw", shape=[radius, radius, input_depth, separable_multiplier],
                           initializer=tf.uniform_unit_scaling_initializer())
      pw = tf.get_variable("pw", shape=[1, 1, separable_multiplier * input_depth, depth],
                           initializer=tf.uniform_unit_scaling_initializer())
      dw = maybe_bound_weights(dw)
      pw = maybe_bound_weights(pw)
      y = tf.nn.separable_conv2d(x, dw, pw, strides=[1, stride, stride, 1], padding=padding)
    else:
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

def residual_block(h, scope=None, depth=None, fn=tf.nn.relu, **conv_layer_kwargs):
  input_depth = h.shape[-1]
  if not depth:
    depth = input_depth
  conv_layer_kwargs["depth"] = depth
  conv_layer_kwargs["fn"] = lambda x: x
  with tf.variable_scope(scope or "res"):
    if depth == input_depth:
      h_thru = h
    else:
      h_thru = conv_layer(h, scope="thru", bias=False, **conv_layer_kwargs)
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
