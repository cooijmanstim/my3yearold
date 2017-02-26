import numpy as np, tensorflow as tf
import tfutil, util

def make(key, *args, **kwargs):
  return BaseCell.make(key, *args, **kwargs)

class BaseCell(tf.contrib.rnn.BasicRNNCell, util.Factory):
  @property
  def state_placeholders(self):
    return [tf.placeholder(dtype=tf.float32, shape=[None, size]) for size in self.state_size]

  def initial_state(self, batch_size):
    return [np.zeros([batch_size, size]) for size in self.state_size]

  @property
  def initial_state_parameters(self):
    with tf.variable_scope(self.scope):
      return [tf.get_variable("initial_state_%i" % i, shape=[size], initializer=tf.constant_initializer(0))
              for i, size in enumerate(self.state_size)]

  def __call__(self, inputs, state, scope=None):
    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]
    state = self.transition(inputs, state, scope=scope)
    output = self.get_output(state)
    return [output, state]

  def transition(self, inputs, state, scope=None):
    raise NotImplementedError()

  def get_output(self, state):
    raise NotImplementedError()

class LSTM(BaseCell):
  key = "lstm"

  def __init__(self, num_units, forget_bias=5.0, activation=tf.nn.tanh, normalize=False, scope=None):
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.activation = activation
    self.normalize = normalize
    self.scope = scope if scope is not None else "lstm-%x" % id(self)

  @property
  def state_size(self):
    return 2 * [self.num_units]

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    return state[1]

  def transition(self, inputs, state, scope=None):
    with tf.variable_scope(scope or self.scope):
      c, h = state
      total_input = tfutil.project_terms([h] + inputs,
                                         depth=4 * self.num_units,
                                         normalize=self.normalize,
                                         scope="ijfo")
      i, j, f, o = tf.split(total_input, 4, axis=1)
      f += self.forget_bias

      new_c = tf.nn.sigmoid(f) * c + tf.nn.sigmoid(i) * self.activation(j)
      output_c = new_c
      if self.normalize:
        output_c = tfutil.batch_normalize(output_c, scope="c")
      new_h = tf.nn.sigmoid(o) * self.activation(output_c)
      new_c = output_c
    return [new_c, new_h]

class QRNN(BaseCell):
  key = "quasi"

  def __init__(self, num_units, forget_bias=5.0, activation=tf.nn.tanh, normalize=False, scope=None):
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.activation = activation
    self.normalize = normalize
    self.scope = scope if scope is not None else "qrnn-%x" % id(self)

  @property
  def state_size(self):
    return 2 * [self.num_units]

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    return state[1]

  def transition(self, inputs, state, scope=None):
    with tf.variable_scope(scope or self.scope):
      c, h = state
      if self.normalize:
        inputs = tfutil.batch_normalize(inputs, scope="jfo")
      f, o, j = tf.split(inputs, 3, axis=2)
      f += self.forget_bias
      new_c = tf.nn.sigmoid(f) * c + (1 - tf.nn.sigmoid(f)) * self.activation(j)
      output_c = new_c
      if self.normalize:
        output_c = tfutil.batch_normalize(output_c, scope="c")
      new_h = tf.nn.sigmoid(o) * self.activation(output_c)
      new_c = output_c
    return [new_c, new_h]
