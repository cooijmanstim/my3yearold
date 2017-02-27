import numpy as np, tensorflow as tf
import util, tfutil, cells
from holster import H

class Rtor(util.Factory): pass
class Gtor(util.Factory): pass
class Dtor(util.Factory): pass

class SillyGtor(Gtor):
  key = "silly"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, context, caption, latent):
    condition = tfutil.layer([caption, latent], depth=256, scope="condition")
    condition = tfutil.toconv(condition, depth=64, height=4, width=4, scope="c2h")
    with tf.variable_scope("red"):
      context_by_size = reduce_image(context, contexts=[condition], downto=2)
    h = context_by_size[2]
    for size in [4, 8, 16, 32, 64]:
      h = tfutil.residual_block(h, depth=128, radius=3, scope="res%i" % size)
      h = tf.image.resize_bilinear(h, tf.shape(h)[1:3] * 2)
      h = tf.concat([h, context_by_size[size]], axis=3)
    for i in range(2):
      h = tfutil.residual_block(h, depth=128, radius=3, scope="postres%i" % i)
    x = tfutil.conv_layer(h, depth=context.shape[3], radius=3, fn=tf.nn.tanh, scope="h2x")
    return H(output=x)

class SillyDtor(Dtor):
  key = "silly"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, image, caption):
    caption = tfutil.toconv(caption, depth=64, height=4, width=4, scope="c2h")
    with tf.variable_scope("red"):
      h = reduce_image(image, contexts=[caption], downto=2)[2]
    h = tfutil.fromconv(h, depth=128)
    y = tfutil.layer([h], depth=1, scope="h2y", normalize=False)
    return H(output=y)

class RecurrentRtor(Rtor):
  key = "recurrent"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, x, length):
    hp = self.hp
    h = H()
    h.input = x
    h.input_length = length
    if hp.bidir:
      h.cell_fw = cells.make(hp.cell.kind, num_units=hp.cell.size, normalize=hp.cell.normalize, scope="fw")
      h.cell_bw = cells.make(hp.cell.kind, num_units=hp.cell.size, normalize=hp.cell.normalize, scope="bw")
      batch_size = tf.shape(x)[0]
      h.output, h.state = tf.nn.bidirectional_dynamic_rnn(
        h.cell_fw, h.cell_bw, x, sequence_length=length,
        initial_state_fw=[tf.tile(s[None, :], [batch_size, 1]) for s in h.cell_fw.initial_state_parameters],
        initial_state_bw=[tf.tile(s[None, :], [batch_size, 1]) for s in h.cell_bw.initial_state_parameters],
        scope="birnn")
      h.summary = tf.concat([output[:, -1] for output in h.output], axis=1)
    else:
      h.cell = cells.make(hp.cell.kind, num_units=hp.cell.size, normalize=hp.cell.normalize, scope="fw")
      batch_size = tf.shape(x)[0]
      h.output, h.state = tf.nn.dynamic_rnn(
        h.cell, x, sequence_length=length,
        initial_state=[tf.tile(s[None, :], [batch_size, 1]) for s in h.cell.initial_state_parameters],
        scope="rnn")
      h.summary = h.output[:, -1]
    h.output_length = length
    return h

class ConvRtor(Rtor):
  key = "conv"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, x, length):
    hp = self.hp
    h = H()
    h.input = x
    h.input_length = length
    w = tf.get_variable("w", shape=[hp.radius, h.input.shape[-1], hp.size],
                        initializer=tf.uniform_unit_scaling_initializer())
    w = tfutil.maybe_bound_weights(w)
    h.output = tf.nn.conv1d(h.input, w, stride=1, padding="VALID")
    h.output_length = h.input_length - (hp.radius - 1)
    # summarize by global average pooling
    h.summary = tf.reduce_mean(h.output, axis=1)
    return h

class CompositeRtor(Rtor):
  key = None # not constructible from hyperparameters

  def __init__(self, hp, children=()):
    self.hp = hp
    self.children = H(children)

  def __call__(self, x, length):
    hp = self.hp
    h = H()
    h.input = x
    h.input_length = length
    for key, child in self.children.Items():
      h[key] = child(x, length)
      x, length, z = h[key].output, h[key].output_length, h[key].summary
    h.output = x
    h.output_length = length
    h.summary = z
    return h

class QuasiRtor(CompositeRtor):
  key = "quasi"

  def __init__(self, hp):
    children = [("conv", ConvRtor(H(radius=hp.radius, size=3 * hp.size))),
                ("rnn", RecurrentRtor(H(bidir=hp.bidir, cell=H(kind="quasi", size=hp.size, normalize=hp.normalize))))]
    super(QuasiRtor, self).__init__(hp, children=children)

class Model(object):
  def __init__(self, hp):
    self.hp = hp
    self.rtor = Rtor.make(hp.rtor.kind, hp.rtor)
    self.gtor = Gtor.make(hp.gtor.kind, hp.rtor)
    self.dtor = Dtor.make(hp.dtor.kind, hp.rtor)

  def __call__(self, inputs):
    # TODO wonder about whether real/fake should be based on different examples,
    # i.e. should have their own image placeholders
    h = H()

    with tf.variable_scope("rtor") as scope:
      h.rtor = self.rtor(tf.one_hot(inputs.caption, self.hp.data_dim),
                         length=inputs.caption_length)
      h.rtor.parameters = tfutil.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    with tf.variable_scope("gtor") as scope:
      # FIXME mscoco assumption
      h.context_mask = np.ones((1, 64, 64, 1), dtype=float)
      h.context_mask[:, 16:48, 16:48, :] = 0
  
      h.context = inputs.image * h.context_mask
      h.gtor = self.gtor(h.context, h.rtor.summary, inputs.latent)
      h.gtor.parameters = tfutil.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    h.real = inputs.image

    h.fakeraw = h.gtor.output
    h.fake = h.context_mask * h.real + (1 - h.context_mask) * h.fakeraw

    with tf.variable_scope("dtor") as scope:
      h.dtor.real = self.dtor(h.real, h.rtor.summary)
      h.dtor.parameters = tfutil.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    with tf.variable_scope("dtor", reuse=True):
      h.dtor.fake = self.dtor(h.fake, h.rtor.summary)

    h.dtor.loss = tf.reduce_mean( h.dtor.fake.output - h.dtor.real.output)
    h.gtor.loss = tf.reduce_mean(-h.dtor.fake.output)

    h.gtor.parameters += h.rtor.parameters
    h.dtor.parameters += h.rtor.parameters
    assert h.gtor.parameters
    return h

def reduce_image(image, contexts=[], downto=1, depth=128):
  reductions = dict()
  size = 64
  reductions[size] = image
  while size > downto:
    reduction = reductions[size]
    reduction = tfutil.conv_layer(reduction, radius=3, depth=depth, scope="conv%i" % size)
    reduction = tf.nn.max_pool(reduction, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    size //= 2
    # integrate caption/latent when they fit
    reduction = tf.concat([reduction] +
                          [context for context in contexts
                           if context.shape[1] == size and context.shape[2] == size],
                          axis=3)
    reductions[size] = reduction
  return reductions
