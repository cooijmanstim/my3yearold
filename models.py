import numpy as np, tensorflow as tf
import util, tfutil, cells
from holster import H

class Convnet(util.Factory): pass
class Reader(util.Factory): pass

class RecurrentReader(Reader):
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
      h.output = tf.concat(h.output, axis=2)
    else:
      h.cell = cells.make(hp.cell.kind, num_units=hp.cell.size, normalize=hp.cell.normalize, scope="fw")
      batch_size = tf.shape(x)[0]
      h.output, h.state = tf.nn.dynamic_rnn(
        h.cell, x, sequence_length=length,
        initial_state=[tf.tile(s[None, :], [batch_size, 1]) for s in h.cell.initial_state_parameters],
        scope="rnn")
    h.output_length = length
    h.summary = h.output[:, -1]
    return h

class ConvReader(Reader):
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

class CompositeReader(Reader):
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

class QuasiReader(CompositeReader):
  key = "quasi"

  def __init__(self, hp):
    children = [("conv", ConvReader(H(radius=hp.radius, size=3 * hp.size))),
                ("rnn", RecurrentReader(H(bidir=hp.bidir, cell=H(kind="quasi", size=hp.size, normalize=hp.normalize))))]
    super(QuasiReader, self).__init__(hp, children=children)

class StraightConvnet(Convnet):
  key = "straight"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, x, cb):
    hp = self.hp
    for i in range(hp.profundity):
      x = tfutil.residual_block(x, depth=hp.depth, radius=hp.radius, separable=hp.separable, scope="res%i" % i)
      x = cb(i, x)
    return H(output=x)

class Merger(util.Factory):
  def __init__(self, hp):
    self.hp = hp

  def should_merge(self, i):
    return i in map(int, self.hp.layers.split(","))

  def __call__(self, i, x, reader):
    if not self.should_merge(i):
      return x
    with tf.variable_scope("merger%i" % i):
      return self.merge(x, reader)

class ConvMerger(Merger):
  key = "conv"

  def merge(self, x, reader):
    z = reader.summary
    z = tfutil.toconv(z, height=x.shape[1], width=x.shape[2], depth=self.hp.depth)
    return tf.stack([x, z], axis=3)

class AttentionMerger(Merger):
  key = "attention"

  def merge(self, x, reader):
    z = reader.output
    xdepth, zdepth = tfutil.get_depth(x), tfutil.get_depth(z)
    # attention weights by inner product
    u = tf.get_variable("u", shape=[xdepth, zdepth],
                        initializer=tf.uniform_unit_scaling_initializer())
    w = tf.einsum("bhwi,ij,btj->bhwt", x, u, z)
    w = tf.nn.softmax(w)
    # attend
    y = tf.einsum("bhwt,btd->bhwd", w, z)
    if xdepth != zdepth:
      y = tfutil.conv_layer(y, depth=xdepth, normalize=False, fn=lambda x: x)
    return x + y

class Model(object):
  def __init__(self, hp):
    self.hp = hp
    self.reader = Reader.make(hp.reader.kind, hp.reader)
    self.convnet = Convnet.make(hp.convnet.kind, hp.convnet)
    self.merger = Merger.make(hp.merger.kind, hp.merger)

  def __call__(self, image, mask, caption, caption_length):
    hp = self.hp
    h = H()

    with tf.variable_scope("reader") as scope:
      h.reader = self.reader(tf.one_hot(caption, hp.caption.depth),
                             length=caption_length)

    h.x = image
    h.px = tf.one_hot(h.x, hp.image.levels)
    h.mask = mask

    h.context = tf.concat([tfutil.collapse(h.px * mask[:, :, :, :, None],
                                           [0, 1, 2, [3, 4]]),
                           h.mask], axis=3)

    with tf.variable_scope("convnet") as scope:
      h.convnet = self.convnet(h.context, lambda i, x: self.merger(i, x, h.reader))

    h.exhat = tf.reshape(
      tfutil.conv_layer(h.convnet.output, radius=1, depth=3 * hp.image.levels,
                        fn=lambda x: x, separable=hp.convnet.separable, scope="exhat"),
      tf.shape(h.px))
    h.pxhat = tf.nn.softmax(h.exhat)
    h.xhat = tf.cast(tf.argmax(h.exhat, axis=4), tf.uint8)

    h.losses = tfutil.softmax_xent(labels=h.px, logits=h.exhat)
    # divide by number of variables masked out; this is the number of conditionals being trained
    h.weights = 1. / tf.reduce_sum(1 - h.mask, axis=[1, 2, 3], keep_dims=True)

    h.loss_total = tf.reduce_mean(h.losses * h.weights)
    h.loss_given = tf.reduce_sum(h.losses * h.weights *      h.mask ) / tf.reduce_sum(    h.mask)
    h.loss_asked = tf.reduce_sum(h.losses * h.weights * (1 - h.mask)) / tf.reduce_sum(1 - h.mask)

    h.loss = h.loss_total if hp.optimize_given else h.loss_asked

    h.entropies = -tf.reduce_sum(tfutil.softmax_xent(labels=h.pxhat, logits=h.exhat), axis=3, keep_dims=True)
    return h
