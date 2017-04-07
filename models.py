from __future__ import division
import numbers, functools as ft
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

class StraightResidualConvnet(Convnet):
  key = "straight_residual"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, x, merger, mergee):
    hp = self.hp
    for i in range(0, hp.profundity, 2):
      x = tfutil.residual_block(x, depth=hp.depth, radius=hp.radius, scope="res%i" % i)
      x = merger(i, x, mergee)
    return H(output=x)

class StraightConvnet(Convnet):
  key = "straight"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, x, merger, mergee):
    hp = self.hp
    for i in range(hp.profundity):
      x = tfutil.conv_layer(x, depth=hp.depth, radius=hp.radius, scope="conv%i" % i)
      x = merger(i, x, mergee)
    return H(output=x)

class StraightDilatedConvnet(Convnet):
  key = "straight_dilated"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, x, merger, mergee):
    hp = self.hp

    # try to do the right thing
    h, w = x.get_shape().as_list()[1], x.get_shape().as_list()[2]
    assert h == w
    ndilations = int(round(np.log2(h) - 1))
    dilation_interval = int(round(hp.profundity / ndilations))

    dilation = 1

    def batch_to_space(x):
      if dilation == 1: return x
      return tf.batch_to_space_nd(x, [dilation, dilation], tf.zeros([2, 2], dtype=tf.int32))

    def space_to_batch(x):
      if dilation == 1: return x
      return tf.space_to_batch_nd(x, [dilation, dilation], tf.zeros([2, 2], dtype=tf.int32))
  
    for i in range(hp.profundity):
      if i != 0 and i % dilation_interval == 0:
        x = batch_to_space(x)
        dilation *= 2
        x = space_to_batch(x)

      x = tfutil.conv_layer(x, depth=hp.depth, radius=hp.radius, scope="conv%i" % i)
      if merger.should_merge(i):
        x = batch_to_space(x)
        x = merger(i, x, mergee)
        x = space_to_batch(x)

    x = batch_to_space(x)
    return H(output=x)

class Merger(util.Factory):
  def __init__(self, hp):
    self.hp = hp
    if isinstance(self.hp.layers, numbers.Integral):
        self.layers = [self.hp.layers]
    elif isinstance(self.hp.layers, basestring):
        self.layers = list(map(int, self.hp.layers.split(",")))
    else:
        raise ValueError()

  def should_merge(self, i):
    return i in self.layers

  def __call__(self, i, x, caption):
    if not self.should_merge(i):
      return x
    with tf.variable_scope("merger%i" % i):
      return self.merge(x, caption)

class ConvMerger(Merger):
  key = "conv"

  def merge(self, x, caption):
    z = caption.summary
    x_shape = x.get_shape().as_list()
    z = tfutil.toconv(z, height=x_shape[1], width=x_shape[2], depth=self.hp.depth)
    return tf.concat([x, z], axis=3)

class AttentionMerger(Merger):
  key = "attention"

  def merge(self, x, caption):
    depth = tfutil.get_depth(x)
    z = tf.reshape(tfutil.layer([tfutil.collapse(caption.output, [[0, 1], 2])], depth=depth),
                   [tf.shape(caption.output)[0], tf.shape(caption.output)[1], depth])
    # attention weights by inner product
    u = tf.get_variable("u", initializer=tf.constant(np.eye(depth).astype(np.float32)))
    w = tf.einsum("bhwi,ij,btj->bhwt", x, u, z)
    w = tf.nn.softmax(w)
    # attend
    y = tf.einsum("bhwt,btd->bhwd", w, z)
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
      h.convnet = self.convnet(h.context, self.merger, h.reader)

    h.exhat = tf.reshape(
      tfutil.conv_layer(h.convnet.output, radius=1, depth=3 * hp.image.levels,
                        fn=lambda x: x, scope="exhat"),
      tf.shape(h.px))
    h.pxhat = tf.nn.softmax(h.exhat)
    h.xhat = tf.cast(tf.argmax(h.exhat, axis=4), tf.uint8)

    lossfn = (dict(xent=tfutil.softmax_xent,
                   emd=ft.partial(tfutil.softmax_emd, distance=tf.abs),
                   emd2=ft.partial(tfutil.softmax_emd, distance=tf.square))
              [hp.loss])

    h.losses = lossfn(labels=h.px, logits=h.exhat)

    h.loss_total = tf.reduce_mean(h.losses)
    h.loss_given = tf.reduce_sum(h.losses *      h.mask  / tf.reduce_sum(    h.mask))
    h.loss_asked = tf.reduce_sum(h.losses * (1 - h.mask) / tf.reduce_sum(1 - h.mask))

    h.loss = h.loss_total if hp.optimize_given else h.loss_asked

    h.entropies = -tf.reduce_sum(tfutil.softmax_xent(labels=h.pxhat, logits=h.exhat), axis=3, keep_dims=True)
    return h
