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

  def __call__(self, latent, context, caption):
    def residual_block(h, scope=None, **conv_layer_kwargs):
      with tf.variable_scope(scope or "res"):
        h_residual = h
        h = tfutil.conv_layer(h, scope="pre",  fn=lambda x: x, **conv_layer_kwargs)
        h = tf.nn.relu(h)
        h = tfutil.conv_layer(h, scope="post", fn=lambda x: x, **conv_layer_kwargs)
        h = tf.nn.relu(h + h_residual)
        return h
  
    resize = tf.image.resize_bilinear
  
    h = tf.concat([tfutil.toconv(latent, depth=64, height=4, width=4, scope="z2h"),
                   tfutil.toconv(caption, depth=64, height=4, width=4, scope="c2h"),
                   resize(context, [4, 4])], axis=3)
  
    for size in [8, 16, 32, 64]:
      # residual_block can't change depth, need to add intermediate layers to map concatenated
      # features back to 128
      h = tfutil.conv_layer(h, depth=128, radius=3, scope="bareuh%i" % size)
  
      h = residual_block(h, depth=128, radius=3, scope="res%i" % size)
      h = resize(h, tf.shape(h)[1:3] * 2)
      h = tf.concat([h, resize(context, [size, size])], axis=3)
  
    for i in range(3):
      h = residual_block(h, depth=128 + 3, radius=3, scope="postres%i" % i)
  
    x = tfutil.conv_layer(h, depth=context.shape[3], radius=3, fn=lambda x: x, scope="h2x")
    x = tf.nn.tanh(x)
  
    return H(output=x)

class SillyDtor(Dtor):
  key = "silly"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, image, caption):
    icl = [0]
    def cl(x, depth, downsample=False, **conv_layer_kwargs):
      conv_layer_kwargs.setdefault("scope", "conv%i" % icl[0])
      conv_layer_kwargs.setdefault("radius", 3)
      x = tfutil.conv_layer(x, depth=depth, **conv_layer_kwargs)
      if downsample:
        x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
      icl[0] += 1
      return x
  
    xh = image
    xh = cl(xh, depth=128, downsample=True) # 32x32
    xh = cl(xh, depth=128, downsample=True) # 16x16
    xh = cl(xh, depth=128, downsample=True) #  8x8
    xh = cl(xh, depth=128, downsample=True) #  4x4
    ch = tfutil.toconv(caption, depth=64, height=4, width=4, scope="c2h")
    h = tf.concat([xh, ch], axis=3)
    h = cl(h, depth=128, downsample=True) #  2x2
    y = tfutil.fromconv(h, depth=1, scope="h2y", normalize=False)
    return H(output=y)

class BidirRtor(Rtor):
  key = "bidir"

  def __init__(self, hp):
    self.hp = hp

  def __call__(self, caption, caption_length):
    hp = self.hp
    h = H()
    h.caption = tf.one_hot(caption, hp.data_dim)
    h.length = caption_length
    h.cell_fw = cells.make(hp.cell.kind, num_units=hp.cell.size, normalize=hp.cell.normalize, scope="fw")
    h.cell_bw = cells.make(hp.cell.kind, num_units=hp.cell.size, normalize=hp.cell.normalize, scope="bw")
    h.outputs, h.states = tf.nn.bidirectional_dynamic_rnn(
      h.cell_fw, h.cell_bw, h.caption, sequence_length=h.length,
      initial_state_fw=[tf.tile(s[None, :], [tf.shape(h.caption)[0], 1]) for s in h.cell_fw.initial_state_parameters],
      initial_state_bw=[tf.tile(s[None, :], [tf.shape(h.caption)[0], 1]) for s in h.cell_bw.initial_state_parameters],
      scope="birnn")
    h.output = tf.concat([output[:, -1] for output in h.outputs], axis=1)
    return h

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
      h.rtor = self.rtor(inputs.caption, inputs.caption_length)
      h.rtor.parameters = tfutil.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    with tf.variable_scope("gtor") as scope:
      # FIXME mscoco assumption
      h.context_mask = np.ones((64, 64), dtype=float)
      h.context_mask[16:48, 16:48] = 0
  
      h.context = inputs.image * h.context_mask[None, :, :, None]
      h.gtor = self.gtor(inputs.latent, h.context, h.rtor.output)
      h.gtor.parameters = tfutil.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    h.real = inputs.image
    h.fake = h.gtor.output

    with tf.variable_scope("dtor") as scope:
      h.dtor.real = self.dtor(h.real, h.rtor.output)
      h.dtor.parameters = tfutil.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    with tf.variable_scope("dtor", reuse=True):
      h.dtor.fake = self.dtor(h.fake, h.rtor.output)

    h.dtor.loss = tf.reduce_mean( h.dtor.fake.output - h.dtor.real.output)
    h.gtor.loss = tf.reduce_mean(-h.dtor.fake.output)

    h.gtor.parameters += h.rtor.parameters
    h.dtor.parameters += h.rtor.parameters
    assert h.gtor.parameters
    return h
