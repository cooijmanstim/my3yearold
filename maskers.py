import numpy as np, tensorflow as tf
import util

def make(key, *args, **kwargs):
  return BaseMasker.make(key, *args, **kwargs)

class BaseMasker(util.Factory): pass

class CenterMasker(BaseMasker):
  key = "center"

  def __init__(self, hp):
    self.hp = hp

  def get_variable(self, batch_size):
    l = (self.hp.image.size - self.hp.size) // 2
    b = l + self.hp.size
    mask = np.ones((1, self.hp.image.size, self.hp.image.size, self.hp.image.depth), dtype=float)
    mask[:, l:b, l:b] = 0
    return tf.constant(mask)

  def get_feed_dict(self, batch_size):
    return {}

class ContiguousMasker(BaseMasker):
  key = "contiguous"

  def __init__(self, hp):
    self.hp = hp

  def get_value(self, batch_size):
    mask = np.ones([batch_size, self.hp.image.size, self.hp.image.size, self.hp.image.depth], dtype=np.float32)
    topleft = np.random.randint(self.hp.image.size - self.hp.size, size=[batch_size, 2])
    vertical = topleft[:, 0, None] + np.arange(self.hp.size)[None, :]
    horizontal = topleft[:, 1, None] + np.arange(self.hp.size)[None, :]
    mask[np.arange(batch_size)[:, None, None], vertical[:, :, None], horizontal[:, None, :], :] = 0.
    assert np.allclose((1 - mask).sum(axis=(0, 1, 2)), batch_size * self.hp.size**2)
    return mask

  def get_variable(self, batch_size):
    r = tf.range(self.hp.image.size)
    ls = tf.random_uniform([batch_size, 2], maxval=self.hp.image.size - self.hp.size, dtype=tf.int32)
    us = ls + self.hp.size
    r = r[None, :]
    ls, us = ls[:, :, None], us[:, :, None]
    vertmask = ~((ls[:, 0] <= r) & (r < us[:, 0]))
    horzmask = ~((ls[:, 1] <= r) & (r < us[:, 1]))
    areamask = tf.to_float(vertmask[:, :, None, None] | horzmask[:, None, :, None])
    return tf.tile(areamask, [1, 1, 1, self.hp.image.depth])

  def get_feed_dict(self, batch_size):
    return {}

class OrderlessMasker(BaseMasker):
  key = "orderless"

  def __init__(self, hp):
    self.hp = hp

  def get_variable(self, batch_size):
    B, H, W, D = batch_size, self.hp.image.size, self.hp.image.size, self.hp.image.depth
    mask_size = tf.random_uniform([B], maxval=H * W * D, dtype=tf.int64)
    # generate a binary mask with `mask_size` ones, then shuffle the ones into random places. Note
    # batch axis comes second because `tf.random_shuffle` only works along the first axis -_-
    mask = tf.to_float(tf.range(H * W * D, dtype=tf.int64)[:, None] < mask_size[None, :])
    mask = tf.random_shuffle(mask)
    mask = tf.reshape(tf.transpose(mask), [B, H, W, D])
    return mask

  def get_feed_dict(self, batch_size):
    return {}

class FunctionMasker(BaseMasker):
  key = None

  def __init__(self, hp, fn):
    self.hp = hp
    self.fn = fn

  @util.memo
  def get_variable(self, batch_size):
    return tf.placeholder(tf.float32, [None, self.hp.image.size, self.hp.image.size, self.hp.image.depth], name="mask")

  def get_feed_dict(self, batch_size):
    return {self.get_variable(): self.fn(batch_size, hp=self.hp)}
