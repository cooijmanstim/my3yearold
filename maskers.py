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
    l = (self.hp.image_size - self.hp.size) // 2
    b = l + self.hp.size
    mask = np.ones((1, self.hp.image_size, self.hp.image_size, 1), dtype=float)
    mask[:, l:b, l:b] = 0
    return tf.constant(mask)

  def get_feed_dict(self, batch_size):
    return {}

class ContiguousMasker(BaseMasker):
  key = "contiguous"

  def __init__(self, hp):
    self.hp = hp

  def get_variable(self, batch_size):
    r = tf.range(self.hp.image_size)
    ls = tf.random_uniform([batch_size, 2], maxval=self.hp.image_size - self.hp.size, dtype=tf.int32)
    us = ls + self.hp.size
    r = r[None, :]
    ls, us = ls[:, :, None], us[:, :, None]
    vertmask = ~((ls[:, 0] <= r) & (r < us[:, 0]))
    horzmask = ~((ls[:, 1] <= r) & (r < us[:, 1]))
    return tf.to_float(vertmask[:, :, None, None] | horzmask[:, None, :, None])

  def get_feed_dict(self, batch_size):
    return {}

class FunctionMasker(BaseMasker):
  key = None

  def __init__(self, hp, fn):
    self.hp = hp
    self.fn = fn

  @util.memo
  def get_variable(self, batch_size):
    return tf.placeholder(tf.float32, [None, self.hp.image_size, self.hp.image_size, None], name="mask")

  def get_feed_dict(self, batch_size):
    return {self.get_variable(): self.fn(batch_size, hp=self.hp)}
