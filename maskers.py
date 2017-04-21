import itertools as it
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

# bernoulli but with rectangular neighborhood around each cell
class ContiguishMasker(BaseMasker):
  key = "contiguish"

  def __init__(self, hp):
    self.hp = hp

  @property
  def didjs(self):
    return list(it.product(range(-self.hp.radius, self.hp.radius + 1), repeat=2))

  def get_value(self, batch_size):
    didjs = self.didjs
    # We want to mimic OrderlessMasker which has uniform mask size, but we can't control the mask
    # size easily and must instead use independent Bernoullis to get a mask wich we then shift
    # around and conjoin. With independent Bernoullis the mask size follows a binomial distribution,
    # but by putting a uniform (0, 1) prior on the Bernoulli probability we miraculously end up with
    # a uniform mask size.
    p = np.random.random()
    # Adjust for the fact that we're conjoining len(didjs) Bernoulli variables.
    p **= (1. / len(didjs))
    mask = np.random.random((batch_size, self.hp.image.size, self.hp.image.size, self.hp.image.depth)) < p
    return np.prod([np.roll(mask, (di, dj), (0, 1)) for di, dj in didjs], axis=0)

  def get_variable(self, batch_size):
    didjs = self.didjs
    p = np.random.random()
    p **= (1. / len(didjs))
    mask = (tf.random_uniform([batch_size,
                               self.hp.image.size + 2 * self.hp.radius,
                               self.hp.image.size + 2 * self.hp.radius,
                               self.hp.image.depth])
            < p)
    mask = tf.to_float(mask)
    # product over shifts in all directions
    mask = tf.reduce_prod([
      mask[:,
           self.hp.radius + di:self.hp.radius + di + self.hp.image.size,
           self.hp.radius + dj:self.hp.radius + dj + self.hp.image.size,
           :]
      for di, dj in didjs],
      axis=0)
    return mask

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
