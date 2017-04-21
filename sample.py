import os, sys, datetime, shutil
import itertools as it
import progressbar
import numpy as np, tensorflow as tf
import scipy.misc
import util, tfutil, datasets, models, maskers
from holster import H
from dynamite import D

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("checkpoint", None, "path to ckpt")
tf.flags.DEFINE_string("strategy", "", "sampling strategy (e.g. independent_gibbs or orderless_ancestral)")
tf.flags.DEFINE_string("basename", "", "base name for run")
tf.flags.DEFINE_integer("num_samples", 20, "number of samples to generate")
tf.flags.DEFINE_float("temperature", 1., "softmax temperature")

# generate based on validation samples.
# TODO allow user to provide image paths and captions.
# TODO experiment with generating from scratch (i.e. image captioning)
# TODO experiment with greedy/antigreedy ordering

def main(argv=()):
  assert not argv[1:]

  checkpoint_dir = os.path.dirname(FLAGS.checkpoint)
  hp_string = open(os.path.join(checkpoint_dir, "hp.conf")).read()

  config = H(data_dir="/Tmp/cooijmat/mscoco",
             basename=FLAGS.basename,
             num_samples=FLAGS.num_samples,
             temperature=FLAGS.temperature)
  config.hp = H(util.parse_hp(hp_string))
  print str(config.hp)

  dirname = "sample_%s_%s_%s_T%s" % (config.basename,
                                     FLAGS.strategy,
                                     datetime.datetime.now().isoformat(),
                                     config.temperature)
  dirname = dirname[:255] # >:-(((((((((((((((((((((((((((((((((((((((((
  config.output_dir = dirname

  if not tf.gfile.Exists(config.output_dir):
    tf.gfile.MakeDirs(config.output_dir)

  data = datasets.MscocoNP(config)
  config.hp.caption.depth = data.caption_depth

  model = models.Model(config.hp)

  config.hp.masker.image = H(config.hp.image) # -_-
  config.masker = maskers.make(config.hp.masker.kind, hp=config.hp.masker)

  with D.Bind(train=False):
    graph = make_graph(data, model, config)

  saver = tf.train.Saver()
  session = tf.Session()
  saver.restore(session, FLAGS.checkpoint)

  def predictor(image, mask):
    feed_dict={
      graph.inputs.image: image,
      graph.inputs.caption: original.caption,
      graph.inputs.caption_length: original.caption_length,
      graph.mask: mask,
    }
    values = graph.Narrow("model.pxhat").FlatCall(session.run, feed_dict=feed_dict)
    return values.model.pxhat

  config.predictor = predictor
  sampler = Strategy.make(FLAGS.strategy, config)

  original = next(data.get_batches(data.get_filenames("valid"),
                                   batch_size=config.num_samples, shuffle=False))

  xs = original.image
  with bamboo.scope("original"):
    masks = np.ones(xs.shape).astype(np.float32)
    bamboo.log(x=xs, mask=masks)

  masks = (maskers
           .ContiguousMasker(H(image=config.hp.masker.image, size=32))
           .get_value(config.num_samples))
  xs = masks * xs
  with bamboo.scope("masked"):
    bamboo.log(x=xs, mask=masks)

  xhats, masks = sampler(xs, masks)

  with bamboo.scope("final"):
    bamboo.log(x=xhats, mask=masks)

  for i, (caption, x, xhat) in enumerate(util.equizip(
      original.caption, xs, xhats)):
    scipy.misc.imsave(os.path.join(config.output_dir, "%i_original.png" % i), x)
    scipy.misc.imsave(os.path.join(config.output_dir, "%i_sample.png" % i), xhat)
    with open(os.path.join(config.output_dir, "%i_caption.txt" % i), "w") as file:
      file.write(data.tokenizer.decode(caption))

  bamboo.dump(os.path.join(config.output_dir, "log.npz"))


def bamboo_scope(label, subsample_factor=None):
  def decorator(fn):
    def wrapped_fn(*args, **kwargs):
      with bamboo.scope(label, subsample_factor=subsample_factor):
        return fn(*args, **kwargs)
    return wrapped_fn
  return decorator


def make_graph(data, model, config, fold="valid"):
  h = H()
  h.inputs = data.get_variables()
  h.mask = tf.placeholder(tf.float32, [None, config.hp.image.size,
                                       config.hp.image.size,
                                       config.hp.masker.image.depth], name="mask")
  h.model = model(image=h.inputs.image, mask=h.mask, caption=h.inputs.caption,
                  caption_length=h.inputs.caption_length)
  return h

def orderless_selector(mask, pxhat):
  # sample uniformly among masked-out variables
  return util.sample((1 - mask).reshape([mask.shape[0], -1]),
                     axis=1, onehot=True).reshape(mask.shape)

def greedy_selector(mask, pxhat, sign=-1):
  # choose variable with lowest entropy
  entropies = -(pxhat * np.log(np.where(pxhat > 0, pxhat, 1))).sum(axis=4)
  # ensure already known variables are never chosen
  scores = entropies - sign * np.where(mask, np.inf, 0)
  flat_scores = scores.reshape([scores.shape[0], -1])
  flat_selection = util.onehot_argmax(sign * flat_scores, axis=1)
  return np.reshape(flat_selection, mask.shape)

def antigreedy_selector(mask, pxhat):
  return greedy_selector(mask, pxhat, sign=+1)

class BaseSampler(object):
  pass

class AncestralSampler(BaseSampler):
  def __init__(self, predictor, selector, temperature=1.):
    self.predictor = predictor
    self.selector = selector
    self.temperature = temperature

  @bamboo_scope("ancestral")
  def __call__(self, x, mask):
    x, mask = x.copy(), mask.copy()
  
    count = np.unique((1 - mask).sum(axis=(1,2,3)))
    assert count.size == 1 # FIXME not if inside gibbs

    with bamboo.scope("sequence", subsample_factor=100):
      with progressbar.ProgressBar(maxval=count) as bar:
        i = 0
        while not mask.all():
          pxhat = self.predictor(x, mask)
          xhat = util.sample(pxhat, axis=4, temperature=self.temperature)

          selection = self.selector(mask, pxhat)
          assert selection.any(axis=(1, 2, 3)).all() # FIXME not if inside gibbs?

          x = np.where(selection, xhat, x)
          bamboo.log(x=x, mask=selection, pxhat=pxhat)
          mask = np.where(selection, 1., mask)

          i += 1
          bar.update(i)

    return x, mask

class IndependentSampler(BaseSampler):
  def __init__(self, predictor, temperature=1.):
    self.predictor = predictor
    self.temperature = temperature

  @bamboo_scope("independent")
  def __call__(self, x, mask):
    x, mask = x.copy(), mask.copy()
  
    pxhat = self.predictor(x, mask)
    xhat = util.sample(pxhat, axis=4, temperature=self.temperature)
  
    x = np.where(mask, x, xhat)
    bamboo.log(x=x, mask=mask, pxhat=pxhat)
    mask = np.ones_like(mask)
    return x, mask

class UniformSampler(BaseSampler):
  @bamboo_scope("uniform")
  def __call__(self, x, mask):
    xhat = np.random.randint(256, size=x.shape)
    x = np.where(mask, x, xhat)
    bamboo.log(x=x, mask=mask)
    mask = np.ones_like(mask)
    return x, mask

class GibbsSampler(BaseSampler):
  def __init__(self, sampler, schedule, num_steps=None):
    self.sampler = sampler
    self.schedule = schedule
    self.num_steps = num_steps

  @bamboo_scope("gibbs")
  def __call__(self, x, mask):
    count = np.unique((1 - mask).sum(axis=(1,2,3)))
    assert count.size == 1 # FIXME not if gibbs within gibbs
  
    num_steps = count if self.num_steps is None else self.num_steps

    x, _ = UniformSampler()(x, mask)
  
    with bamboo.scope("sequence", subsample_factor=100):
      with progressbar.ProgressBar(maxval=num_steps) as bar:
        for i in range(num_steps):
          pm = self.schedule(i, num_steps)
          inner_mask = np.random.random(mask.shape) < pm
          x, _ = self.sampler(x, np.logical_or(mask, inner_mask).astype(np.float32))
          bar.update(i)

    return x, np.ones_like(mask)

class Strategy(util.Factory):
  def __call__(self, x, mask):
    return self.sampler(x, mask)

class OrderlessAncestralStrategy(Strategy):
  key = "orderless_ancestral"

  def __init__(self, config):
    self.sampler = AncestralSampler(predictor=config.predictor,
                                    selector=orderless_selector,
                                    temperature=config.temperature)

class GreedyAncestralStrategy(Strategy):
  key = "greedy_ancestral"

  def __init__(self, config):
    self.sampler = AncestralSampler(predictor=config.predictor,
                                    selector=greedy_selector,
                                    temperature=config.temperature)

class AntigreedyAncestralStrategy(Strategy):
  key = "antigreedy_ancestral"

  def __init__(self, config):
    self.sampler = AncestralSampler(predictor=config.predictor,
                                    selector=antigreedy_selector,
                                    temperature=config.temperature)

class IndependentGibbsStrategy(Strategy):
  key = "independent_gibbs"

  def __init__(self, config):
    self.sampler = GibbsSampler(sampler=IndependentSampler(predictor=config.predictor,
                                                           temperature=config.temperature),
                                schedule=yao_schedule,
                                num_steps=None)

def yao_schedule(i, n, pmax=0.9, pmin=0.1, alpha=0.4):
  wat = (pmax - pmin) * i / n
  return max(pmin, pmax - wat / alpha)

bamboo = util.Bamboo()

if __name__ == "__main__":
  tf.app.run()
