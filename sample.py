import os, sys, datetime, shutil
import itertools as it
import numpy as np, tensorflow as tf
import util, tfutil, datasets, models, maskers
from holster import H
from dynamite import D

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("checkpoint", None, "path to directory containing ckpt")
tf.flags.DEFINE_string("basename", "", "base name for run")
tf.flags.DEFINE_integer("num_samples", 100, "number of samples to generate")
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
  config.label = util.make_label(config)

  dirname = "sample_%s_%s" % (datetime.datetime.now().isoformat(), config.label)
  dirname = dirname[:255] # >:-(((((((((((((((((((((((((((((((((((((((((
  config.output_dir = dirname

  if not tf.gfile.Exists(config.output_dir):
    tf.gfile.MakeDirs(config.output_dir)

  data = datasets.MscocoNP(config)
  config.hp.caption.depth = data.caption_depth

  model = models.Model(config.hp)

  config.hp.masker.image_size = config.hp.image.size # -_-
  config.masker = maskers.make(config.hp.masker.kind, hp=config.hp.masker)

  sampler = Sampler(data, model, config)

  saver = tf.train.Saver()
  session = tf.Session()
  saver.restore(session, FLAGS.checkpoint)

  sampler(session, data)

class Sampler(object):
  def __init__(self, data, model, config):
    self.data = data
    self.config = config
    self.model = model
    self.graph = make_graph(data, model, config)

  def __call__(self, session, supervisor):
    mask = maskers.ContiguousMasker(H(image_size=64, size=32)).get_value(self.config.num_samples)
    original = next(self.data.get_batches(self.data.get_filenames("valid"),
                                          batch_size=self.config.num_samples, shuffle=False))

    def predictor(image, mask):
      feed_dict={
        self.graph.inputs.image: image,
        self.graph.inputs.caption: original.caption,
        self.graph.inputs.caption_length: original.caption_length,
        self.graph.mask: mask,
      }
      values = self.graph.Narrow("model.pxhat").FlatCall(session.run, feed_dict=feed_dict)
      return values.model.pxhat

    sample = sample_ancestral(predictor, mask, original, config=self.config)
    for i, caption, x, xhat, logp in enumerate(util.equizip(
        original.caption, original.image, sample.image, sample.logp)):
      scipy.misc.imsave("%i_original.png" % i, x)
      scipy.misc.imsave("%i_sample.png" % i, xhat)
      open("%i_caption.txt" % i).write(caption)
      np.savez_compressed("%i_logp.npz" % i, logp)

def sample_ancestral(predictor, mask, original, config):
  x = original.image.copy()
  mask = mask.copy()
  logp = np.zeros(x.shape, dtype=np.float32)

  B, H, W, D = mask.shape

  while not mask.all():
    pxhat = predictor(image=x, mask=mask)
    xhat = util.sample(pxhat, axis=4, temperature=config.temperature)

    # choose variable to sample, by sampling according to the normalized mask. this is uniform as
    # all masked out variables have equal positive weight.
    selection = util.sample(mask.reshape([B, H * W * D]), axis=1, onehot=True).reshape(mask.shape)

    x = np.where(selection, xhat, x)
    mask = np.where(selection, 1., mask)
    logp = np.where(selection,
                    np.log(pxhat[np.arange(pxhat.shape[0])[:, None, None, None],
                                 np.arange(pxhat.shape[1])[None, :, None, None],
                                 np.arange(pxhat.shape[2])[None, None, :, None],
                                 np.arange(pxhat.shape[3])[None, None, None, :],
                                 xhat]),
                    0.)

  return original.With(image=x, logp=logp)

def make_graph(data, model, config, fold="valid"):
  h = H()
  h.inputs = data.get_variables()
  h.mask = tf.placeholder(tf.float32, [None, config.hp.image.size,
                                       config.hp.image.size,
                                       config.hp.image.depth], name="mask")
  h.model = model(image=h.inputs.image, mask=h.mask, caption=h.inputs.caption,
                  caption_length=h.inputs.caption_length)
  return h

if __name__ == "__main__":
  tf.app.run()
