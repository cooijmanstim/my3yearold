import os, subprocess
import cPickle as pkl
from collections import OrderedDict as ordict
import itertools as it, functools as ft
import numpy as np, tensorflow as tf
import tfutil, cells
from PIL import Image

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "100", "batch size")
tf.flags.DEFINE_bool("resume", False, "resume training from previous checkpoint")

IMAGE_DEPTH = 3
IMAGE_SHAPE = (64, 64)
IMAGE_AREA = int(np.prod(IMAGE_SHAPE))

context_mask = np.ones((64, 64), dtype=float)
context_mask[16:48, 16:48] = 0

data_dir = "/Tmp/cooijmat/mscoco"
if not os.path.exists(os.path.join(data_dir)):
  print "copying data to", data_dir
  os.makedirs(data_dir)
  subprocess.check_call("tar xf".split() +
                        [os.environ["MSCOCO_INPAINTING_TARBALL"]] +
                        "--directory".split() +
                        [data_dir])
  print "done"

filenames = dict(train=tf.gfile.Glob(os.path.join(data_dir, "inpainting", "train2014", "*.jpg")),
                 valid=tf.gfile.Glob(os.path.join(data_dir, "inpainting",   "val2014", "*.jpg")))

caption_dict = pkl.load(open(os.path.join(data_dir, "inpainting", "dict_key_imgID_value_caps_train_and_valid.pkl")))

print "determining alphabet"
alphabet = set(it.chain.from_iterable(it.chain.from_iterable(caption_dict.values())))
alphabet.add("/")
alphabet = dict((character, code) for code, character in enumerate(sorted(alphabet)))
print "done"

def batches(xs, batch_size=1, shuffle=False):
  xs = list(xs)
  np.random.shuffle(xs)
  for start in range(0, len(xs), batch_size):
    yield xs[start:start+batch_size]

def load_file(filename):
  identifier = os.path.splitext(os.path.basename(filename))[0]

  caption_strings = caption_dict[identifier]
  np.random.shuffle(caption_strings)
  caption_string = " ".join(caption_strings)
  caption = np.array([alphabet[c] for c in caption_string], dtype=int)

  image = np.array(Image.fromarray(np.array(Image.open(filename))), dtype=np.float32)
  image /= 255 / 2.
  image -= 1

  if image.ndim == 2:
    # grayscale image; just replicate across channels
    image = image[:, :, None] * np.ones((3,))[None, None, :]

  return image, caption

def load_batch(filenames):
  images, captions = zip(*list(map(load_file, filenames)))
  images = np.array(images)
  caption_lengths = np.array(list(map(len, captions)))
  captions = np.array([padto(caption, max(caption_lengths))
                       for caption in captions])
  return images, captions, caption_lengths

def padto(x, length, axis=0):
  return np.pad(x, [(0, 0) if i is not axis else (0, length - dim)
                    for i, dim in enumerate(x.shape)],
                mode="constant")

def main(argv=()):
  assert not argv[1:]

  output_dir = "/Tmp/cooijmat/gan/debug"
  if not FLAGS.resume:
    if tf.gfile.Exists(output_dir):
      tf.gfile.DeleteRecursively(output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  latent_dim = 64
  num_iterations = 1000

  latent = tf.placeholder(tf.float32, [None, latent_dim], name="latent")
  image = tf.placeholder(tf.float32, [None] + list(IMAGE_SHAPE) + [3], name="image")
  caption = tf.placeholder(tf.int32, [None, None], name="caption")
  caption_length = tf.placeholder(tf.int32, [None], name="caption_length")

  global_step = tf.Variable(0, "global_step")
  variables = construct(image, caption, caption_length, latent, global_step)
  summary_op = tf.summary.merge_all()

  supervisor = tf.train.Supervisor(logdir=output_dir, summary_op=None, global_step=variables["global_step"])
  with supervisor.managed_session() as session:
    while True:
      # infinite batches!
      batchit = it.chain.from_iterable(batches(filenames["train"], batch_size=FLAGS.batch_size, shuffle=True)
                                       for _ in it.count(0))

      if supervisor.should_stop():
        print "should stop"
        break

      def run(variables):
        batch = next(batchit)
        images, captions, caption_lengths = load_batch(batch)
        return session.run(variables,
                           feed_dict={latent: np.random.randn(len(batch), latent_dim),
                                      image: images, caption: captions, caption_length: caption_lengths})

      for _ in range(3):
        def dstep():
          dloss, _ = run([variables["dloss"], variables["dtor_train_op"]])
  
        def gstep():
          gloss, _ = run([variables["gloss"], variables["dtor_train_op"]])
  
        for _ in range(3):
          dstep()
        gstep()

      dloss, summary = run([variables["dloss"], summary_op])
      supervisor.summary_computed(session, summary)
      print "%5i %f" % (global_step.eval(session), dloss)


def construct(image, caption, caption_length, latent, global_step):
  # TODO wonder about whether real/fake should be based on different examples
  context = image * context_mask[None, :, :, None]

  real = image

  with tf.variable_scope("caption"):
    caption_embedding = process_caption(caption, caption_length)

  with tf.variable_scope("generator"):
    fake = generator(latent, context, caption_embedding)

  with tf.variable_scope("discriminator"):
    real_score = discriminator(real, caption_embedding)
  with tf.variable_scope("discriminator", reuse=True):
    fake_score = discriminator(fake, caption_embedding)

  dloss = tf.reduce_mean(fake_score - real_score)
  gloss = tf.reduce_mean(-fake_score)

  caption_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="caption")
  dtor_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
  gtor_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")

  for parameter in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    tf.summary.scalar("norm_%s" % parameter.name, tf.norm(parameter))

  def make_optimizer(loss, parameters, prefix=""):
    parameters = list(parameters)
    optimizer = tf.train.RMSPropOptimizer(1e-4, centered=True)
    gradients = optimizer.compute_gradients(loss, var_list=parameters)
    for gradient, parameter in gradients:
      tf.summary.scalar("%s_gradnorm_%s" % (prefix, parameter.name), tf.norm(gradient))
    return optimizer_apply_gradients(gradients, global_step=global_step)

  dtor_train_op = make_optimizer(dloss, var_list=dtor_parameters + caption_parameters, "dtor_")
  gtor_train_op = make_optimizer(gloss, var_list=gtor_parameters + caption_parameters, "gtor_")

  tf.summary.image("real", real, max_outputs=3)
  tf.summary.image("fake", fake, max_outputs=3)
  tf.summary.scalar("dloss", dloss)
  tf.summary.scalar("gloss", gloss)

  return locals()

def process_caption(caption, length):
  caption = tf.one_hot(caption, len(alphabet))
  # FIXME: LSTM is not lipschitz or is it
  cell_fw = cells.LSTM(num_units=200, normalize=True, scope="fw")
  cell_bw = cells.LSTM(num_units=200, normalize=True, scope="bw")
  outputs, states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw, cell_bw, caption, sequence_length=length,
    initial_state_fw=[tf.tile(s[None, :], [tf.shape(caption)[0], 1]) for s in cell_fw.initial_state_parameters],
    initial_state_bw=[tf.tile(s[None, :], [tf.shape(caption)[0], 1]) for s in cell_bw.initial_state_parameters])
  return tf.concat([output[:, -1] for output in outputs], axis=1)

def generator(z, context, caption):
  icl = [0]
  def cl(x, depth, upsample=False, **conv_layer_kwargs):
    conv_layer_kwargs.setdefault("scope", "conv%i" % icl[0])
    conv_layer_kwargs.setdefault("radius", 3)
    if upsample:
      x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * 2)
    x = tfutil.conv_layer(x, depth=depth, **conv_layer_kwargs)
    icl[0] += 1
    return x

  zh = toconv(z, depth=64, height=4, width=4, scope="z2h")
  ch = toconv(caption, depth=64, height=4, width=4, scope="c2h")
  h = tf.concat([zh, ch], axis=3) # fuse z and caption
  h = cl(h, depth=128, upsample=True) #  8x8
  h = cl(h, depth=128, upsample=True) # 16x16
  h = cl(h, depth=128, upsample=True) # 32x32
  h = cl(h, depth=128, upsample=True) # 64x64
  h = tf.concat([h, context], axis=3) # fuse z/caption and context
  # FIXME residual connections
  h = cl(h, depth=128)
  h = cl(h, depth=128)
  x = cl(h, depth=IMAGE_DEPTH, fn=tf.nn.tanh, scope="h2x", normalize=False)
  return x

def discriminator(x, caption):
  icl = [0]
  def cl(x, depth, downsample=False, **conv_layer_kwargs):
    conv_layer_kwargs.setdefault("scope", "conv%i" % icl[0])
    conv_layer_kwargs.setdefault("radius", 3)
    x = tfutil.conv_layer(x, depth=depth, **conv_layer_kwargs)
    if downsample:
      x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    icl[0] += 1
    return x

  xh = cl(x, depth=128)
  xh = cl(xh, depth=128, downsample=True) # 32x32
  xh = cl(xh, depth=128, downsample=True) # 16x16
  xh = cl(xh, depth=128, downsample=True) #  8x8
  xh = cl(xh, depth=128, downsample=True) #  4x4
  ch = toconv(caption, depth=64, height=4, width=4, scope="c2h")
  h = tf.concat([xh, ch], axis=3)
  h = cl(h, depth=128, downsample=True) #  2x2
  h = cl(h, depth=128, downsample=True) #  1x1
  h = fromconv(h, depth=1, scope="h2y", normalize=False)
  return h

def toconv(x, depth, height, width, **project_terms_kwargs):
  return tf.reshape(tfutil.project_terms([x], depth=depth * height * width,
                                         **project_terms_kwargs),
                    [tf.shape(x)[0], height, width, depth])

def fromconv(x, depth, **project_terms_kwargs):
  return tfutil.project_terms([tf.reshape(x, [tf.shape(x)[0], int(np.prod(x.shape[1:]))])],
                              depth=depth, **project_terms_kwargs)

if __name__ == "__main__":
  tf.app.run()
