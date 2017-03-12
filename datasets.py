import os, shutil, hashlib, cPickle as pkl
from collections import OrderedDict as ordict
import functools as ft, itertools as it
import tensorflow as tf
from PIL import Image
import util, tfutil
from holster import H

class Mscoco(object):
  IMAGE_HEIGHT = 64
  IMAGE_WIDTH = 64
  IMAGE_DEPTH = 3
  LEVELS = 256

class MscocoTF(Mscoco):
  def __init__(self, config):
    self.config = config
    self.tfrecord_dir = os.path.join(self.config.data_dir, "tfrecord")
    self.ensure_local_copy()
    self.alphabet = pkl.load(open(os.path.join(self.tfrecord_dir, "alphabet.pkl")))
    print "TF alphabet md5sum %s" % hashlib.md5("".join(self.alphabet.keys())).hexdigest()

  def ensure_local_copy(self):
    if not tf.gfile.Exists(self.tfrecord_dir):
      print "copying data to", self.config.data_dir
      tf.gfile.MakeDirs(self.config.data_dir)
      shutil.copytree(os.environ["MSCOCO_TFRECORD_DIR"], self.tfrecord_dir)
      print "done"

  def get_tfrecord_path(self, fold):
    return os.path.join(self.config.data_dir, "tfrecord", "%s.tfrecords" % fold)

  def get_variables(self, filenames, batch_size, num_epochs=None):
    with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer(
          list(filenames), num_epochs=num_epochs)
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          # in TF land, a variable-length string is a fixed-length scalar
          features=dict(image=tf.FixedLenFeature([], tf.string),
                        caption=tf.FixedLenFeature([], tf.string),
                        identifier=tf.FixedLenFeature([], tf.string)))
      image = tf.image.decode_jpeg(features["image"], channels=Mscoco.IMAGE_DEPTH)
      image.set_shape([Mscoco.IMAGE_HEIGHT, Mscoco.IMAGE_WIDTH, Mscoco.IMAGE_DEPTH])

      caption = tf.decode_raw(features["caption"], tf.uint8)
      caption_length = tf.shape(caption)[0]

      singular = H(image=image, caption=caption, caption_length=caption_length)
  
      # FIXME: batch doesn't do shuffling :-( but it does do dynamic padding
      plural = singular.FlatCall(
        ft.partial(tf.train.batch, batch_size=batch_size, num_threads=2,
                   capacity=30 * batch_size, dynamic_pad=True, allow_smaller_final_batch=True))
  
      return plural

  def get_feed_dicts(self, *args, **kwargs):
    return it.repeat({})

class MscocoNP(Mscoco):
  def __init__(self, config):
    self.config = config
    self.ensure_local_copy()
    self.load()

  def ensure_local_copy(self):
    if not tf.gfile.Exists(self.config.data_dir):
      print "copying data to", self.config.data_dir
      tf.gfile.MakeDirs(self.config.data_dir)
      subprocess.check_call("tar xf".split() +
                            [os.environ["MSCOCO_INPAINTING_TARBALL"]] +
                            "--directory".split() +
                            [self.config.data_dir])
      print "done"

  def load(self):
    folds = dict(train=tf.gfile.Glob(os.path.join(self.config.data_dir, "inpainting", "train2014", "*.jpg")),
                 valid=tf.gfile.Glob(os.path.join(self.config.data_dir, "inpainting",   "val2014", "*.jpg")))
    captions = pkl.load(open(os.path.join(self.config.data_dir, "inpainting", "dict_key_imgID_value_caps_train_and_valid.pkl")))
  
    print "determining alphabet"
    # NOTE: when we get the test set, make sure it doesn't change the alphabet, or else retrain the model
    alphabet = set(it.chain.from_iterable(it.chain.from_iterable(caption_dict.values())))
    alphabet.add("|") # we use the pipe character to concatenate multiple captions for one image
    alphabet = ordict((character, code) for code, character in enumerate(sorted(alphabet)))
    print "NP alphabet md5sum %s" % hashlib.md5("".join(alphabet.keys())).hexdigest()

    self.folds = folds
    self.captions = captions
    self.alphabet = alphabet

  def get_variables(self):
    h = H()
    h.image = tf.placeholder(tf.uint8, [None, Mscoco.IMAGE_HEIGHT, Mscoco.IMAGE_WIDTH,
                                        Mscoco.IMAGE_DEPTH], name="image")
    h.caption = tf.placeholder(tf.int32, [None, None], name="caption")
    h.caption_length = tf.placeholder(tf.int32, [None], name="caption_length")
    return h

  @util.memo
  def get_filenames(self, fold):
    return tf.gfile.Glob(os.path.join(self.config.data_dir, "inpainting", "%s2014" % fold, "*.jpg"))

  def get_batches(self, filenames, batch_size, shuffle=False):
    return iter(map(self.load_batch, util.batches(filenames, batch_size=batch_size, shuffle=shuffle)))

  def get_feed_dicts(self, placeholders, filenames, batch_size, shuffle=False):
    return iter(map(ft.partial(self.get_feed_dict, placeholders),
                    self.get_batches(filenames, batch_size, shuffle=shuffle)))

  def get_feed_dict(self, placeholders, values):
    return dict(placeholders.Zip(values))

  def load_batch(self, filenames):
    images, captions = zip(*list(map(load_file, filenames)))
    h = H()
    h.images = np.array(images)
    h.caption_lengths = np.array(list(map(len, captions)))
    h.captions = np.array([padto(caption, max(caption_lengths))
                           for caption in captions])
    return h

  def load_file(self, filename):
    identifier = os.path.splitext(os.path.basename(filename))[0]
  
    caption_strings = self.captions[identifier]
    np.random.shuffle(caption_strings)
    caption_string = "|".join(caption_strings)
    caption = np.array([self.alphabet[c] for c in caption_string], dtype=int)
  
    image = np.array(Image.fromarray(np.array(Image.open(filename))), dtype=np.uint8)
  
    if image.ndim == 2:
      # grayscale image; just replicate across channels
      image = image[:, :, None] * np.ones((Mscoco.IMAGE_DEPTH,))[None, None, :]
  
    return image, caption
