import re, os, shutil, hashlib, cPickle as pkl, nltk, tarfile, time, collections
from collections import OrderedDict as ordict
import functools as ft, itertools as it, operator as op
import tensorflow as tf, numpy as np
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
    self.load()

  @property
  def caption_depth(self):
    return len(self.tokenmap)

  def ensure_local_copy(self):
    if not tf.gfile.Exists(self.tfrecord_dir):
      print "copying data to", self.config.data_dir
      tf.gfile.MakeDirs(self.config.data_dir)
      shutil.copytree(os.environ["MSCOCO_TFRECORD_DIR"], self.tfrecord_dir)
      print "done"

  def load(self):
    self.tokenmap = pkl.load(open(os.path.join(self.tfrecord_dir, "tokenmap_%s.pkl" % self.config.hp.caption.token)))

  def get_tfrecord_path(self, fold):
    return os.path.join(self.config.data_dir, "tfrecord", "%s.tfrecords" % fold)

  def get_variables(self, filenames, batch_size, num_epochs=None):
    with tf.name_scope('input'):
      reader = tf.TFRecordReader()

      filename_queue = tf.train.string_input_producer(
        list(filenames), num_epochs=num_epochs, name="filequeue")

      record_queue_capacity = 10000
      record_queue = tf.RandomShuffleQueue(capacity=record_queue_capacity,
                                           min_after_dequeue=record_queue_capacity // 2,
                                           dtypes=[tf.string], name="shufflequeue")
      enqueue_ops = []
      for _ in range(2):
        _, record = reader.read(filename_queue)
        enqueue_ops.append(record_queue.enqueue([record]))
      tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
          record_queue, enqueue_ops))
      tf.summary.scalar(
          "queue/%s/fraction_of_%d_full" % (record_queue.name, record_queue_capacity),
          tf.cast(record_queue.size(), tf.float32) / record_queue_capacity)

      serialized_example = record_queue.dequeue()
      context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        context_features={
          "image/data": tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
          "image/caption_characters": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "image/caption_words":      tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

      image = tf.image.decode_jpeg(context["image/data"], channels=Mscoco.IMAGE_DEPTH)
      image.set_shape([Mscoco.IMAGE_HEIGHT, Mscoco.IMAGE_WIDTH, Mscoco.IMAGE_DEPTH])

      caption = sequence["image/caption_%ss" % self.config.hp.caption.token]
      caption_length = tf.shape(caption)[0]

      singular = H(image=image, caption=caption, caption_length=caption_length)

      plural = singular.FlatCall(tf.train.batch, batch_size=batch_size, num_threads=2,
                                 capacity=30 * batch_size, dynamic_pad=True,
                                 allow_smaller_final_batch=True)

      return plural

  def get_feed_dicts(self, *args, **kwargs):
    return it.repeat({})

class MscocoNP(Mscoco):
  def __init__(self, config):
    self.config = config
    self.numpy_dir = os.path.join(self.config.data_dir, "numpy")
    self.ensure_local_copy()
    self.load()

  @property
  def caption_depth(self):
    return len(self.tokenizer.tokenmap)

  def ensure_local_copy(self):
    if not tf.gfile.Exists(self.numpy_dir):
      print "copying data to", self.numpy_dir
      tf.gfile.MakeDirs(self.numpy_dir)
      # FIXME ow we have a double toplevel dir
      tarfile.open(os.environ["MSCOCO_INPAINTING_TARBALL"], 'r:bz2').extractall(self.numpy_dir)
      print "done"

  def get_caption_string(self, identifier):
    return "|".join(self.captions[identifier])

  def get_tokenizer(self, token):
    cache_path = os.path.join(self.numpy_dir,
                              "tokenizer_%s.pkl" % self.config.hp.caption.token)
    try:
      with open(cache_path, "rb") as cache:
        tokenizer = pkl.load(cache)
    except IOError:
      print "determining tokenmap"
      start = time.time()
      tokenizer = Tokenizer.make(token)
      tokenizer.prepare("|".join(candidates) for candidates in self.captions.values())
      end = time.time()
      print "NP %s tokenmap checksum %s (%f seconds)" % (token, tokenizer.checksum, end - start)
      with open(cache_path, "wb") as cache:
        pkl.dump(tokenizer, cache)
    return tokenizer

  def load(self):
    self.folds = dict(train=self.get_filenames("train"),
                      valid=self.get_filenames("valid"))
    self.captions = pkl.load(open(os.path.join(self.numpy_dir, "inpainting",
                                               "dict_key_imgID_value_caps_train_and_valid.pkl")))
    self.tokenizer = self.get_tokenizer(self.config.hp.caption.token)

  def get_variables(self):
    h = H()
    h.image = tf.placeholder(tf.uint8, [None, Mscoco.IMAGE_HEIGHT, Mscoco.IMAGE_WIDTH,
                                        Mscoco.IMAGE_DEPTH], name="image")
    h.caption = tf.placeholder(tf.int32, [None, None], name="caption")
    h.caption_length = tf.placeholder(tf.int32, [None], name="caption_length")
    return h

  @util.memo
  def get_filenames(self, fold):
    fold = dict(train="train", valid="val")[fold] # -_-
    return tf.gfile.Glob(os.path.join(self.numpy_dir, "inpainting", "%s2014" % fold, "*.jpg"))

  def get_batches(self, filenames, batch_size, shuffle=False):
    return iter(map(self.load_batch, util.batches(filenames, batch_size=batch_size, shuffle=shuffle)))

  def get_feed_dicts(self, placeholders, filenames, batch_size, shuffle=False):
    return iter(map(ft.partial(self.get_feed_dict, placeholders),
                    self.get_batches(filenames, batch_size, shuffle=shuffle)))

  def get_feed_dict(self, placeholders, values):
    return dict(placeholders.Zip(values))

  def load_batch(self, filenames):
    images, captions = zip(*list(map(self.load_file, filenames)))
    h = H()
    h.image = np.array(images)
    h.caption_length = np.array(list(map(len, captions)))
    h.caption = np.array([util.padto(np.array(caption, dtype=int),
                                     max(h.caption_length))
                          for caption in captions])
    return h

  def load_file(self, filename):
    identifier = os.path.splitext(os.path.basename(filename))[0]
    caption = self.tokenizer.process(self.get_caption_string(identifier))
    image = np.array(Image.fromarray(np.array(Image.open(filename))), dtype=np.uint8)
    if image.ndim == 2:
      # grayscale image; just replicate across channels
      image = image[:, :, None] * np.ones((Mscoco.IMAGE_DEPTH,))[None, None, :]
    return image, caption

class Tokenizer(util.Factory):
  def process(self, string):
    tokens = self.tokenize(string)
    codes = [self.tokenmap[t] for t in tokens]
    return codes

  def decode(self, codes):
    tokens = [self.inverse_tokenmap[c] for c in codes]
    return self.join(tokens)

  @property
  def checksum(self):
    return hashlib.md5("".join(self.tokenmap.keys())).hexdigest()

class WordTokenizer(Tokenizer):
  key = "word"

  def prepare(self, strings):
    counts = collections.Counter(self.tokenize(" ".join(strings)))
    tokens = list(reversed(sorted(counts.keys(), key=counts.__getitem__)))
    rare_tokens = set(it.takewhile(lambda token: counts[token] < 5, tokens))
    common_tokens = set(tokens) - rare_tokens

    tokenmap = ordict((token, code) for code, token in enumerate(common_tokens))
    inverse_tokenmap = ordict((code, token) for code, token in enumerate(common_tokens))

    # map rare tokens to a common <UNK> code
    tokenmap.update((token, len(common_tokens)) for token in rare_tokens)
    inverse_tokenmap[len(common_tokens)] = "<UNK>"

    self.tokenmap = tokenmap
    self.inverse_tokenmap = inverse_tokenmap

  def tokenize(self, s):
    return nltk.word_tokenize(
      re.sub(r"[^\w\s.]+", r" ",
             re.sub(r"([.!?|])+", r" \1 ", s)))

  def join(self, tokens):
    return " ".join(tokens)

class CharacterTokenizer(Tokenizer):
  key = "character"

  def prepare(self, strings):
    tokens = set(self.tokenize("".join(strings)))
    self.tokenmap = ordict((token, code) for code, token in enumerate(tokens))
    self.inverse_tokenmap = ordict((code, token) for code, token in enumerate(common_tokens))

  def tokenize(self, s):
    return list(s)

  def join(self, tokens):
    return "".join(tokens)
