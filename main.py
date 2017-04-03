import os, sys, datetime, shutil
import itertools as it
import numpy as np, tensorflow as tf
import util, tfutil, datasets, models, maskers
from holster import H
from dynamite import D

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("base_output_dir", "/Tmp/cooijmat/gan", "root directory under which runs will be stored")
tf.flags.DEFINE_string("basename", "", "base name for run")
tf.flags.DEFINE_string("hp", "", "hyperparameter string")
tf.flags.DEFINE_bool("resume", False, "resume training from previous checkpoint")

def main(argv=()):
  assert not argv[1:]

  config = H(data_dir="/Tmp/cooijmat/mscoco",
             base_output_dir=FLAGS.base_output_dir,
             basename=FLAGS.basename,
             resume=FLAGS.resume)
  config.hp = H(util.parse_hp(FLAGS.hp))
  print str(config.hp)
  config.label = util.make_label(config)
  dirname = "%s_%s" % (datetime.datetime.now().isoformat(), config.label)
  dirname = dirname[:255] # >:-(((((((((((((((((((((((((((((((((((((((((
  config.output_dir = os.path.join(config.base_output_dir, dirname)

  data = datasets.MscocoTF(config)
  config.hp.caption.depth = data.caption_depth

  # NOTE: all hyperparameters must be set at this point
  prepare_run_directory(config)

  model = models.Model(config.hp)

  config.hp.masker.image_size = config.hp.image.size # -_-
  config.masker = maskers.make(config.hp.masker.kind, hp=config.hp.masker)

  config.global_step = tf.Variable(0, name="global_step", trainable=False)
  with tf.name_scope("train"):
    trainer = Trainer(data, model, config)
  tf.get_variable_scope().reuse_variables()
  with tf.name_scope("valid"):
    evaluator = Evaluator(data, model, config)

  earlystopper = EarlyStopper(config)
  supervisor = tf.train.Supervisor(logdir=config.output_dir, summary_op=None)
  with supervisor.managed_session() as session:
    while True:
      global_step = tf.train.global_step(session, config.global_step)

      if supervisor.should_stop():
        print "supervisor says should stop"
        break
      if earlystopper.should_stop(global_step):
        print "earlystopper says should stop"
        break

      trainer(session, supervisor)

      if global_step % config.hp.validate.interval == 0:
        values = evaluator(session, supervisor)
        print "%5i loss:%10f loss asked:%10f loss given:%10f" % (
          global_step, values.model.loss, values.model.loss_asked, values.model.loss_given)
        earlystopper(global_step, values.model.loss, lambda: session.run(trainer.graph.lr_decay_op))

      if global_step >= config.hp.num_steps:
        print "hp.num_steps reached"
        break

class EarlyStopper(object):
  def __init__(self, config):
    self.config = config
    self.best_loss = None
    self.reset_time = 0
    self.stale_time = 0
    self.saver = tf.train.Saver()

  def __call__(self, step, loss, decayer):
    if self.best_loss is None or loss < self.best_loss:
      self.best_loss = loss
      # don't save too often if we're steadily improving
      if step - self.stale_time > self.config.hp.validate.interval:
        self.saver.save(session,
                        os.path.join(self.config.output_dir,
                                     "best_%i_%s.ckpt" % (step, loss)),
                        global_step=self.config.global_step)
      self.reset_time = step
      self.stale_time = step
    elif step - self.reset_time > self.config.hp.lr.patience:
      decayer()
      self.reset_time = step

  def should_stop(self, step):
    return step - self.stale_time > 2 * self.config.hp.lr.patience

class Trainer(object):
  def __init__(self, data, model, config):
    self.data = data
    self.config = config
    self.model = model
    self.graph = make_training_graph(data, model, config, fold="train")

    self.summaries = []
    self.summaries.extend(self.graph.summaries)
    for parameter, gradient in util.equizip(self.graph.parameters, self.graph.gradients):
      self.summaries.append(
        tf.summary.scalar("gradmla_%s" %
                          parameter.name.replace(":", "_"),
                          tfutil.meanlogabs(gradient)))
    for parameter in tf.trainable_variables():
      self.summaries.append(
        tf.summary.scalar("mla_%s" % parameter.name.replace(":", "_"),
                          tfutil.meanlogabs(parameter)))
    self.graph.summary_op = tf.summary.merge(self.summaries)

    self.feed_dicts = it.chain.from_iterable(
      self.data.get_feed_dicts(self.graph.inputs, "train", batch_size=self.config.hp.batch_size, shuffle=True)
      for _ in it.count(0))

  def __call__(self, session, supervisor):
    feed_dict = dict(next(self.feed_dicts))
    feed_dict.update(self.config.masker.get_feed_dict(self.config.hp.batch_size))
    values = self.graph.Narrow("model.loss train_op summary_op").FlatCall(
      session.run, feed_dict=feed_dict)
    supervisor.summary_computed(session, values.summary_op)
    if np.isnan(values.model.loss):
      raise ValueError("nan encountered")
    return values

class Evaluator(object):
  def __init__(self, data, model, config):
    self.data = data
    self.config = config
    self.model = model
    self.graph = make_evaluation_graph(data, model, config, fold="valid")

    self.summaries = list(self.graph.summaries)
    self.graph.summary_op = tf.summary.merge(self.summaries)

  def __call__(self, session, supervisor):
    # validate on one batch
    batch_size = 10 * self.config.hp.batch_size
    feed_dict = dict(next(self.data.get_feed_dicts(self.graph.inputs, "valid", batch_size=batch_size, shuffle=False)))
    feed_dict.update(self.config.masker.get_feed_dict(batch_size))
    values = self.graph.Narrow("model.loss model.loss_given model.loss_asked summary_op").FlatCall(
      session.run, feed_dict=feed_dict)
    supervisor.summary_computed(session, values.summary_op)
    return values

def make_graph(data, model, config, fold="valid"):
  h = H()
  h.inputs = data.get_variables([data.get_tfrecord_path(fold)], config.hp.batch_size)
  h.mask = config.masker.get_variable(tf.shape(h.inputs.image)[0])
  h.global_step = config.global_step
  h.model = model(image=h.inputs.image, mask=h.mask,
                  caption=h.inputs.caption, caption_length=h.inputs.caption_length)

  if D.train:
    h.lr = tf.Variable(config.hp.lr.init, name="learning_rate", trainable=False, dtype=tf.float32)
    tf.summary.scalar("learning_rate", h.lr)
    h.lr_decay_op = tf.assign(h.lr, config.hp.lr.decay * h.lr)

    h.loss = h.model.loss
    h.parameters = tf.trainable_variables()
    h.gradients = tf.gradients(h.loss, h.parameters)
    h.optimizer = tf.train.AdamOptimizer(h.lr)
    h.train_op = h.optimizer.apply_gradients(
      util.equizip(h.gradients, h.parameters),
      global_step=h.global_step)

  h.summaries = []
  h.summaries.extend(
    tf.summary.scalar(k, v)
    for k, v in h.Narrow("model.loss model.loss_given model.loss_asked").Items())

  if not D.train:
    h.summaries.extend([
      tf.summary.image("real", h.model.x, max_outputs=3),
      tf.summary.image("fake", h.model.xhat, max_outputs=3),
      tf.summary.image("realfake", tf.cast(h.mask * tf.cast(h.model.x, tf.float32) +
                                           (1 - h.mask) * tf.cast(h.model.xhat, tf.float32),
                                           tf.uint8),
                       max_outputs=3),
      tf.summary.image("mask", h.mask, max_outputs=3),
      tf.summary.image("entropies", h.model.entropies, max_outputs=3),
    ])

  return h

def make_training_graph(*args, **kwargs):
  kwargs.setdefault("fold", "train")
  with D.Bind(train=True):
    return make_graph(*args, **kwargs)

def make_evaluation_graph(*args, **kwargs):
  kwargs.setdefault("fold", "valid")
  with D.Bind(train=False):
    return make_graph(*args, **kwargs)

def prepare_run_directory(config):
  # FIXME instead make a flag resume_from, load hyperparameters from there
  if not config.resume:
    if tf.gfile.Exists(config.output_dir):
      tf.gfile.DeleteRecursively(config.output_dir)
  if not tf.gfile.Exists(config.output_dir):
    tf.gfile.MakeDirs(config.output_dir)
  if not config.resume:
    with open(os.path.join(config.output_dir, "hp.conf"), "w") as f:
      f.write(util.serialize_hp(config.hp, outer_separator="\n"))
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)),
                    os.path.join(config.output_dir, "code"))

if __name__ == "__main__":
  tf.app.run()
