import os, sys, datetime, shutil
import itertools as it
import numpy as np, tensorflow as tf
import util, tfutil, datasets, models, maskers
from holster import H
from dynamite import D

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "/Tmp/cooijmat/mscoco", "directory to place local copy of data")
tf.flags.DEFINE_string("base_output_dir", ".", "root directory under which runs will be stored")
tf.flags.DEFINE_string("basename", "", "base name for run")
tf.flags.DEFINE_string("hp", "", "hyperparameter string")
tf.flags.DEFINE_bool("resume", False, "resume training from previous checkpoint")
tf.flags.DEFINE_float("trace_fraction", 0.001, "how often to trace graph execution and dump timeline")

def main(argv=()):
  assert not argv[1:]

  config = H(data_dir=FLAGS.data_dir,
             base_output_dir=FLAGS.base_output_dir,
             basename=FLAGS.basename,
             resume=FLAGS.resume,
             trace_fraction=FLAGS.trace_fraction)
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
  trainer = Trainer(data, model, config)
  tf.get_variable_scope().reuse_variables()
  evaluator = Evaluator(data, model, config)

  earlystopper = EarlyStopper(config, trainer.graph.lr_decay_op)
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
      sys.stdout.write("\r%i " % global_step)
      sys.stdout.flush()

      if global_step % config.hp.validate.interval == 0:
        values = evaluator(session, supervisor)
        print "%5i loss:%10f loss asked:%10f loss given:%10f" % (
          global_step, values.model.loss, values.model.loss_asked, values.model.loss_given)
        earlystopper.track(global_step, values.model.loss, session)

      if global_step >= config.hp.num_steps:
        print "hp.num_steps reached"
        break

class EarlyStopper(object):
  def __init__(self, config, decay_op):
    self.config = config
    self.decay_op = decay_op
    self.best_loss = None
    self.reset_time = 0
    self.stale_time = 0
    self.saver = tf.train.Saver(max_to_keep=1)
    # validation estimate is noisy, so filter it
    self.filter = util.MovingMedianFilter(10)

  def track(self, step, loss, session):
    filtered_loss = self.filter(loss)
    if self.best_loss is None or filtered_loss < self.best_loss:
      self.best_loss = filtered_loss
      # don't save too often if we're steadily improving
      if step - self.stale_time > self.config.hp.validate.interval:
        self.saver.save(session,
                        os.path.join(self.config.output_dir,
                                     "best_%i_%s.ckpt" % (step, loss)),
                        global_step=self.config.global_step)
      self.reset_time = step
      self.stale_time = step
    elif step - self.reset_time > self.config.hp.lr.patience:
      session.run(self.decay_op)
      self.reset_time = step

  def should_stop(self, step):
    return False # this is finicky and we don't really care
    return step - self.stale_time > 2 * self.config.hp.lr.patience

class Trainer(object):
  def __init__(self, data, model, config):
    self.data = data
    self.config = config
    self.model = model
    self.graph = self.make_graph()

    self.feed_dicts = it.chain.from_iterable(
      self.data.get_feed_dicts(self.graph.inputs, "train", batch_size=self.config.hp.batch_size, shuffle=True)
      for _ in it.count(0))

  def make_graph(self):
    with tf.name_scope("train"):
      graph = make_training_graph(self.data, self.model, self.config, fold="train")
    summaries = [var for var in tf.get_collection(tf.GraphKeys.SUMMARIES)
                      if var.name.startswith("train")]
    for parameter, gradient in util.equizip(graph.parameters, graph.gradients):
      summaries.append(
        tf.summary.scalar("gradmla_%s" %
                          parameter.name.replace(":", "_"),
                          tfutil.meanlogabs(gradient)))
    for parameter in tf.trainable_variables():
      summaries.append(
        tf.summary.scalar("mla_%s" % parameter.name.replace(":", "_"),
                          tfutil.meanlogabs(parameter)))
    graph.summary_op = tf.summary.merge(summaries)
    return graph

  def __call__(self, session, supervisor):
    def tracing_run(*args, **kwargs):
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      result = session.run(*args, options=run_options, run_metadata=run_metadata, **kwargs)
      from tensorflow.python.client import timeline
      tl = timeline.Timeline(run_metadata.step_stats)
      with open(os.path.join(self.config.output_dir, 'timeline.json'), 'w') as f:
        f.write(tl.generate_chrome_trace_format())
      print "trace written"
      return result

    feed_dict = dict(next(self.feed_dicts))
    feed_dict.update(self.config.masker.get_feed_dict(self.config.hp.batch_size))

    runner = tracing_run if np.random.rand() < self.config.trace_fraction else session.run
    values = self.graph.Narrow("model.loss train_op summary_op").FlatCall(runner, feed_dict=feed_dict)

    supervisor.summary_computed(session, values.summary_op)
    if np.isnan(values.model.loss):
      raise ValueError("nan encountered")
    return values

class Evaluator(object):
  def __init__(self, data, model, config):
    self.data = data
    self.config = config
    self.model = model
    self.graph = self.make_graph()

  def make_graph(self):
    with tf.name_scope("valid"):
      graph = make_evaluation_graph(self.data, self.model, self.config, fold="valid")
    summaries = [var for var in tf.get_collection(tf.GraphKeys.SUMMARIES)
                 if var.name.startswith("valid")]
    graph.summary_op = tf.summary.merge(summaries)
    return graph

  def __call__(self, session, supervisor):
    aggregates = H({"model.loss": util.MeanAggregate(),
                    "model.loss_given": util.MeanAggregate(),
                    "model.loss_asked": util.MeanAggregate(),
                    "summary_op": util.LastAggregate()})
    batch_size = 10 * self.config.hp.batch_size
    # spend at most 1/16 of the time validating
    max_num_batches = self.config.hp.validate.interval // 16

    for _, feed_dict in zip(range(max_num_batches),
                         self.data.get_feed_dicts(self.graph.inputs, "valid", batch_size=batch_size, shuffle=False)):
      feed_dict = dict(feed_dict)
      feed_dict.update(self.config.masker.get_feed_dict(batch_size))
      values = self.graph.Narrow("model.loss model.loss_given model.loss_asked summary_op").FlatCall(
        session.run, feed_dict=feed_dict)
      for aggregate, value in aggregates.Zip(values):
        aggregate(value)

    values = H((key, aggregate.value) for key, aggregate in aggregates.Items())
    supervisor.summary_computed(session, values.summary_op)
    for key, value in values.Items():
      # summary_ops return strings?? the plot thickens -__-
      if not isinstance(value, basestring):
        value = tf.Summary(value=[tf.Summary.Value(tag="valid/%s" % key, simple_value=value)])
      supervisor.summary_computed(session, value)
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

  if D.train:
    for k, v in h.Narrow("model.loss model.loss_given model.loss_asked").Items():
      tf.summary.scalar(k, v)
  else:
    tf.summary.image("real", h.model.x, max_outputs=3)
    tf.summary.image("fake", h.model.xhat, max_outputs=3)
    tf.summary.image("fake_asked", tf.cast((1 - h.mask) * tf.cast(h.model.xhat, tf.float32), tf.uint8),
                     max_outputs=3)
    tf.summary.image("realfake", tf.cast(h.mask * tf.cast(h.model.x, tf.float32) +
                                         (1 - h.mask) * tf.cast(h.model.xhat, tf.float32),
                                         tf.uint8),
                     max_outputs=3)
    tf.summary.image("mask", h.mask, max_outputs=3)
    tf.summary.image("entropies", h.model.entropies, max_outputs=3)

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
