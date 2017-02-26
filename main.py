import os
import numpy as np, tensorflow as tf
import util, tfutil, datasets, models
from holster import H

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
  config.output_dir = os.path.join(config.base_output_dir, config.label)

  data = datasets.MscocoTF(config)
  config.alphabet_size = len(data.alphabet)
  config.hp.rtor.data_dim = config.alphabet_size

  # NOTE: all hyperparameters must be set at this point
  util.prepare_run_directory(config)

  model = models.Model(config.hp)
  trainer = Trainer(data, model, config)

  supervisor = tf.train.Supervisor(logdir=config.output_dir, summary_op=None)
  with supervisor.managed_session() as session:
    while not supervisor.should_stop():
      trainer(session, supervisor)
    print "should stop"

class Trainer(object):
  def __init__(self, data, model, config):
    self.data = data
    self.config = config
    self.model = model
    self.graph = H(global_step=tf.Variable(0, name="global_step", trainable=False))
    with tf.variable_scope("graph"):
      self.graph.train = make_training_graph(data, model, config, fold="train",
                                             global_step=self.graph.global_step)
    with tf.variable_scope("graph", reuse=True):
      self.graph.valid = make_evaluation_graph(data, model, config, fold="valid",
                                               global_step=self.graph.global_step)

    self.summaries = H(train=[], valid=[])

    for key in "dtor gtor".split():
      for parameter, gradient in util.equizip(self.graph.train.model[key].parameters,
                                              self.graph.train.model[key].gradients):
        self.summaries.train.append(
          tf.summary.scalar("%s_gradmla_%s" %
                            (key, parameter.name.replace(":", "_")),
                            tfutil.meanlogabs(gradient)))

    for parameter in tf.trainable_variables():
      self.summaries.valid.append(
        tf.summary.scalar("mla_%s" % parameter.name.replace(":", "_"),
                          tfutil.meanlogabs(parameter)))

    self.summaries.valid.extend([
      tf.summary.image("real", self.graph.valid.model.real, max_outputs=3),
      tf.summary.image("fake", self.graph.valid.model.fake, max_outputs=3),
      tf.summary.image("fakeraw", self.graph.valid.model.fakeraw, max_outputs=3),
      tf.summary.scalar("dtor.loss", self.graph.valid.model.dtor.loss),
      tf.summary.scalar("gtor.loss", self.graph.valid.model.gtor.loss)
    ])

    self.graph.train.summary_op = tf.summary.merge(self.summaries.train)
    self.graph.valid.summary_op = tf.summary.merge(self.summaries.valid)

  def __call__(self, session, supervisor):
    for _ in range(3):
      for _ in range(3):
        values = self.graph.train.Narrow("model.dtor.loss model.dtor.train_op summary_op").FlatCall(session.run)
        supervisor.summary_computed(session, values.summary_op)
      for _ in range(3):
        values = self.graph.train.Narrow("model.gtor.loss model.gtor.train_op summary_op").FlatCall(session.run)
        supervisor.summary_computed(session, values.summary_op)
    values = self.graph.valid.Narrow("model.dtor.loss summary_op global_step").FlatCall(session.run)
    supervisor.summary_computed(session, values.summary_op)
    print "%5i %f" % (values.global_step, values.model.dtor.loss)

def make_graph(data, model, config, global_step=None, fold="valid", train=False):
  h = H()
  h.inputs = data.get_variables([data.get_tfrecord_path(fold)], config.hp.batch_size)
  h.inputs.latent = tf.truncated_normal([tf.shape(h.inputs.image)[0], config.hp.latent_dim])
  h.global_step = global_step
  h.model = model(h.inputs)
  for key in "dtor gtor".split():
    submodel = h.model[key]
    if train:
      submodel.gradients = tf.gradients(submodel.loss, submodel.parameters)
      submodel.optimizer = tf.train.RMSPropOptimizer(1e-4, centered=True)
      submodel.train_op = submodel.optimizer.apply_gradients(
        util.equizip(submodel.gradients, submodel.parameters),
        global_step=h.global_step)
  return h

def make_training_graph(*args, **kwargs):
  kwargs.setdefault("fold", "train")
  kwargs.setdefault("train", True)
  return make_graph(*args, **kwargs)

def make_evaluation_graph(*args, **kwargs):
  kwargs.setdefault("fold", "valid")
  kwargs.setdefault("train", False)
  return make_graph(*args, **kwargs)

if __name__ == "__main__":
  tf.app.run()
