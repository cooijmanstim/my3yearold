import sys, os, gzip, cPickle as pkl
import numpy as np, tensorflow as tf
import util

from tensorflow.python.summary import event_accumulator as ea

paths = sys.argv[1:]

things = []
for path in paths:
  with open(os.path.join(path, "hp.conf")) as hpfile:
    hp = util.parse_hp(hpfile.read())

  events = ea.EventAccumulator(path)
  events.Reload()

  tag = "valid/model.loss_asked"
  try:
    losses = [e.value for e in events.Scalars(tag=tag)]
  except KeyError:
    print "no %s in %s" % (tag, path)
    losses = []

  things.append(dict(path=path, losses=losses, hp=hp))

#things.sort(key=lambda thing: -min(thing["losses"]))
#for thing in things:
#  print "%10f %s" % (thing["loss"], thing["path"])

with gzip.open("summary.pkl.gz", "w") as file:
  pkl.dump(things, file, protocol=pkl.HIGHEST_PROTOCOL)
