import os, subprocess, datetime, numbers
import numpy as np
import util
from holster import H

trainsh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cluster_train.sh")
assert os.path.exists(trainsh_path)

def uniform(a, b):
  if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
    return np.random.random_integers(a, b)
  else:
    return np.random.random() * (b - a) + a

def boolean():
  return np.random.rand() < 0.5

def categorical(xs):
  return np.random.choice(xs)

defaults = H(**util.parse_hp("""
  lr.init=0.001
  lr.decay=0.1
  lr.patience=1000
  validate.interval=100
  num_steps=100000
  masker.kind=orderless
  image.size=64
  image.depth=3
  image.levels=256
  caption.token=word
  reader.kind=quasi
"""))

def sample_hp():
  hp = H(defaults)
  hp.batch_size = uniform(5, 30)
  hp.optimize_given = boolean()
  hp["reader.radius"] = uniform(4, 32)
  hp["reader.size"] = uniform(32, 256)
  hp["reader.bidir"] = boolean()
  hp["reader.normalize"] = boolean()
  hp["convnet.kind"] = categorical("straight straight_residual straight_dilated".split())
  hp["convnet.depth"] = uniform(64, 512)
  hp["convnet.profundity"] = uniform(4, 48)
  hp["convnet.radius"] = uniform(2, 7)
  hp["merger.kind"] = categorical("attention conv".split())
  hp.merger.kind = "conv"
  if hp.merger.kind == "conv":
    hp.merger.depth = uniform(4, 64)
  nmerges = uniform(1, min(5, hp.convnet.profundity))
  layers = np.random.choice(hp.convnet.profundity, size=(nmerges,), replace=False)
  hp["merger.layers"] = ",".join(map(str, sorted(layers)))
  return hp

def main():
  timestamp = datetime.datetime.now().isoformat()

  def abbrev(key):
    return ".".join(parts[:1] for parts in key.split("."))

  # most will run out of memory and die early
  n = 50
  for i in range(n):
    hp = sample_hp()
    basename = ",".join("%s=%s" % (abbrev(k), v) for k, v in hp.Items())
    hpfile_path = "%s_%s_hp.conf" % (timestamp, basename[:200])
    with open(hpfile_path, "wb") as hpfile:
      hpfile.write(util.serialize_hp(hp))
    subprocess.check_call(
      ["jobdispatch", "--gpu", "--mem=16G", "--raw=#PBS -l feature=k80",
       "bash", trainsh_path, "--basename", basename, "--hpfile", hpfile_path])

if __name__ == "__main__":
  main()

