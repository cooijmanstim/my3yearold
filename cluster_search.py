import os, subprocess
import numpy as np

trainsh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cluster_train.sh")

class Distribution(object):
  pass

class Uniform(Distribution):
  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __call__(self):
    return np.random.random_integers(a, b)

class Boolean(Distribution):
  def __call__(self):
    return np.random.rand() < 0.5

class Categorical(Distribution):
  def __init__(self, categories):
    self.categories = categories

  def __call__(self):
    return np.random.choice(self.categories)

def main():
  distribution = H(**{
    "batch_size": Uniform(5, 100),
    "optimize_given": Boolean(),
    "convnet.kind": Categorical("straight straight_residual straight_dilated".split()),
    "convnet.depth": Uniform(64, 512),
    "convnet.profundity": Uniform(4, 48),
    "convnet.radius": Uniform(2, 7),
    "merger.kind": Categorical("attention conv".split()),
    "merger.delay": Uniform(0, 48),
    "merger.rate": Uniform(1, 24),
  })

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
    reader.bidir=1
    reader.radius=8
    reader.size=200
    reader.normalize=0
  """))

  def abbrev(key):
    return ".".join(parts[:1] for parts in key.split("."))

  # most will run out of memory and die early
  n = 50
  for i in range(n):
    hp = H(defaults)
    hp.Update((key, distribution())
              for key, distribution in distributions.items())
    basename = ",".join("=".join(abbrev(k), v) for k, v in hp.Items())
    env = dict(os.environ)
    env["MSCOCONET_HYPERPARAMETERS"] = util.serialize_hp(hp)
    subprocess.check_call(
      ["jobdispatch", "--gpu", "--mem=16G", '--raw="#PBS -l feature=k80"',
       trainsh_path, "--basename", basename],
      env=env)

if __name__ == "__main__":
  main()
