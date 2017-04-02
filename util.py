import numpy as np
import os, contextlib

DEFAULT = object()

def batches(xs, batch_size=1, shuffle=False):
  xs = list(xs)
  if shuffle:
    np.random.shuffle(xs)
  for start in range(0, len(xs), batch_size):
    yield xs[start:start+batch_size]

def padto(x, length, axis=0):
  return np.pad(x, [(0, 0) if i is not axis else (0, length - dim)
                    for i, dim in enumerate(x.shape)],
                mode="constant")

# use with care; makes functions stateful and leaks memory
def memo(f):
  cache = dict()
  def g(*args, **kwargs):
    key = (args, frozenset(kwargs.items()))
    try:
      return cache[key]
    except KeyError:
      cache[key] = f(*args, **kwargs)
      return cache[key]
  return g

def equizip(*xs):
  xs = list(map(list, xs))
  assert all(len(x) == len(xs[0]) for x in xs)
  return zip(*xs)

def argany(xs):
  for x in xs:
    if x:
      return x

def deepsubclasses(klass):
  for subklass in klass.__subclasses__():
    yield subklass
    for subsubklass in deepsubclasses(subklass):
      yield subsubklass

class Factory(object):
  @classmethod
  def make(klass, key, *args, **kwargs):
    for subklass in deepsubclasses(klass):
      if subklass.key == key:
        return subklass(*args, **kwargs)
    else:
      raise KeyError("unknown %s subclass key %s" % (klass, key))

def softmax(p, axis=None, temperature=1):
  if axis is None:
    axis = p.ndim - 1
  if temperature == 0.:
    # NOTE: in case of multiple equal maxima, returns uniform distribution over them
    p = p == np.max(p, axis=axis, keepdims=True)
  else:
    oldp = p
    logp = np.log(p)
    logp /= temperature
    logp -= logp.max(axis=axis, keepdims=True)
    p = np.exp(logp)
  p /= p.sum(axis=axis, keepdims=True)
  if np.isnan(p).any():
    import pdb; pdb.set_trace()
  return p

def sample(p, axis=None, temperature=1, onehot=False):
  assert (p >= 0).all() # just making sure we don't put log probabilities in here

  if axis is None:
    axis = p.ndim - 1

  if temperature != 1:
    p = p ** (1. / temperature)
  cmf = p.cumsum(axis=axis)
  totalmasses = cmf[tuple(slice(None) if d != axis else slice(-1, None) for d in range(cmf.ndim))]
  u = np.random.random([p.shape[d] if d != axis else 1 for d in range(p.ndim)])
  i = np.argmax(u * totalmasses < cmf, axis=axis)

  return to_onehot(i, axis=axis, depth=p.shape[axis]) if onehot else i

def to_onehot(i, depth, axis=None):
  if axis is None:
    axis = i.ndim
  x = np.eye(depth)[i]
  if axis != i.ndim:
    # move new axis forward
    axes = list(range(i.ndim))
    axes.insert(axis, i.ndim)
    x = np.transpose(x, axes)
  assert np.allclose(x.sum(axis=axis), 1)
  return x

def parse_hp(s):
  d = dict()
  for a in s.split():
    key, value = a.split("=")
    try:
      value = int(value)
    except ValueError:
      try:
        value = float(value)
      except ValueError:
        pass
    d[key] = value
  return d

def serialize_hp(hp, outer_separator=" ", inner_separator="="):
  return outer_separator.join(sorted(["%s%s%s" % (k, inner_separator, v) for k, v in hp.Items()]))

def make_label(config):
  return "%s%s%s" % (config.basename,
                     "_" if config.basename else "",
                     serialize_hp(config.hp, outer_separator="_"))

# unobtrusive structured logging of arbitrary values
class Bamboo(object):
  def __init__(self):
    self.root = BambooScope("root", subsample_factor=1)
    self.stack = [self.root]

  @contextlib.contextmanager
  def scope(self, label, subsample_factor=None):
    new_scope = BambooScope(label, subsample_factor=subsample_factor)
    self.stack[-1].log(new_scope)
    self.stack.append(new_scope)
    yield
    self.stack.pop()

  def log(self, **kwargs):
    self.stack[-1].log(kwargs)

  def dump(self, path):
    dikt = {}
    def _compile_npz_dict(item, path):
      i, node = item
      if isinstance(node, BambooScope):
        for subitem in node.items:
          _compile_npz_dict(subitem, os.path.join(path, "%s_%s" % (i, node.label)))
      else:
        for k, v in node.items():
          dikt[os.path.join(path, "%s_%s" % (i, k))] = v
    _compile_npz_dict((0, self.root), "")
    np.savez_compressed(path, **dikt)

class BambooScope(object):
  def __init__(self, label, subsample_factor=None):
    self.label = label
    self.subsample_factor = 1 if subsample_factor is None else subsample_factor
    self.items = []
    self.i = 0

  def log(self, x):
    # append or overwrite such that we retain every `subsample_factor`th value and the last value
    item = (self.i, x)
    if (self.subsample_factor == 1 or
        self.i % self.subsample_factor == 1 or
        not self.items):
      self.items.append(item)
    else:
      self.items[-1] = item
    self.i += 1
