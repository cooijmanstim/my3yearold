import numpy as np

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
