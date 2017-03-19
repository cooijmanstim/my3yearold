import contextlib
from collections import OrderedDict as ordict
import itertools as it, functools as ft
import util

# key can be
#   composition of attr lookups: "attr1.attr2.attr3"
#   disjunction of multiple space-separated keys: "attr1 attr2.attr3"

# NOTE: disjunctions grow quadratically
def composekey(*keys):
  keys = [key.split() for key in keys]
  composites = [[]]
  for alternatives in keys:
    composites = [com + [alt] for alt in alternatives for com in composites]
  return " ".join(".".join(key) for key in composites)

def uncomposekey(prefix, key):
  prefix, key = prefix.split("."), key.split(".")
  while prefix:
    a, b = prefix.pop(0), key.pop(0)
    assert a == b
  return ".".join(key)

def insubforest(subkey, key):
  """`subkey` has one of alternatives in `key` as prefix"""
  try:
    subalts = subalternatives(subkey, key)
  except KeyError:
    return False
  return not subalts

def insubtree(subkey, key):
  """`subkey` has `key` as prefix"""
  try:
    subalt = subalternative(subkey, key)
  except KeyError:
    return False
  return not subalt

def subalternative(key, alt):
  """Leftover of constraint `alt` after selecting `key`.

  Mainly used in Narrow, where a key may select a subtree of the non-narrowed Holster and needs
  further narrowing. For example, if `h = H(a=H(b=3, c=5))` then `h.Narrow("a.b").a` should select
  the narrowed subtree `HolsterSubtree(h, "a").Narrow("b")`. In this case, `subalternative("a",
  "a.b") == "b"`, as `b` is the yet unenforced part of the narrowing constraint `a.b` after
  selecting `a`.

  Also used by `insubtree`, which requires that there is no leftover, i.e. `key` is fully underneath
  `alt`.

  Returns:
    If `key` is a prefix of `alt`, returns `alt` with the prefix `key` removed. If `alt` is a prefix
    of `key`, returns the empty string (meaning no constraints left to enforce).
  Raises:
    KeyError if `key` and `alt` do not share a prefix (i.e. `key` violates `alt`).
  """
  assert " " not in key
  assert " " not in alt
  keyparts, altparts = key.split("."), alt.split(".")
  while keyparts and altparts:
    a, b = keyparts.pop(0), altparts.pop(0)
    if a != b:
      raise KeyError()
  return ".".join(altparts)

def subalternatives(key, alts):
  """Leftovers of constraints in `alts` after selecting `key`.

  This is like `subalternative`, but `alts` is a composite key that represents the union of multiple
  keys.

  Mainly used in Narrow, where a key may select a subtree of the non-narrowed Holster and needs
  further narrowing. For example, if `h = H(a=H(b=3, c=5))` then `h.Narrow("a.b").a` should select
  the narrowed subtree `HolsterSubtree(h, "a").Narrow("b")`. In this case, `subalternative("a",
  "a.b") == "b"`, as `b` is the yet unenforced part of the narrowing constraint `a.b` after
  selecting `a`.

  Also used by `insubforest`, which requires that there is no leftover, i.e. `key` is fully
  underneath at least one of `alts`.

  Returns:
    A composite key disjoining the leftover constraints.
  Raises:
    KeyError if `key` violates all of `alts`.
  """
  assert " " not in key
  subalts = []
  for alt in alts.split(" "):
    try:
      subalt = subalternative(key, alt)
    except KeyError:
      continue
    subalts.append(subalt)
  if not subalts:
    raise KeyError()
  return " ".join(subalt for subalt in subalts if subalt)

def keyancestors(key, strict=False):
  alternatives = key.split(" ")
  for alternative in alternatives:
    parts = alternative.split(".")
    for i in range(1, len(parts) + (0 if strict else 1)):
      yield ".".join(parts[:i])

class BaseHolster(object):
  def __getattr__(self, key):
    if key[0].isupper():
      return self.__dict__[key]
    return self.Get(key)

  def __setattr__(self, key, value):
    if key[0].isupper():
      self.__dict__[key] = value
    else:
      self[key] = value

  def __getitem__(self, key):
    return self.Get(key)

  def __setitem__(self, key, value):
    self.Set(key, value)

  def __delitem__(self, key):
    self.Delete(key)

  # NOTE: Holster.Keys() and Holster.__contains__() are inconsistent in the sense that keys not
  # listed by Holster.Keys() may be reported as contained in the data structure. Keys() yields leaf
  # node keys, whereas __contains__() is true for internal node keys as well.
  def __contains__(self, key):
    try:
      _ = self.Get(key)
      return True
    except KeyError:
      return False

  def __iter__(self):
    return self.Keys()

  def __eq__(self, other):
    if not isinstance(other, BaseHolster):
      return False
    if self is other:
      return True
    for key in self.Keys():
      if key not in other or self.Get(key) != other.Get(key):
        return False
    for key in other.Keys():
      if key not in self or other.Get(key) != self.Get(key):
        return False
    return True

  def __ne__(self, other):
    return not self == other

  def __bool__(self):
    return any(self.Keys())

  __nonzero__ = __bool__

  def Keys(self):
    raise NotImplementedError()

  def Get(self, key):
    raise NotImplementedError()

  def Set(self, key, value):
    raise NotImplementedError()

  def Delete(self, key):
    raise NotImplementedError()

  def Values(self):
    for key in self.Keys():
      yield self.Get(key)

  def Items(self):
    for key in self.Keys():
      yield (key, self.Get(key))

  def Update(self, other):
    if isinstance(other, BaseHolster):
      for key, value in other.Items():
        self[key] = value
    elif hasattr(other, "keys"): # dict
      for key, value in other.items():
        self[key] = value
    else: # sequence of pairs
      for key, value in other:
        self[key] = value

  def Narrow(self, key):
    return HolsterNarrow(self, key)
  Y = Narrow

  def FlatCall(self, fn, *args, **kwargs):
    return Holster(util.equizip(self.Keys(), fn(list(self.Values()), *args, **kwargs)))

  def Zip(self, other):
    # NOTE: order determined by self
    return ((self.Get(key), other.Get(key)) for key in self.Keys())

  def AsDict(self):
    return ordict(self.Items())

  def With(self, items, **kwargs):
    h = H(self)
    h.Update(items)
    h.Update(kwargs)
    return h

  @contextlib.contextmanager
  def Bind(self, items=(), **kwargs):
    old = H()
    for key, value in it.chain(items, kwargs.items()):
      if key in self:
        old.Set(key, self[key])
      self.Set(key, value)
    yield self
    for key, value in it.chain(items, kwargs.items()):
      if key in old:
        self.Set(key, old.Get(key))
      else:
        self.Delete(key)

class Holster(BaseHolster):
  """Everything within reach."""

  def __init__(self, items=(), **kwargs):
    self.Data = ordict()
    self.Update(items)
    self.Update(kwargs)

  def Keys(self):
    return self.Data.keys()

  def Get(self, key):
    try:
      return self.Data[key]
    except KeyError:
      self.CheckAncestors(key)
      return HolsterSubtree(self, key)

  def Set(self, key, value):
    self.CheckAncestors(key)
    try:
      del self[key] # to ensure descendants are gone
    except KeyError:
      pass
    if isinstance(value, BaseHolster):
      for k, v in value.Items():
        self.Set(composekey(key, k), v)
    else:
      for alt in key.split(" "):
        self.Data[alt] = value

  def Delete(self, key):
    for alt in key.split(" "):
      subtree = self.Get(alt)
      if isinstance(subtree, BaseHolster):
        for subkey in list(subtree.Keys()):
          # recursive, but in the inner call subtree will be a leaf
          self.Delete(subkey)
      del self.Data[key]

  def __repr__(self):
    return "Holster([%s])" % ", ".join("(%r, %r)" % (key, value)
                                       for key, value in self.Items())

  def __str__(self):
    return "H{%s}" % ", ".join("%s: %s" % (key, value)
                               for key, value in self.Items())

  def CheckAncestors(self, key):
    ancestor = util.argany(a for a in keyancestors(key, strict=True)
                           if a in self.Data)
    if ancestor:
      raise KeyError("cannot descend into leaf node %s at %s"
                     % (repr(self.Data[ancestor])[:50], ancestor),
                     key)

class HolsterSubtree(BaseHolster):
  def __init__(self, other, key):
    assert " " not in key
    self.Other = other
    self.Key = key
    if not self:
      raise KeyError("nonexistent subtree", self.Key)

  def Keys(self):
    for key in self.Other.Keys():
      if insubtree(key, self.Key):
        yield uncomposekey(self.Key, key)

  def Get(self, key):
    return self.Other.Get(composekey(self.Key, key))

  def Set(self, key, value):
    self.Other[composekey(self.Key, key)] = value

  def Delete(self, key):
    self.Other[composekey(self.Key, key)]

  def __repr__(self):
    return "HolsterSubtree(%r, %r)" % (self.Other, self.Key)

  def __str__(self):
    return "HS{%s}" % ", ".join("%s: %s" % (key, value)
                                for key, value in self.Items())

class HolsterNarrow(BaseHolster):
  def __init__(self, other, key):
    self.Other = other
    self.Key = key
    self.RequireAllKeysExist()

  def RequireAllKeysExist(self):
    for alt in self.Key.split(" "):
      if alt not in self.Other:
        raise KeyError("narrowing to nonexistent key", alt)

  def Keys(self):
    for key in self.Other.Keys():
      if insubforest(key, self.Key):
        yield key

  def Get(self, key):
    assert " " not in key
    # two cases:
    # (1) key is a (nonstrict) child of one of the self.Key alternatives.
    #     In this case everything below key will match that self.Key alternative, and hence we
    #     can just return self.Other.get(key) without any further narrowing constraints.
    # (2) key selects a supertree of one of the self.Key alternatives.
    #     In this case things below key may not match the self.Key alternative, and we need to
    #     narrow the subtree returned from self.Other.get(key).
    try:
      subalts = subalternatives(key, self.Key)
    except KeyError:
      subalts = None
    if subalts is None:
      raise KeyError("key excluded by Narrow expression %s" % self.Key, key)
    result = self.Other.Get(key)
    if subalts and isinstance(result, HolsterSubtree):
      result = result.Narrow(subalts)
    return result

  def Set(self, key, value):
    raise NotImplementedError() # not sure what to do until I need it
    assert " " not in key
    try:
      subalts = subalternatives(key, self.Key)
    except KeyError:
      subalts = None
    # for write operations, key must be a (nonstrict) child of one of the self.Key alternatives.
    if subalts or subalts is None:
      raise KeyError("cannot write outside Narrow expression %s" % self.Key, key)
    self.Other.Set(key, value)

  def Delete(self, key):
    raise NotImplementedError() # not sure what to do until I need it
    assert " " not in key
    try:
      subalts = subalternatives(key, self.Key)
    except KeyError:
      subalts = None
    # for write operations, key must be a (nonstrict) child of one of the self.Key alternatives.
    if subalts or subalts is None:
      raise KeyError("cannot write outside Narrow expression %s" % self.Key, key)
    self.Other.Delete(key)

  def __repr__(self):
    return "HolsterNarrow(%r, %r)" % (self.Other, self.Key)

  def __str__(self):
    return "HN{%s}" % ", ".join("%s: %s" % (key, value)
                                for key, value in self.Items())

  def Narrow(self, key):
    return HolsterNarrow(self.Other, key)

H = Holster

if __name__ == "__main__":
  import unittest

  class HolsterTest(unittest.TestCase):
    def test(self):
      h = H([("c.d", H(e=3)), ("c.f.g c.f.h", 4), ("c.f.i", [5])], a=1, b=2)
      self.assertEqual(h.a, 1)
      self.assertEqual(h.b, 2)
      self.assertEqual(h.c.d.e, 3)
      self.assertEqual(h.c.f.g, 4)
      self.assertEqual(h.c.f.h, 4)
      self.assertEqual(h.c.f.i, [5])
      g = h.Narrow("c.d.e c.f.i")
      with self.assertRaises(KeyError): g.a
      with self.assertRaises(KeyError): g.c.f.g
      self.assertEqual(g.c.d.e, 3)
      self.assertEqual(g.c.f.i, [5])
      g = h.c.f
      self.assertEqual(dict(g.Items()),
                       dict(g=4, h=4, i=[5]))
      self.assertEqual(set(h.Keys()), set("c.d.e c.f.g c.f.h c.f.i a b".split()))
      self.assertEqual(h.FlatCall(lambda x: x), h)
      self.assertEqual(h.c.FlatCall(lambda x: x), h.c)

    def test_subalternatives(self):
      self.assertEqual(subalternatives("a", "a.b a.c.e a.c.d d.a"), "b c.e c.d")
      with self.assertRaises(KeyError): subalternatives("a.e", "a.b a.c.e a.c.d d.a")
      self.assertEqual(subalternatives("a.b", "a.b a.c.e a.c.d d.a"), "")

    def test_regression1(self):
      h = H()
      h["graph.train"] = 0
      h.graph.valid = 1
      self.assertEqual(set(h.graph.Keys()), set("train valid".split()))
      self.assertEqual(set(h.graph.Narrow("train valid").Keys()), set("train valid".split()))

  unittest.main()
