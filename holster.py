import sys
from collections import OrderedDict as ordict
import itertools as it, functools as ft
import util

# key can be
#   composition of attr lookups: "attr1.attr2.attr3"
#   disjunction of multiple space-separated keys: "attr1 attr2.attr3"
#   (maybe, later: with wildcard expressions: "*.attr{1,2,3}.attr4")

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
  assert " " not in subkey
  return any(insubtree(subkey, alt) for alt in key.split(" "))

def insubtree(subkey, key):
  assert " " not in key
  subparts, parts = subkey.split("."), key.split(".")
  if len(parts) > len(subparts):
    return False
  for subpart, part in zip(subparts, parts):
    if subpart != part:
      return False
  return True

# what do i really want
#def keymatch(subkey, key, subthing=any, thing=any):
#  return subthing(thing(issubkey(subalt, alt) for alt in key.split(" "))
#                  for subalt in subkey.split(" "))

def keyancestors(key, strict=False):
  alternatives = key.split(" ")
  for alternative in alternatives:
    parts = alternative.split(".")
    for i in range(1, len(parts) + (0 if strict else 1)):
      yield ".".join(parts[:i])

# TODO Glob, Regex methods to get a HolsterGlobView/HolsterRegexView on more flexible selections of keys
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

  def __contains__(self, key):
    value = self.Get(key)
    if isinstance(value, BaseHolster) and not value:
      return False
    return True

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

  def Keys(self):
    for key in self.Other.Keys():
      if insubforest(key, self.Key):
        yield key

  def Get(self, key):
    assert " " not in key
    if not insubforest(key, self.Key):
      raise KeyError(key)
    return self.Other.Get(key)

  def Set(self, key, value):
    assert " " not in key
    if not insubforest(key, self.Key):
      raise KeyError(key)
    self.Other.Set(key, value)

  def Delete(self, key):
    assert " " not in key
    if not insubforest(key, self.Key):
      raise KeyError(key)
    self.Other.Delete(key)

  def __repr__(self):
    return "HolsterNarrow(%r, %r)" % (self.Other, self.Key)

  def __str__(self):
    return "HN{%s}" % ", ".join("%s: %s" % (key, value)
                                for key, value in self.Items())

  def Narrow(self, key):
    return HolsterNarrow(self.Other, key)

H = Holster # bite me

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
      g = h[""]
      self.assertFalse(g)
      g = h["c.d.e c.f.i"]
      self.assertFalse(g.a)
      self.assertFalse(g.c.f.g)
      self.assertEqual(g.c.d.e, 3)
      self.assertEqual(g.c.f.i, [5])
      g = h.c.f
      self.assertEqual(dict(g.Items()),
                       dict(g=4, h=4, i=[5]))
      self.assertEqual(set(h.Keys()), set("c.d.e c.f.g c.f.h c.f.i a b".split()))
      self.assertEqual(h.FlatCall(lambda x: x), h)
      self.assertEqual(h.c.FlatCall(lambda x: x), h.c)

    def regression1(self):
      h = H()
      h.graph.train = 0
      h.graph.valid = 1
      self.assertEqual(set(h.graph.Keys()), set("train valid".split()))
      self.assertEqual(set(h.graph["train valid"].Keys()), set("train valid".split()))

  unittest.main()
