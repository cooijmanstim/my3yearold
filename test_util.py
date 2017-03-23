import unittest
import numpy as np, scipy.stats
import util

class SampleTest(unittest.TestCase):
  def test_choice(self):
    # chi-squared test using numpy's multinomial sampler for comparison
    return
    p = np.exp(np.random.randn(10))
    p = p / p.sum()
    n = 10000
    q = 0
    for _ in range(n):
      q = q + np.random.multinomial(1, p)
    chisq, pval = scipy.stats.chisquare(q, p * n)
    np.testing.assert_array_less(0.05, pval)

  def test_sample1(self):
    p = np.exp(np.random.randn(10))
    n = 10000
    q = 0
    for _ in range(n):
      q = q + util.sample(p, onehot=True)
    p = p / p.sum()
    chisq, pval = scipy.stats.chisquare(q, p * n)
    np.testing.assert_array_less(0.05, pval)

  def test_sample2(self):
    axis = 1
    p = np.exp(np.random.randn(2, 3))
    n = 10000
    q = 0
    for _ in range(n):
      q = q + util.sample(p, axis=axis, onehot=True)
    p = p / p.sum(axis=axis, keepdims=True)
    chisq, pval = scipy.stats.chisquare(q, p * n, axis=axis)
    np.testing.assert_array_less(0.05, pval)

if __name__ == "__main__":
  unittest.main()
