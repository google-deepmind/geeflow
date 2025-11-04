# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for computing stats over various data sources."""

import collections
import dataclasses
import json
import os
from typing import Any

from absl import logging
import dataclasses_json
from geeflow import utils
import jax
import ml_collections
import numpy as np

from tensorflow.io import gfile

META_PATH = "/tmp/geeflow/stats"
BAND_PREFIX = "_band_"

MAX_NUM_BINS = 10_000_000


@dataclasses.dataclass
class StatsAccumulator(dataclasses_json.DataClassJsonMixin):
  """Accumulator for common stats.

  Supported cumulative stats: mean, std, var, min, max, total, size,
                              sample std, sample var.
  Example usage:
    sa = StatsAccumulator()
    sa += [1, 2, 3]
    sa += [4, 5, 6, 7]
    print(sa.mean, sa.std, sa.min, sa.max, sa.total, sa.size)
  """

  sum: float = 0.0
  sum2: float = 0.0
  n: int = 0
  min: float | None = None
  max: float | None = None

  def merge(self, other):
    self.sum += other.sum
    self.sum2 += other.sum2
    self.n += other.n
    if other.min is not None:
      self.min = other.min if self.min is None else min(self.min, other.min)
    if other.max is not None:
      self.max = other.max if self.max is None else max(self.max, other.max)

  def add(self, x):
    # TODO: Support masking out values on the fly during accumulation.
    # This would avoid recomputation after masking and having exact non-bins
    # statistics values.
    x = np.array(x)
    if not x.size:
      return
    x = x.flatten()
    self.sum += x.sum()
    self.sum2 += (x.astype(float)**2).sum()
    self.n += x.size
    self.min = x.min() if self.min is None else min(x.min(), self.min)
    self.max = x.max() if self.max is None else max(x.max(), self.max)

  def __add__(self, x):
    self.add(x)
    return self

  def clear(self):
    self.sum = 0.0
    self.sum2 = 0.0
    self.n = 0
    self.min = None
    self.max = None

  def is_valid(self):
    return bool(self.n)

  def recast(self):
    """Ensures casting to all pure python types."""
    self.sum = float(self.sum)
    self.sum2 = float(self.sum2)
    self.n = int(self.n)
    self.min = float(self.min) if self.min is not None else self.min
    self.max = float(self.max) if self.max is not None else self.max

  def as_dict(self, drop_support=False):
    del drop_support
    if not self.is_valid():
      return {}
    self.recast()
    d = self.to_dict()
    names = ["size"]
    if self.total:
      names = ["mean", "std", "var", "total"]
      if self.n >= 2:
        names += ["sample_std", "sample_var"]
    for name in names:
      d[name] = getattr(self, name)
    return d

  def save_json(self, path, split_name=None, postfix=None, drop_support=False):
    """Saves stats data as a dict json (dropping supportive args)."""
    path = utils.standardized_path(path, split_name, postfix, META_PATH)
    d = self.as_dict(drop_support)
    os.umask(0o022); gfile.makedirs(os.path.dirname(path))
    print(f"Saving data in {path}")
    logging.info("Saving data in %s", path)

    # np.int64 is not JSON serializable, so convert it to python int.
    d = jax.tree.map(
        lambda w: int(w) if isinstance(w, np.int64) else w, d)

    with gfile.GFile(path, "w") as f:
      json.dump(d, f, indent=4, sort_keys=True, separators=(",", ":"))

  @property
  def std(self):
    return np.sqrt(self.var)

  @property
  def var(self):
    assert self.n > 0, "Requires at least 1 element."
    return self.sum2 / self.n - (self.sum**2) / (self.n**2)

  @property
  def sample_std(self):
    return np.sqrt(self.sample_var)

  @property
  def sample_var(self):
    assert self.n > 1, "Sample stats require at least 2 elements."
    return (self.sum2 - (self.sum**2)/self.n) / (self.n - 1)

  @property
  def mean(self):
    return self.sum / self.n

  @property
  def total(self):
    return self.sum

  @property
  def size(self):
    return self.n


@dataclasses.dataclass
class CounterAccumulator(StatsAccumulator):
  """Accumulator for common stats using counters.

  If data is in form of strings or other non-numeric objects, specify
  `numeric=False` to avoid computing numerical statistics.
  If data is of float types, by default it will be binned into integer bins,
  while the default numerical statistics are based on the original values.

  Supported cumulative stats for numeric types:
    mean, std, var, min, max, total, size, sample std, sample var.
  CounterAccumulator spectific stats:
    mode, bins, hist, counters.

  Example usage:
    sa = CounterAccumulator()
    sa += [1, 2, 3]
    sa += [4, 5, 6, 7]
    print(sa.mean, sa.std, sa.min, sa.max, sa.total, sa.size)
  """
  c: collections.Counter[Any] | None = None  # pylint: disable=g-bare-generic
  n_masked: int = 0
  numeric: bool = True  # Set to False if not counter over numbers.
  to_int: bool = True  # If float dtype, convert to int for bin counts.

  def merge(self, other):
    self.n_masked += other.n_masked
    if other.c:
      if self.c:
        self.c.update(other.c)
      else:
        self.c = other.c
    super().merge(other)

  def add(self, x, mask=None):
    x = np.array(x).flatten()
    if not x.size:
      return
    if mask is not None:
      flat_masks = np.array(mask).flatten().astype(bool)
      masked_values = x[~flat_masks]
      x = x[flat_masks]
      self.n_masked += masked_values.size
    if self.numeric:
      super().add(x)
    else:
      self.n += x.size
    if self.c is None:
      self.c = collections.Counter()
    if self.to_int and x.dtype in (float, np.float32, np.float64):
      x = x.astype(int)
    self.c.update(x.tolist())

  def clear(self):
    self.c = None
    self.n_masked = 0
    super().clear()

  def is_valid(self):
    # It can become invalid after masking out all elements.
    return self.c and bool(sum(self.c.values()))

  def as_dict(self, drop_support=False):
    if not self.is_valid():
      return {}
    d = super().as_dict()
    names = ["mode"]
    if self.numeric:
      names += ["bins_mean", "bins_median", "bins_std", "bins_iqr",
                "bins_iqr_std", "bins_mad", "bins_mad_std"]
    support_names = ["c", "numeric", "to_int"]
    for name in names:
      d[name] = getattr(self, name)
    for name in support_names:
      if drop_support:
        del d[name]
      else:
        d[f"~{name}"] = d.pop(name)
    return d

  @property
  def counters(self) -> dict[Any, int]:
    assert self.c
    return dict(self.c.most_common())

  @property
  def mode(self):
    """Returns the most frequent element (can be ambiguous if many)."""
    assert self.c
    return self.c.most_common(1)[0][0]

  def bins(self, min=None, max=None) -> list[int] | None:  # pylint: disable=redefined-builtin
    min = min or int(self.min)
    max = max or int(self.max)
    if max - min > MAX_NUM_BINS:
      return None
    out = [self.c[j] for j in range(min, max+1)]
    if not out[0] or not out[-1]:  # min and/or max got masked out.
      idx = np.where(out)[0]
      out = out[idx.min(): idx.max()+1]
    return out

  def hist(self, min=None, max=None) -> tuple[list[int], list[int]] | None:  # pylint: disable=redefined-builtin
    """Returns bins positions x and bin counts b."""
    # Note, np.histogram() returns the reversed tuple (b, x), and x has one
    # more element (len(x)==len(b)+1), denoting bin ranges.
    # Potentially we could adapt to it as well. For large quantities of numbers
    # and bins, this should be insignificant.
    min = min or int(self.min)
    max = max or int(self.max)
    if max - min > MAX_NUM_BINS:
      return None
    out = [self.c[j] for j in range(min, max+1)]
    if not out[0] or not out[-1]:  # min and/or max got masked out.
      idx = np.where(out)[0]
      out = out[idx.min(): idx.max()+1]
      # Add initial min value of the first bin.
      min, max = min + idx.min(), min + idx.max()
    return list(range(min, max+1)), out

  @property
  def bins_mean(self):
    """Returns mean computed from current bins data."""
    n, total = 0, 0
    for bin_i, bin_count in self.c.items():
      n += bin_count
      total += bin_i * bin_count
    return total / n

  @property
  def bins_std(self):
    """Returns std computed from current bins data."""
    n, total, sum2 = 0, 0, 0
    for bin_i, bin_count in self.c.items():
      n += bin_count
      total += bin_i * bin_count
      sum2 += (bin_i**2) * bin_count
    return np.sqrt(sum2 / n - (total**2) / (n**2))

  @property
  def bins_median(self):
    """Returns the median computed from current bins data."""
    hist = self.hist()
    if not hist: return None
    x, b = hist
    return hist_quantile(x, b, 0.5)

  @property
  def bins_mad(self):
    """Returns Median Absolute Deviation (MAD) for current bins data."""
    # Robust statistic of variability.
    hist = self.hist()
    if not hist: return None
    x, b = hist
    median = hist_quantile(x, b, 0.5)
    x = np.abs(np.array(x) - median)
    x, b = zip(*sorted(zip(x, b)))
    return hist_quantile(x, b, 0.5)

  @property
  def bins_mad_std(self):
    """Relates MAD to standard deviation for a normal distribution."""
    # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
    return None if self.bins_mad is None else self.bins_mad * 1.4826

  @property
  def bins_iqr(self):
    """Returns the interquartile range (IQR) from current bins data."""
    # Robust statistic of variability.
    hist = self.hist()
    if not hist: return None
    x, b = hist
    p25, p75 = hist_quantile(x, b, [0.25, 0.75])  # pylint: disable=unbalanced-tuple-unpacking
    return p75 - p25

  @property
  def bins_iqr_std(self):
    """Relates IQR to standard deviation for a normal distribution."""
    # https://en.wikipedia.org/wiki/Interquartile_range#Distributions
    return None if self.bins_iqr is None else self.bins_iqr / 1.349

  def mask(self, values, recompute=False):
    """Moves counter for mask value to a separate counter."""
    if not isinstance(values, (tuple, list)):
      values = [values]
    for value in values:
      if self.c and self.c[value]:
        self.n_masked += self.c[value]  # Keep track of all masked values.
        self.c[value] = 0
    if recompute and self.n_masked:
      self.recompute()

  def recompute(self):
    """Recomputes all bins-based stats after masking selected values."""
    # Would potentially as well change the results without masking, as the
    # original stats were computed on non-binned data.
    self.sum, self.sum2, self.n = 0, 0, 0
    self.min, self.max = None, None
    assert self.c
    for bin_i, bin_count in self.c.items():
      if bin_count:
        self.n += bin_count
        self.sum += bin_i * bin_count
        self.sum2 += (bin_i**2) * bin_count
        self.min = bin_i if self.min is None else min(bin_i, self.min)
        self.max = bin_i if self.max is None else max(bin_i, self.max)

  def print(self, n=None):
    if not self.c:
      print("No data.")
      return
    for i, (k, v) in enumerate(self.c.most_common(n)):
      print(f"{i:3}: {v:12,} - {k}")


class BandsAccumulator():
  """Class for separate accumulators for different data bands."""

  def __init__(self, n_bands=None, accumulator_class=CounterAccumulator,
               **kwargs):
    self._accumulator_cls = accumulator_class
    self._kwargs = kwargs
    self.accs = None
    if n_bands is not None:
      self._setup(n_bands)

  def _setup(self, n_bands):
    self.n_bands = n_bands
    self.accs = [self._accumulator_cls(**self._kwargs) for _ in range(n_bands)]

  def add(self, arr, mask=None):
    arr = np.array(arr)
    if not arr.size:
      return
    if self.accs is None:
      self._setup(arr.shape[-1])
    if arr.shape[-1] != self.n_bands:
      raise ValueError(f"Band dim of {arr.shape} does not match {self.n_bands}")
    for i, accumulator in enumerate(self.accs):
      accumulator.add(arr[..., i], mask[..., i] if mask is not None else None)

  def __add__(self, arr):
    self.add(arr)
    return self

  def mask(self, values, recompute=False):
    for accumulator in self.accs:
      accumulator.mask(values, recompute=recompute)

  def recompute(self):
    for accumulator in self.accs:
      accumulator.recompute()

  def as_dict(self, drop_support=False):
    d = {}
    for i, acc in enumerate(self.accs):
      d[i] = acc.as_dict(drop_support=drop_support)
      if not d[i]:
        print(f"\nWARNING: no data for band {i}. Dropping it.")
        logging.warning("No data for band %s. Dropping it.", i)
        del d[i]
    return d

  def save_json(self, path, split_name=None, postfix=None, drop_support=False):
    """Saves stats data as a dict json (dropping supportive args)."""
    path = utils.standardized_path(path, split_name, postfix, META_PATH)
    d = self.as_dict(drop_support)
    os.umask(0o022); gfile.makedirs(os.path.dirname(path))
    print(f"Saving data in {path}")
    logging.info("Saving data in %s", path)
    with gfile.GFile(path, "w") as f:
      json.dump(d, f, indent=4, sort_keys=True, separators=(",", ":"))


def load_json(path, split_name=None, postfix=None, as_cd=False,
              drop_support=True):
  """Returns dict of accumulated stats."""
  full_path = utils.standardized_path(path, split_name, postfix, META_PATH)
  if gfile.exists(full_path):
    with gfile.GFile(full_path, "r") as f:
      d = json.load(f)
  else:
    full_path = utils.standardized_path(
        path, split_name, postfix + BAND_PREFIX + "*", META_PATH
    )
    d = {}
    for filename in gfile.Glob(full_path):
      band_position = filename.rindex(BAND_PREFIX) + len(BAND_PREFIX)
      band_id = filename[band_position: -5]  # -5 is for ".json"
      with gfile.GFile(filename, "r") as f:
        d[band_id] = json.load(f)

  for k in list(d):
    if isinstance(d[k], dict):  # For bands accumulators.
      for kk in list(d[k]):
        if kk.startswith("~"):
          if drop_support:
            del d[k][kk]
          else:
            d[k][kk[1:]] = d[k].pop(kk)
    if k.startswith("~"):  # rename support variables to original names.
      if drop_support:
        del d[k]
      else:
        d[k[1:]] = d.pop(k)
    if k.isnumeric():
      d[int(k)] = d.pop(k)  # Keep band keys as numeric for now.
  if as_cd:
    return ml_collections.ConfigDict(d)
  return d


def hist_quantile(x, b, quantiles):
  """Returns quantile values for bins b at positions x."""
  # Similar to np.quantile (but simpler), but works on histogram data.
  # Not extensively tested and might be non-ideal in small data regimes.
  is_scalar = isinstance(quantiles, float)
  if is_scalar:
    quantiles = [quantiles]
  cumulative_frequency = np.cumsum(b)
  out = []
  for q in quantiles:
    q_frequency = (cumulative_frequency[-1] + 1) * q
    ind = np.argmin(np.abs(cumulative_frequency-q_frequency))
    out.append(((x[ind] + x[ind+1]) / 2) if ind < len(x)-1 else x[ind])
  return out[0] if is_scalar else out
