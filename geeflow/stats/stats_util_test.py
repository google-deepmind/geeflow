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

from absl.testing import absltest
from absl.testing import parameterized
from geeflow.stats import stats_util
import numpy as np


class StatsUtilTest(parameterized.TestCase):

  def test_stats_accumulator_per_scalar(self):
    arr = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    sa = stats_util.StatsAccumulator()
    for x in arr:
      sa += x
    self.assertEqual(sa.mean, arr.mean())
    self.assertEqual(sa.std, arr.std())
    self.assertEqual(sa.var, arr.var())
    self.assertEqual(sa.min, arr.min())
    self.assertEqual(sa.max, arr.max())
    self.assertEqual(sa.total, arr.sum())
    self.assertEqual(sa.size, arr.size)
    self.assertEqual(sa.sample_std, arr.std(ddof=1))
    self.assertEqual(sa.sample_var, arr.var(ddof=1))

  def test_counter_accumulator_per_scalar(self):
    arr = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    sa = stats_util.CounterAccumulator()
    for x in arr:
      sa += x
    self.assertEqual(sa.mode, 4)  # most frequent.
    self.assertEqual(sa.bins(), [1, 0, 3, 2, 0, 1, 0, 1])
    self.assertEqual(sa.hist(), ([2, 3, 4, 5, 6, 7, 8, 9],
                                 [1, 0, 3, 2, 0, 1, 0, 1]))
    self.assertEqual(sa.mean, arr.mean())
    self.assertEqual(sa.std, arr.std())
    self.assertEqual(sa.var, arr.var())
    self.assertEqual(sa.min, arr.min())
    self.assertEqual(sa.max, arr.max())
    self.assertEqual(sa.total, arr.sum())
    self.assertEqual(sa.size, arr.size)
    self.assertEqual(sa.sample_std, arr.std(ddof=1))
    self.assertEqual(sa.sample_var, arr.var(ddof=1))

  def test_counter_accumulator_with_floats(self):
    arr = np.array([-1.9, 3.1, 3.7])
    sa = stats_util.CounterAccumulator()
    sa += arr
    self.assertEqual(sa.mode, 3)  # most frequent.
    self.assertEqual(sa.bins(), [1, 0, 0, 0, 2])
    self.assertEqual(sa.hist(), ([-1, 0, 1, 2, 3],
                                 [1, 0, 0, 0, 2]))
    self.assertEqual(sa.min, -1.9)
    self.assertEqual(sa.max, 3.7)
    self.assertEqual(sa.mean, arr.mean())
    self.assertEqual(sa.std, arr.std())
    self.assertEqual(sa.var, arr.var())
    self.assertEqual(sa.min, arr.min())
    self.assertEqual(sa.max, arr.max())

  def test_counter_accumulator_masked_bins(self):
    arr = np.array([2, 5, 4, 5, 99])
    sa = stats_util.CounterAccumulator()
    sa += arr
    sa.mask(5)  # Mask out values 5 and 99. Keep only 2 and 4.
    sa.mask(99)

    self.assertEqual(sa.mean, arr.mean())  # Over all original inputs.
    self.assertEqual(sa.bins_mean, 3)  # Over remaining [2, 4].
    self.assertEqual(sa.std, arr.std())  # Over all original inputs.
    self.assertEqual(sa.bins_std, 1)  # Over remaining [2, 4].
    self.assertEqual(sa.bins(), [1, 0, 1])
    self.assertEqual(sa.hist(), ([2, 3, 4], [1, 0, 1]))

  def test_counter_accumulator_direct_masking(self):
    arr = np.array([2, 5, 4, 5, 99])
    mask = np.array([1, 0, 1, 0, 0])
    sa = stats_util.CounterAccumulator()
    sa.add(arr, mask)

    arr_masked = arr[mask.astype(bool)]
    self.assertEqual(sa.n_masked, 3)
    self.assertEqual(sa.mean, arr_masked.mean())
    self.assertEqual(sa.bins_mean, 3)  # Over remaining [2, 4].
    self.assertEqual(sa.std, arr_masked.std())
    self.assertEqual(sa.bins_std, 1)  # Over remaining [2, 4].
    self.assertEqual(sa.min, 2)
    self.assertEqual(sa.max, 4)
    self.assertEqual(sa.bins(), [1, 0, 1])
    self.assertEqual(sa.hist(), ([2, 3, 4], [1, 0, 1]))

  def test_bands_accumulator(self):
    n_bands = 4
    arr = np.ones((3, n_bands))
    sa = stats_util.BandsAccumulator(n_bands)
    sa += arr
    self.assertLen(sa.accs, n_bands)
    self.assertEqual(sa.accs[0].mean, 1)

  def test_bands_accumulator_lazy_init(self):
    n_bands = 4
    arr = np.ones((3, n_bands))
    sa = stats_util.BandsAccumulator()
    sa += arr
    self.assertLen(sa.accs, n_bands)
    self.assertEqual(sa.accs[0].mean, 1)

  @parameterized.parameters((stats_util.StatsAccumulator,),
                            (stats_util.CounterAccumulator,),
                            (stats_util.BandsAccumulator,))
  def test_accumulator_with_empty_data(self, accumulator):
    sa = accumulator()
    sa.add([])  # Add empty data. Should not change the accumulator.
    if accumulator == stats_util.BandsAccumulator:
      self.assertIsNone(sa.accs)
    else:
      self.assertEqual(sa.n, 0)

  def test_counter_accumulator_masked_recompute(self):
    arr = np.array([2, 5, 4, 5, 99])
    sa = stats_util.CounterAccumulator()
    sa += arr
    sa.mask(5)  # Mask out values 5 and 99. Keep only 2 and 4.
    sa.mask(99)

    self.assertEqual(sa.mean, 23)  # Over all original inputs.
    self.assertEqual(sa.std, 38.015786194684964)  # Over all original inputs.
    self.assertEqual(sa.min, 2)  # After recomputation.
    self.assertEqual(sa.max, 99)  # After recomputation.
    self.assertEqual(sa.bins_mean, 3)  # Over remaining [2, 4].
    self.assertEqual(sa.bins_std, 1)  # Over remaining [2, 4].

    sa.recompute()
    self.assertEqual(sa.mean, 3)  # After recomputation.
    self.assertEqual(sa.std, 1)  # After recomputation.
    self.assertEqual(sa.min, 2)  # After recomputation.
    self.assertEqual(sa.max, 4)  # After recomputation.
    self.assertEqual(sa.bins(), [1, 0, 1])
    self.assertEqual(sa.hist(), ([2, 3, 4], [1, 0, 1]))

  def test_counter_accumulator_median_and_iqr_mad(self):
    arr = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    sa = stats_util.CounterAccumulator()
    sa += arr
    self.assertEqual(sa.bins(), [1, 0, 3, 2, 0, 1, 0, 1])
    self.assertEqual(sa.hist(), ([2, 3, 4, 5, 6, 7, 8, 9],
                                 [1, 0, 3, 2, 0, 1, 0, 1]))
    self.assertEqual(sa.mode, 4)
    self.assertEqual(sa.mean, 5)
    self.assertEqual(sa.bins_mean, 5)
    self.assertEqual(sa.bins_median, 4.5)
    self.assertEqual(sa.std, 2)
    self.assertEqual(sa.bins_std, 2)
    self.assertEqual(sa.bins_iqr, 5)
    self.assertEqual(sa.bins_mad, 1)
    self.assertAlmostEqual(sa.bins_iqr_std, 3.7064492216456637)
    self.assertAlmostEqual(sa.bins_mad_std, 1.4826)

  def test_hist_quantile(self):
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 1000])
    sa = stats_util.CounterAccumulator()
    sa += arr
    data = sa.hist()
    assert data
    x, b = data
    q25, median, q75 = stats_util.hist_quantile(x, b, (0.25, 0.5, 0.75))
    self.assertEqual(q25, 2.5)
    self.assertEqual(median, 4.5)
    self.assertEqual(q75, 7.5)

  def test_standardized_path(self):
    path = stats_util.standardized_path("planted/x:0.0.1", "test", "100n")
    expected = "/tmp/geeflow/stats/planted/x/0.0.1/test_100n.json"
    self.assertEqual(expected, path)


if __name__ == "__main__":
  absltest.main()
