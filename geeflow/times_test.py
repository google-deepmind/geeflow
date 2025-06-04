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

"""Tests for dates/times routines."""

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from geeflow import times


class TimesTest(parameterized.TestCase):

  @parameterized.parameters(("2015-01-01",), (1420070400*1e3,),
                            (datetime.datetime(2015, 1, 1),),
                            (datetime.datetime(2015, 1, 1, tzinfo=times.UTC),))
  def test_date_list(self, start):
    seasonal = times.incremental_date_list(start=start, n=4, months=3)
    expected = [datetime.datetime(2015, 1, 1, tzinfo=times.UTC),
                datetime.datetime(2015, 4, 1, tzinfo=times.UTC),
                datetime.datetime(2015, 7, 1, tzinfo=times.UTC),
                datetime.datetime(2015, 10, 1, tzinfo=times.UTC)]
    self.assertEqual(expected, seasonal)

  @parameterized.parameters(
      (datetime.datetime(2015, 1, 1), False),
      (datetime.datetime(2015, 1, 1, tzinfo=times.UTC), True))
  def test_is_tza(self, d, expected):
    is_tza = times.is_tza(d)
    self.assertEqual(expected, is_tza)

  def test_make_tza(self):
    d = datetime.datetime(2015, 1, 1)
    d_tza = times.make_tza(d)
    expected = datetime.datetime(2015, 1, 1, tzinfo=times.UTC)
    self.assertEqual(expected, d_tza)

  @parameterized.parameters((datetime.datetime(2015, 1, 1),),
                            (1420070400*1e3,), ("2015-01-01",))
  def test_get_date_ranges(self, start):
    n = 2
    ranges = times.get_date_ranges(start, n, 3)
    self.assertEqual(ranges, [("2015-01-01", 3, 0), ("2015-04-01", 3, 0)])

  @parameterized.parameters((datetime.datetime(2015, 1, 1),),
                            (1420070400*1e3,), ("2015-01-01",))
  def test_get_date_ranges_days(self, start):
    n = 2
    ranges = times.get_date_ranges(start, n, days=5)
    self.assertEqual(ranges, [("2015-01-01", 0, 5), ("2015-01-06", 0, 5)])


if __name__ == "__main__":
  absltest.main()
