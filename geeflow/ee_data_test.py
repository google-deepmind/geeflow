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

"""Tests for EE datasets."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from geeflow import ee_data
import numpy as np


class EeDataTest(parameterized.TestCase):

  @mock.patch.object(ee_data, "ee")
  def test_qiu_disturbance(self, mock_ee):
    qiu = ee_data.QiuDisturbance()
    self.assertEqual(
        qiu.asset_name, "users/ShiQiu/product/conus/disturbance/v081/APRI"
    )
    _ = qiu.im  # Call property to trigger the function execution.
    self.assertGreater(mock_ee.ImageCollection.call_count, 0)
    self.assertGreater(mock_ee.Image.call_count, 0)
    self.assertGreater(mock_ee.Reducer.sum.call_count, 0)

  @parameterized.parameters(("L2A", "COPERNICUS/S2_SR_HARMONIZED"),
                            ("L1C", "COPERNICUS/S2_HARMONIZED"),)
  @mock.patch.object(ee_data, "ee")
  def test_s2_asset(self, mode, asset_name, mock_ee):
    s2 = ee_data.Sentinel2(mode)
    self.assertEqual(s2.asset_name, asset_name)
    _ = s2.ic
    self.assertEqual(mock_ee.ImageCollection.call_count, 1)

  def test_s2_stack(self):
    d = {"B1": np.zeros((2, 2)), "B2": np.zeros((2, 2)), "B3": np.zeros((2, 2)),
         "B4": np.zeros((2, 2)), "B5": np.zeros((2, 2))}
    arr = ee_data.Sentinel2.stack(d)
    self.assertEqual(arr.shape, (2, 2, 5))

  def test_s2_stack_vis(self):
    d = {"B1": np.zeros((2, 2)), "B2": np.zeros((2, 2)), "B3": np.zeros((2, 2)),
         "B4": np.zeros((2, 2)), "B5": np.zeros((2, 2))}
    arr = ee_data.Sentinel2.stack(d, vis=True)
    self.assertEqual(arr.shape, (2, 2, 3))

  def test_s2_vis_and_vis_norm(self):
    d = {"B1": np.zeros((2, 2)), "B2": np.full((2, 2), 1000),
         "B3": np.full((2, 2), 20000), "B4": np.zeros((2, 2))}
    rgb = ee_data.Sentinel2.vis(d)
    self.assertLessEqual(rgb.max(), 1.)
    self.assertGreaterEqual(rgb.min(), 0.)


if __name__ == "__main__":
  absltest.main()
