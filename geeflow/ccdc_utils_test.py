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
from geeflow import ccdc_utils
from geeflow import ee_tests_mocks  # pylint: disable=unused-import
from ml_collections import config_dict
import numpy as np

import ee


class CcdcUtilsTest(parameterized.TestCase):

  def test_im_add_ccdc_bands_2d(self):
    ic = ee.Image({"band1_coefs": [1, 2], "band2_coefs": [3, 4]})
    num_segments = 2
    im = ccdc_utils.im_add_ccdc_bands_2d(ic, num_segments)
    self.assertIsInstance(im, ee.Image)

  def test_im_add_ccdc_bands_1d(self):
    ic = ee.Image({"band1": [1, 2], "band2": [3, 4]})
    num_segments = 2
    im = ccdc_utils.im_add_ccdc_bands_1d(ic, num_segments)
    self.assertIsInstance(im, ee.Image)

  def test_get_ccdc_pixels(self):
    pixels = {
        "ccdc_BLUE#0#0": np.array([[1]]),
        "ccdc_BLUE#1#0": np.array([[2]]),
    }
    cfg = config_dict.ConfigDict({
        "select": ["BLUE"],
        "sampling_kw": {"num_segments": 2},
    })
    ccdc_pixels = ccdc_utils.get_ccdc_pixels(pixels, cfg, "ccdc")
    self.assertIn("ccdc_BLUE", ccdc_pixels)
    self.assertEqual(ccdc_pixels["ccdc_BLUE"].shape, (1, 1, 2, 1))

  def test_generate_ccdc_1year(self):
    data = {
        "ccdc_tStart": np.array([[[2000, 2001]]]).reshape((1, 1, 2, 1)),
        "ccdc_BLUE_mag": np.array([[[1, 2]]]).reshape((1, 1, 2, 1)),
        "ccdc_BLUE_coefs": np.arange(16).reshape((1, 1, 2, 8)),
    }
    cfg = config_dict.ConfigDict({
        "select": ["tStart", "BLUE_mag", "BLUE_coefs"],
        "sampling_kw": {"num_segments": 2},
        "format_config": {"from": 2000, "to": 2000, "selection": "middle"}
    })
    ccdc, ccdc_mask = ccdc_utils.generate_ccdc(data, cfg, "ccdc")
    self.assertEqual(ccdc.shape, (1, 1, 1, 10))
    self.assertEqual(ccdc_mask.shape, (1, 1, 1))

  def test_generate_ccdc_2years(self):
    data = {
        "ccdc_tStart": np.array([[[2000, 2001]]]).reshape((1, 1, 2, 1)),
        "ccdc_BLUE_mag": np.array([[[1, 2]]]).reshape((1, 1, 2, 1)),
        "ccdc_BLUE_coefs": np.arange(16).reshape((1, 1, 2, 8)),
    }
    cfg = config_dict.ConfigDict({
        "select": ["tStart", "BLUE_mag", "BLUE_coefs"],
        "sampling_kw": {"num_segments": 2},
        "format_config": {"from": 2000, "to": 2001, "selection": "middle"}
    })
    ccdc, ccdc_mask = ccdc_utils.generate_ccdc(data, cfg, "ccdc")
    self.assertEqual(ccdc.shape, (2, 1, 1, 10))
    self.assertEqual(ccdc_mask.shape, (2, 1, 1))

  def test_generate_ccdc_raw(self):
    data = {
        "ccdc_tStart": np.array([[[2000, 2001]]]).reshape((1, 1, 2, 1)),
        "ccdc_BLUE_mag": np.array([[[1, 2]]]).reshape((1, 1, 2, 1)),
        "ccdc_BLUE_coefs": np.arange(16).reshape((1, 1, 2, 8)),
    }
    cfg = config_dict.ConfigDict({
        "select": ["tStart", "BLUE_mag", "BLUE_coefs"],
        "sampling_kw": {"num_segments": 2},
    })
    ccdc, ccdc_mask = ccdc_utils.generate_ccdc(data, cfg, "ccdc")
    self.assertEqual(ccdc.shape, (2, 1, 1, 10))
    self.assertEqual(ccdc_mask.shape, (2, 1, 1))

  @parameterized.parameters(("longest", 1), ("middle", 0))
  def test_generate_ccdc_selection_methods(self, selection, expected):
    data = {
        "ccdc_tStart":
            np.array([[[1999.8, 2000.51, 2000.95]]]).reshape((1, 1, 3, 1)),
        "ccdc_tEnd":
            np.array([[[2000.2, 2000.91, 2021.9]]]).reshape((1, 1, 3, 1)),
        "ccdc_BLUE_mag": np.array([[[1, 2, 3]]]).reshape((1, 1, 3, 1)),
    }
    cfg = config_dict.ConfigDict({
        "select": ["tStart", "tEnd", "BLUE_mag"],
        "sampling_kw": {"num_segments": 3},
        "format_config": {"from": 2000, "to": 2000, "selection": selection}
    })
    ccdc, ccdc_mask = ccdc_utils.generate_ccdc(data, cfg, "ccdc")
    self.assertEqual(ccdc.shape, (1, 1, 1, 3))
    self.assertEqual(ccdc_mask.shape, (1, 1, 1))
    self.assertEqual(ccdc[0, 0, 0, 0], data["ccdc_tStart"][0, 0, expected])
    self.assertEqual(ccdc[0, 0, 0, 1], data["ccdc_tEnd"][0, 0, expected])
    self.assertEqual(ccdc[0, 0, 0, 2], data["ccdc_BLUE_mag"][0, 0, expected])
    self.assertTrue(ccdc_mask[0, 0, 0])


if __name__ == "__main__":
  absltest.main()
