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

"""Tests for coordinates utils."""

from absl.testing import absltest
from absl.testing import parameterized
from geeflow import coords
import numpy as np


class CoordsTest(parameterized.TestCase):

  def test_utm_grid_mapping(self):
    utm_grid_mapping = coords.UtmGridMapping(
        utm_zone="10N",
        cell_size=0.5,
        width=10,
        height=10,
        utm_x_min=1000.5,
        utm_y_min=2000.0,
    )
    expected_crs_transform = (0.5, 0.0, 1000.5, 0.0, -0.5, 2005.0)
    crs_transform = utm_grid_mapping.crs
    self.assertEqual(crs_transform, expected_crs_transform)

  def test_utm_grid_mapping_rescale(self):
    grid = coords.UtmGridMapping("10N", 10, 640, 480, 1000, 2000)
    rescaled = grid.rescale(5)
    self.assertEqual(rescaled.cell_size, 5.0)
    self.assertEqual(rescaled.width, grid.width * 2)
    self.assertEqual(rescaled.height, grid.height * 2)
    self.assertEqual(rescaled.utm_x_min, rescaled.utm_x_min)
    self.assertEqual(rescaled.utm_y_min, rescaled.utm_y_min)

  def test_utm_grid_mapping_from_bbox(self):
    bbox = (222723.818, 130298.284, 227929.316, 139839.571)
    utm_grid_mapping = coords.UtmGridMapping.from_bbox("18N", 10.0, bbox)
    self.assertEqual(utm_grid_mapping.utm_zone, "18N")
    self.assertEqual(utm_grid_mapping.width, 520)
    self.assertEqual(utm_grid_mapping.height, 954)
    self.assertEqual(utm_grid_mapping.utm_x_min, 222720.0)
    self.assertEqual(utm_grid_mapping.utm_y_min, 130300.0)

  def test_utm_grid_mapping_from_latlon(self):
    utm_grid_mapping = coords.UtmGridMapping.from_latlon_center(
        1.17, -77.4916575059, 10.0, 1000
    )
    self.assertEqual(utm_grid_mapping.utm_zone, "18N")
    self.assertEqual(utm_grid_mapping.width, 1000)
    self.assertEqual(utm_grid_mapping.height, 1000)
    self.assertEqual(utm_grid_mapping.utm_x_min, 217710.0)
    self.assertEqual(utm_grid_mapping.utm_y_min, 124440.0)

  @parameterized.parameters(
      (1, -179.99),
      (1, 179.99),
  )
  def test_longitudal_wrapping(self, lat, lon):
    roi = coords.UtmGridMapping.from_latlon_center(lat, lon, 5000, 1, 1)
    new_lat, new_lon = roi.centroid_latlon
    print(roi.bbox_latlon)
    np.testing.assert_allclose((new_lat, new_lon), (lat, lon), 1e-2, 1e-2)

  def test_vectorization(self):
    lat = -4
    lon = 45
    roi = coords.UtmGridMapping.from_latlon_center(lat, lon, 1, 1, 1)
    new_lat, new_lon = coords.UtmGridMapping(
        roi.utm_zone, 1, 1, 1,
        np.array([roi.utm_x_min]),
        np.array([roi.utm_y_min])).centroid_latlon
    np.testing.assert_allclose((new_lat, new_lon), ([lat], [lon]), 1e-5, 1e-5)

  def test_utm_grid_mapping_from_bbox_fails_for_latlon(self):
    bbox = (-77.49, 1.177, -77.12, 1.26)
    with self.assertRaises(AssertionError):
      coords.UtmGridMapping.from_bbox("18N", 10.0, bbox)

  def test_utm_grid_mapping_bbox_latlon(self):
    bbox = (222723.818, 130298.284, 227929.316, 139839.571)
    utm_grid_mapping = coords.UtmGridMapping.from_bbox("18N", 10.0, bbox)
    latlon = utm_grid_mapping.bbox_latlon
    expected = (1.177741637956, -77.491578, 1.264015162621, -77.444959)
    np.testing.assert_allclose(latlon, expected)

  @parameterized.parameters(
      ("18N", "EPSG:32618"),
      ("10C", "EPSG:32710"),
      ("49S", "EPSG:32649"),
      ("9m", "EPSG:32709"),
  )
  def test_utm_to_epsg(self, utm_zone, expected_epsg):
    utm_grid = coords.UtmGridMapping(utm_zone, 1, 10, 10, 0, 0)
    self.assertEqual(expected_epsg, utm_grid.epsg)

  @parameterized.parameters(
      (
          {"name": "ny1", "lat": 40.7128, "lon": -74.0060, "resolution": 1.0},
          (583459.372324085, 1.0, 0, 4507850.998243321, 0, -1.0),
          "EPSG:32618",
      ),
      (
          {"name": "ny2", "lat": 40.7128, "lon": -74.0060, "resolution": 10.0},
          (583459.372324085, 10.0, 0, 4507850.998243321, 0, -10.0),
          "EPSG:32618",
      ),
      (
          {"name": "ny3", "lat": 40.7128, "lon": -74.0060, "resolution": 0.5},
          (583459.372324085, 0.5, 0, 4507850.998243321, 0, -0.5),
          "EPSG:32618",
      ),
      (
          {
              "name": "London",
              "lat": 51.5074,
              "lon": -0.1278,
              "resolution": 10.0,
          },
          (698816.2343119299, 10.0, 0, 5710663.758080996, 0, -10.0),
          "EPSG:32630",
      ),
      (
          {
              "name": "Tokyo",
              "lat": 35.6895,
              "lon": 139.6917,
              "resolution": 10.0,
          },
          (381122.23003942776, 10.0, 0, 3950798.9078813544, 0, -10.0),
          "EPSG:32654",
      ),
  )
  def test_get_geotransform_info(
      self, test_case, expected_geotransform, expected_epsg
  ):
    lat, lon = test_case["lat"], test_case["lon"]
    image_width, resolution = 1000.0, test_case["resolution"]
    geotransform_info = coords.get_geotransform_info(
        lat, lon, image_width, resolution
    )
    self.assertEqual(geotransform_info["geotransform"], expected_geotransform)
    self.assertEqual(geotransform_info["epsg"], expected_epsg)


if __name__ == "__main__":
  absltest.main()
