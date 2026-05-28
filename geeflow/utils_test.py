# Copyright 2026 DeepMind Technologies Limited.
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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from geeflow import ee_algo
from geeflow import utils
import geopandas as gpd
import pandas as pd
import shapely.geometry

import ee


class CommonTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_parse_arg_works(self, lazy):
    spec = dict(
        res=224,
        lr=0.1,
        runlocal=False,
        schedule="short",
    )

    def check(result, runlocal, schedule, res, lr):
      self.assertEqual(result.runlocal, runlocal)
      self.assertEqual(result.schedule, schedule)
      self.assertEqual(result.res, res)
      self.assertEqual(result.lr, lr)
      self.assertIsInstance(result.runlocal, bool)
      self.assertIsInstance(result.schedule, str)
      self.assertIsInstance(result.res, int)
      self.assertIsInstance(result.lr, float)

    check(utils.parse_arg(None, lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("", lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("runlocal=True", lazy=lazy, **spec),
          True, "short", 224, 0.1)
    check(utils.parse_arg("runlocal=False", lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("runlocal=", lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("runlocal", lazy=lazy, **spec),
          True, "short", 224, 0.1)
    check(utils.parse_arg("res=128", lazy=lazy, **spec),
          False, "short", 128, 0.1)
    check(utils.parse_arg("128", lazy=lazy, **spec),
          False, "short", 128, 0.1)
    check(utils.parse_arg("schedule=long", lazy=lazy, **spec),
          False, "long", 224, 0.1)
    check(utils.parse_arg("runlocal,schedule=long,res=128",
                          lazy=lazy, **spec),
          True, "long", 128, 0.1)

  @parameterized.parameters(
      (None, {}, {}),
      (None, {"res": 224}, {"res": 224}),
      ("640", {"res": 224}, {"res": 640}),
      ("runlocal", {}, {"runlocal": True}),
      ("res=640,lr=0.1,runlocal=false,schedule=long", {},
       {"res": 640, "lr": 0.1, "runlocal": False, "schedule": "long"}),
      )
  def test_lazy_parse_arg_works(self, arg, spec, expected):
    self.assertEqual(dict(utils.parse_arg(arg, lazy=True, **spec)),
                     expected)

  def test_standardized_path(self):
    path = utils.standardized_path(
        "planted/x:0.0.1",
        split_name="test",
        postfix="100n",
    )
    expected = "planted/x/0.0.1/test_100n.json"
    self.assertEqual(expected, path)


class UtilsTest(absltest.TestCase):

  def test_fix_orientation_for_bigquery_polygon_cw(self):
    p = shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])  # CW
    self.assertFalse(p.exterior.is_ccw)
    oriented_p = utils.fix_orientation_for_bigquery(p)
    self.assertIsInstance(oriented_p, shapely.geometry.Polygon)
    self.assertTrue(oriented_p.exterior.is_ccw)

  def test_fix_orientation_for_bigquery_polygon_ccw(self):
    p = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # CCW
    self.assertTrue(p.exterior.is_ccw)
    oriented_p = utils.fix_orientation_for_bigquery(p)
    self.assertIsInstance(oriented_p, shapely.geometry.Polygon)
    self.assertTrue(oriented_p.exterior.is_ccw)

  def test_fix_orientation_for_bigquery_multipolygon(self):
    p1 = shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])  # CW
    p2 = shapely.geometry.Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])  # CCW
    self.assertFalse(p1.exterior.is_ccw)
    self.assertTrue(p2.exterior.is_ccw)
    mp = shapely.geometry.MultiPolygon([p1, p2])
    oriented_mp = utils.fix_orientation_for_bigquery(mp)
    self.assertIsInstance(oriented_mp, shapely.geometry.MultiPolygon)
    self.assertLen(oriented_mp.geoms, 2)
    for p in oriented_mp.geoms:
      self.assertTrue(p.exterior.is_ccw)

  def test_fix_orientation_for_bigquery_empty(self):
    p = shapely.geometry.Polygon()
    oriented_p = utils.fix_orientation_for_bigquery(p)
    self.assertTrue(oriented_p.is_empty)

  @mock.patch.object(ee, "Image")
  def test_get_assets_geometries(self, mock_ee_image):
    mock_geometry = mock.MagicMock()
    mock_geometry.getInfo.return_value = {"type": "Polygon", "coordinates": []}
    mock_self_mask = mock.MagicMock()
    mock_self_mask.geometry.return_value = mock_geometry
    mock_ee_image.return_value.selfMask.return_value = mock_self_mask

    assets = ["asset1", "asset2"]
    geometries = utils.get_assets_geometries(assets, log=False)

    self.assertIsInstance(geometries, dict)
    self.assertLen(geometries, 2)
    self.assertIn("asset1", geometries)
    self.assertIn("asset2", geometries)
    self.assertIsInstance(geometries["asset1"], shapely.geometry.Polygon)

  @mock.patch.object(ee_algo, "fetch_image_collection_properties")
  @mock.patch.object(utils, "get_assets_geometries")
  def test_get_image_properties_no_geom(
      self, mock_get_assets_geometries, mock_fetch_props
  ):
    mock_fetch_props.return_value = [
        {"system:index": "1", "system:time_start": 1577836800000},
        {"system:index": "2", "system:time_start": 1577836800000},
    ]
    ic = mock.MagicMock()
    asset_name = "test_asset"
    properties = utils.get_image_properties(ic, asset_name, fetch_geom=False)

    self.assertIsInstance(properties, pd.DataFrame)
    self.assertNotIn("geometry", properties.columns)
    self.assertEqual(properties.shape[0], 2)
    mock_get_assets_geometries.assert_not_called()

  @mock.patch.object(ee_algo, "fetch_image_collection_properties")
  @mock.patch.object(utils, "get_assets_geometries")
  def test_get_image_properties_with_geom(
      self, mock_get_assets_geometries, mock_fetch_props
  ):
    mock_fetch_props.return_value = [
        {"system:index": "1", "system:time_start": 1577836800000},
        {"system:index": "2", "system:time_start": 1577836800000},
    ]
    mock_get_assets_geometries.return_value = {
        "test_asset/1": shapely.geometry.Polygon(),
        "test_asset/2": shapely.geometry.Polygon(),
    }
    ic = mock.MagicMock()
    asset_name = "test_asset"
    properties = utils.get_image_properties(
        ic, asset_name, fetch_geom=True, log=False
    )

    self.assertIsInstance(properties, gpd.GeoDataFrame)
    self.assertIn("geometry", properties.columns)
    self.assertEqual(properties.shape[0], 2)
    mock_get_assets_geometries.assert_called_once()

  @mock.patch.object(
      ee_algo, "fetch_image_collection_properties", return_value=[]
  )
  def test_get_image_properties_empty(self, mock_fetch_props):
    ic = mock.MagicMock()
    asset_name = "test_asset"
    properties = utils.get_image_properties(ic, asset_name)
    self.assertIsNone(properties)
    mock_fetch_props.assert_called_once()


if __name__ == "__main__":
  absltest.main()
