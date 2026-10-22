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

import os
import tempfile
from typing import Any
import unittest

from geeflow import export_rasters_utils
import geopandas as gpd
import numpy as np
try:
  from osgeo import gdal
except ImportError:
  gdal = None
import shapely.geometry as geom
import shapely.geometry.base as shapely_base

from absl.testing import absltest as googletest
from absl.testing import parameterized

requires_gdal = unittest.skipIf(gdal is None, "GDAL not installed")


class ExportRastersUtilsTest(parameterized.TestCase):

  @requires_gdal
  def test_vectorize_raster_success(self):
    # Create a sample binary raster
    raster = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    # Define a sample transform
    transform = (0, 1, 0, 4, 0, -1)
    # Vectorize the raster
    gdf = export_rasters_utils.vectorize_raster(
        raster, transform, epsg_code=4326
    )
    # Check if the returned object is a GeoDataFrame
    self.assertIsInstance(gdf, gpd.GeoDataFrame)
    # Check if the GeoDataFrame is not empty
    self.assertFalse(gdf.empty)
    # Check the type of the geometry column
    self.assertTrue(
        all(
            isinstance(geom, shapely_base.BaseGeometry) for geom in gdf.geometry
        )
    )
    # Create the expected GeoDataFrame
    expected_gdf = gpd.GeoDataFrame(
        geometry=[
            geom.MultiPolygon([
                geom.Polygon([(3, 1), (4, 1), (4, 0), (3, 0), (3, 1)]),
                geom.Polygon([
                    (1, 4),
                    (3, 4),
                    (3, 3),
                    (4, 3),
                    (4, 2),
                    (2, 2),
                    (2, 1),
                    (1, 1),
                    (1, 2),
                    (0, 2),
                    (0, 3),
                    (1, 3),
                    (1, 4),
                ]),
            ]),
        ],
        crs=4326,
    )
    # Assert that the unary_union of the geometries in the vectorized
    # GeoDataFrame is equal to the unary_union of the geometries in the expected
    # GeoDataFrame
    self.assertEqual(gdf.unary_union, expected_gdf.unary_union)

  @requires_gdal
  def test_vectorize_raster_invalid_dimensions(self):
    # Create a 3D raster (invalid)
    raster = np.zeros((2, 2, 2), dtype=np.uint8)
    transform = (0, 1, 0, 4, 0, -1)
    with self.assertRaisesRegex(
        ValueError, r"Raster must be 2D, got \(2, 2, 2\)"
    ):
      export_rasters_utils.vectorize_raster(raster, transform, epsg_code=4326)

    # Create a 1D raster (invalid)
    raster = np.zeros(4, dtype=np.uint8)
    with self.assertRaisesRegex(ValueError, r"Raster must be 2D, got \(4,\)"):
      export_rasters_utils.vectorize_raster(raster, transform, epsg_code=4326)

  @requires_gdal
  def test_vectorize_raster_non_binary(self):
    # Create a non-binary raster
    raster = np.array(
        [
            [0, 1, 2, 0],
            [1, 2, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    transform = (0, 1, 0, 4, 0, -1)
    with self.assertRaisesRegex(ValueError, "Raster must be binary."):
      export_rasters_utils.vectorize_raster(raster, transform, epsg_code=4326)

  @requires_gdal
  def test_vectorize_raster_empty_gdf(self):
    # Create a raster with all zeros
    raster = np.zeros((4, 4), dtype=np.uint8)
    transform = (0, 1, 0, 4, 0, -1)
    with self.assertRaisesRegex(
        ValueError, "Empty GeoDataFrame. Polygonization failed."
    ):
      export_rasters_utils.vectorize_raster(raster, transform, epsg_code=4326)

  def _get_numpy_data_default_kwargs(
      self,
      mask: np.ndarray,
      preds: np.ndarray | None = None,
  ) -> dict[str, Any]:
    grid_params = dict(
        lat=-10.0,
        lon=-55.0,
        lat_end=-9.9,
        lon_end=-54.9,
        grid_spacing_m=960.0,
        cell_size=30.0,
        plot_size_m=960.0,
        num_splits=1,
    )
    x_splits, y_splits = export_rasters_utils.get_info(**grid_params)
    utm_x = int(x_splits[0] + 480)
    utm_y = int(y_splits[0] + 480)
    if preds is None:
      preds = np.ones((32, 32, 1), dtype=np.float32)
    return dict(
        inference_data={(utm_x, utm_y, 0, 0): [preds]},
        x_split=0,
        y_split=0,
        border_mode="any",
        normalize=False,
        clip_min=None,
        clip_max=None,
        discretization_factor=None,
        add_argmax="",
        shift_argmax=0,
        output_type="float32",
        inference_masks={(utm_x, utm_y, 0, 0): [mask]},
        **grid_params,
    )

  @parameterized.named_parameters(
      dict(testcase_name="mask_2d", mask_shape=(32, 32)),
      dict(testcase_name="mask_3d_single_channel", mask_shape=(32, 32, 1)),
  )
  def test_get_numpy_data_accepts_single_channel_mask_with_multiband_preds(
      self, mask_shape: tuple[int, ...]
  ):
    kwargs = self._get_numpy_data_default_kwargs(
        preds=np.ones((32, 32, 9), dtype=np.float32),
        mask=np.ones(mask_shape, dtype=bool),
    )

    tile, tile_mask = export_rasters_utils.get_numpy_data(**kwargs)

    self.assertEqual(tile.shape[-1], 9)
    self.assertEqual(tile_mask.ndim, 2)
    self.assertGreater(tile_mask.sum(), 0)

  @parameterized.named_parameters(
      dict(testcase_name="mask_1d", mask_shape=(32,)),
      dict(testcase_name="mask_3d_multichannel", mask_shape=(32, 32, 64)),
      dict(testcase_name="mask_4d_spatiotemporal", mask_shape=(9, 32, 32, 64)),
  )
  def test_get_numpy_data_rejects_invalid_mask_shapes(
      self, mask_shape: tuple[int, ...]
  ):
    kwargs = self._get_numpy_data_default_kwargs(
        mask=np.ones(mask_shape, dtype=bool),
    )

    with self.assertRaisesRegex(ValueError, "Expected 2D spatial mask"):
      export_rasters_utils.get_numpy_data(**kwargs)

  @parameterized.named_parameters(
      dict(
          testcase_name="cog_uint8",
          write_cogs=True,
          dtype=np.uint8,
          expected_nodata=255.0,
          expected_overviews=6,
          expected_structure={
              "LAYOUT": "COG",
              "COMPRESSION": "ZSTD",
              "INTERLEAVE": "BAND",
          },
      ),
      dict(
          testcase_name="cog_float32",
          write_cogs=True,
          dtype=np.float32,
          expected_nodata=-1.0,
          expected_overviews=6,
          expected_structure={
              "LAYOUT": "COG",
              "COMPRESSION": "ZSTD",
              "INTERLEAVE": "BAND",
          },
      ),
      dict(
          testcase_name="gtiff_int16",
          write_cogs=False,
          dtype=np.int16,
          expected_nodata=-1.0,
          expected_overviews=0,
          expected_structure={"COMPRESSION": "DEFLATE"},
      ),
  )
  @requires_gdal
  def test_write_tif_writes_valid_raster_and_unlinks_vsimem(
      self,
      write_cogs: bool,
      dtype: type[np.generic],
      expected_nodata: float,
      expected_overviews: int,
      expected_structure: dict[str, str],
  ):
    gdal_warnings: list[str] = []
    gdal.PushErrorHandler(
        lambda err_class, err_num, msg: gdal_warnings.append(msg.strip())
    )
    self.addCleanup(gdal.PopErrorHandler)

    rng = np.random.default_rng(42)
    geotransform = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    mask = np.ones((64, 64), dtype=bool)
    mask[0:8, 0:8] = False
    arr = rng.uniform(1, 100, size=(64, 64, 2)).astype(dtype)
    out_path = os.path.join(self.create_tempdir().full_path, "out.tif")

    returned_path = export_rasters_utils.write_tif(
        arr,
        mask,
        geotransform=geotransform,
        epsg="EPSG:32632",
        write_cogs=write_cogs,
        output_filename=out_path,
    )

    self.assertEqual(returned_path, out_path)
    self.assertTrue(os.path.exists(out_path))
    self.assertEmpty(gdal_warnings)
    vsimem_files = gdal.ReadDir("/vsimem") or []
    self.assertEmpty([f for f in vsimem_files if f.endswith(".tif")])

    ds = gdal.Open(out_path)
    self.assertIsNotNone(ds)
    self.assertEqual(
        (ds.RasterXSize, ds.RasterYSize, ds.RasterCount), (64, 64, 2)
    )
    self.assertEqual(ds.GetGeoTransform(), geotransform)
    image_structure = ds.GetMetadata("IMAGE_STRUCTURE")
    self.assertEqual(image_structure | expected_structure, image_structure)

    for b in (0, 1):
      rb = ds.GetRasterBand(b + 1)
      self.assertEqual(rb.GetNoDataValue(), expected_nodata)
      self.assertEqual(rb.GetOverviewCount(), expected_overviews)
      band_data = rb.ReadAsArray()
      np.testing.assert_array_equal(band_data[mask], arr[..., b][mask])
      np.testing.assert_array_equal(band_data[~mask], expected_nodata)

  @requires_gdal
  def test_write_tif_none_output_filename_writes_to_tempdir(self):
    arr = np.ones((16, 16, 1), dtype=np.uint8)
    geotransform = (0.0, 1.0, 0.0, 16.0, 0.0, -1.0)

    path = export_rasters_utils.write_tif(
        arr,
        None,
        geotransform=geotransform,
        epsg="EPSG:32632",
        write_cogs=False,
        output_filename=None,
    )
    self.addCleanup(os.remove, path)

    self.assertTrue(path.startswith(tempfile.gettempdir()))
    self.assertTrue(os.path.exists(path))

  @requires_gdal
  def test_stream_vsimem_to_gfile_raises_runtime_error_when_vsi_file_missing(
      self,
  ):
    out_path = os.path.join(self.create_tempdir().full_path, "missing.tif")
    with self.assertRaisesRegex(
        RuntimeError, "Failed to open /vsimem/nonexistent.tif"
    ):
      export_rasters_utils._stream_vsimem_to_gfile(
          "/vsimem/nonexistent.tif", out_path
      )

  @requires_gdal
  def test_write_geotiff_for_split_writes_tile_and_returns_uri(self):
    arr = np.full((32, 32, 1), 7, dtype=np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    geotransform = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    split_data = export_rasters_utils.SplitData(
        utm_zone="32N",
        x_split=1,
        y_split=2,
        np_data=arr,
        np_data_mask=mask,
        geotransform=geotransform,
        epsg="EPSG:32632",
    )
    gcs_folder = self.create_tempdir().full_path

    returned_uri = export_rasters_utils.write_geotiff_for_split(
        split_data=split_data,
        gcs_folder=gcs_folder,
        ee_asset_id="projects/my-proj/assets/coll",
        raster_name="tile_1_2",
        write_cogs=True,
        resampling="AVERAGE",
    )

    expected_path = os.path.join(
        gcs_folder, "projects/my-proj/assets/coll", "tile_1_2.tif"
    )
    self.assertEqual(returned_uri, expected_path)
    self.assertTrue(os.path.exists(expected_path))
    ds = gdal.Open(expected_path)
    self.assertIsNotNone(ds)
    self.assertEqual(
        (ds.RasterYSize, ds.RasterXSize, ds.RasterCount), (32, 32, 1)
    )


if __name__ == "__main__":
  googletest.main()
