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

"""Tests for ee_export_utils and export_beam_tfds_utils scalar batching."""

from unittest import mock

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_util
from geeflow import ee_export_utils
from geeflow import export_beam_tfds_utils
from geeflow import pipelines
from geeflow import scalar_fetch
import ml_collections
import numpy as np


def _make_mixed_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()
  config.labels = ml_collections.ConfigDict({
      "img_width_m": 20,
      "max_cell_size_m": 10,
      "use_utm": True,
      "scalar_batch_size": 2,
      "scalar_batch_cell_size_deg": 1.0,
  })
  config.sources = ml_collections.ConfigDict({
      "bio": {
          "module": "Bioclim",
          "scalar": True,
          "algo": "im_static",
          "select": ["bio01", "bio12"],
          "mask_value": -9999.0,
      },
      "patch": {
          "module": "Bioclim",
          "scalar": False,
          "algo": "im_static",
          "select": ["b0"],
          "scale": 10,
      },
  })
  return config


class EeExportUtilsScalarBatchingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(ee_export_utils.utils, "initialize_ee", autospec=True)
    )
    self.enter_context(mock.patch.object(scalar_fetch.ee.Geometry, "Point"))
    self.enter_context(mock.patch.object(scalar_fetch.ee, "Feature"))
    self.enter_context(mock.patch.object(scalar_fetch.ee, "FeatureCollection"))

  def test_get_info_only_fetches_unbatched_sources_and_preserves_scalars(self):
    config = _make_mixed_config()
    mock_patch_im = mock.MagicMock()

    with (
        mock.patch.object(
            ee_export_utils.pipelines,
            "pipeline_sources",
            return_value={"patch": mock_patch_im},
        ) as mock_pipeline_sources,
        mock.patch.object(
            ee_export_utils.pipelines,
            "pipeline_item_to_rois",
            return_value={"patch": mock.MagicMock()},
        ),
        mock.patch.object(
            ee_export_utils.pipelines,
            "get_requests_fn",
        ) as mock_get_requests_fn,
    ):
      req = pipelines.Request(
          image=mock_patch_im,
          roi=mock.MagicMock(),
          scale=10,
          crs="EPSG:32632",
          name="patch",
          fixed_band_names=True,
      )
      mock_get_requests_fn.return_value = lambda rois, it: ([req], dict(it))

      _, non_scalar_config = scalar_fetch.split_scalar_config(config)
      get_info = ee_export_utils.GetInfo("test-project", non_scalar_config)
      get_info.start_bundle()

      mock_pipeline_sources.assert_called_once_with(non_scalar_config)
      self.assertEqual(set(non_scalar_config.sources), {"patch"})

      item = {
          "id": 7,
          "lat": 47.37,
          "lon": 8.54,
          "bio": np.array([[[12.5, 800.0]]], dtype=np.float32),
          "bio_mask": np.array([[[1, 1]]], dtype=np.uint8),
          "bio_bands": np.array(["bio01", "bio12"]),
      }
      with mock.patch.object(
          get_info,
          "_make_request",
          return_value={
              "patch/b0": np.ones((2, 2), dtype=np.float32) * 3.0,
              "patch_mask/mask": np.ones((2, 2), dtype=np.uint8),
          },
      ):
        result = next(get_info.process(item))

    np.testing.assert_allclose(
        result["bio"], np.array([[[12.5, 800.0]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result["bio_mask"], np.array([[[1, 1]]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        result["bio_bands"], np.array(["bio01", "bio12"])
    )
    np.testing.assert_allclose(
        result["patch"], np.ones((2, 2, 1), dtype=np.float32) * 3.0
    )
    np.testing.assert_array_equal(result["patch_bands"], np.array(["b0"]))

  def test_get_info_passes_through_failed_items(self):
    config = _make_mixed_config()
    get_info = ee_export_utils.GetInfo("test-project", config)
    failed_item = {"failed": True, "id": 99, "error": "batch failed"}
    self.assertEqual(list(get_info.process(failed_item)), [failed_item])

  def test_process_single_item_with_only_batched_scalars_skips_get_info(self):
    config = _make_mixed_config()
    del config.sources["patch"]
    del config.labels["img_width_m"]

    mock_bio_im = mock.MagicMock()
    sampled_fc = mock.MagicMock()
    sampled_fc.getInfo.return_value = {
        "features": [{"properties": {"_idx": 0, "bio01": 11.0, "bio12": 900.0}}]
    }
    mock_bio_im.sampleRegions.return_value = sampled_fc

    with (
        mock.patch.object(
            scalar_fetch.pipelines,
            "pipeline_sources",
            return_value={"bio": mock_bio_im},
        ),
        mock.patch.object(
            ee_export_utils, "GetInfo", autospec=True
        ) as mock_get_info_cls,
    ):
      out = ee_export_utils.process_single_item(
          {"id": 1, "lat": 47.37, "lon": 8.54}, config
      )

    mock_get_info_cls.assert_not_called()
    np.testing.assert_allclose(
        out["bio"], np.array([[[11.0, 900.0]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out["bio_mask"], np.array([[[1, 1]]], dtype=np.uint8)
    )
    self.assertEqual(out["bio_bands"], ["bio01", "bio12"])

  def test_build_gee_fetch_pipeline_with_batched_scalars_only(self):
    config = _make_mixed_config()
    del config.sources["patch"]

    items = [
        {"id": 1, "lat": 47.37, "lon": 8.54},
        {"id": 2, "lat": 47.38, "lon": 8.55},
    ]
    with (
        mock.patch.object(
            scalar_fetch.SampleScalarSources,
            "start_bundle",
            autospec=True,
            return_value=None,
        ),
        mock.patch.object(
            scalar_fetch.SampleScalarSources,
            "process",
            autospec=True,
            side_effect=_fake_sample_process,
        ),
        mock.patch.object(
            ee_export_utils, "GetInfo", autospec=True
        ) as mock_get_info_cls,
    ):
      with test_pipeline.TestPipeline() as p:
        pcoll = p | beam.Create(items)
        examples = export_beam_tfds_utils.build_gee_fetch_pipeline(
            pcoll, config, "test-project"
        )
        beam_util.assert_that(examples, _check_fetch_pipeline_output)
      mock_get_info_cls.assert_not_called()


def _fake_sample_process(self_dofn, batch):
  del self_dofn
  for it in batch:
    out = dict(it)
    out["bio"] = np.array([[[float(it["id"]), 100.0]]], dtype=np.float32)
    out["bio_mask"] = np.array([[[1, 1]]], dtype=np.uint8)
    out["bio_bands"] = np.array(["bio01", "bio12"])
    yield out


def _check_fetch_pipeline_output(actual):
  assert len(actual) == 2, f"Expected 2 elements, got {len(actual)}"
  by_key = dict(actual)
  assert sorted(by_key.keys()) == ["1", "2"]
  np.testing.assert_allclose(
      by_key["1"]["bio"], np.array([[[1.0, 100.0]]], dtype=np.float32)
  )
  np.testing.assert_allclose(
      by_key["2"]["bio"], np.array([[[2.0, 100.0]]], dtype=np.float32)
  )
  assert by_key["1"]["bio_bands"] == ["bio01", "bio12"]


if __name__ == "__main__":
  absltest.main()
