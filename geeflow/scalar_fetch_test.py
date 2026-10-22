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

from typing import Any
from unittest import mock

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_util
from geeflow import scalar_fetch
import ml_collections
import numpy as np


def _make_config(*names: str, **overrides: Any) -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()
  config.labels = ml_collections.ConfigDict()
  config.sources = ml_collections.ConfigDict()
  for name in names:
    src = {
        "module": "Bioclim",
        "scalar": True,
        "algo": "im_static",
        "select": ["bio01", "bio12"],
        "mask_value": -9999.0,
    }
    src.update(overrides)
    config.sources[name] = ml_collections.ConfigDict(src)
  return config


def _item(item_id: int, lat: float, lon: float) -> dict[str, Any]:
  return {"id": item_id, "lat": lat, "lon": lon}


class BatchBySpatialCellTest(absltest.TestCase):

  def test_spatial_cell_key_combines_utm_and_degree_buckets(self):
    key_a = scalar_fetch.spatial_cell_key(47.2, 8.3, cell_size_deg=1.0)
    key_b = scalar_fetch.spatial_cell_key(47.8, 8.9, cell_size_deg=1.0)
    key_diff_lat = scalar_fetch.spatial_cell_key(48.1, 8.3, cell_size_deg=1.0)
    self.assertEqual(key_a, "32T:47:8")
    self.assertEqual(key_a, key_b)
    self.assertNotEqual(key_a, key_diff_lat)

  def test_groups_by_utm_and_degree_cell_and_respects_batch_size(self):
    items = [
        _item(0, 47.37, 8.54),
        _item(1, 47.38, 8.55),
        _item(2, 47.39, 8.56),
        _item(3, -33.86, 151.21),
    ]
    with test_pipeline.TestPipeline() as p:
      batches = (
          p
          | beam.Create(items)
          | scalar_fetch.BatchBySpatialCell(batch_size=2, cell_size_deg=1.0)
      )
      beam_util.assert_that(batches, _check_spatial_cell_batches)

  def test_from_config_reads_label_settings(self):
    config = _make_config("bio")
    config.labels.scalar_batch_size = 128
    config.labels.scalar_batch_cell_size_deg = 2.0
    transform = scalar_fetch.BatchBySpatialCell.from_config(config)
    self.assertEqual(transform.batch_size, 128)
    self.assertEqual(transform.cell_size_deg, 2.0)


class SampleScalarSourcesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(scalar_fetch.utils, "initialize_ee", autospec=True)
    )
    self.enter_context(mock.patch.object(scalar_fetch.ee.Geometry, "Point"))
    self.enter_context(mock.patch.object(scalar_fetch.ee, "Feature"))
    self.enter_context(mock.patch.object(scalar_fetch.ee, "FeatureCollection"))

  def test_split_scalar_config_partitions_supported_scalar_sources(self):
    config = _make_config("bio")
    config.sources.reduced_scalar = ml_collections.ConfigDict({
        "module": "Bioclim",
        "scalar": True,
        "algo": "ic_reduced",
    })
    config.sources.temporal_scalar = ml_collections.ConfigDict({
        "module": "Bioclim",
        "scalar": True,
        "algo": "ic_temporal_composites",
    })
    config.sources.fc_scalar = ml_collections.ConfigDict({
        "module": "CustomFC",
        "scalar": True,
        "algo": "fc_rasterize",
    })
    config.sources.patch = ml_collections.ConfigDict({
        "module": "Bioclim",
        "scalar": False,
        "algo": "im_static",
    })
    scalar_cfg, non_scalar_cfg = scalar_fetch.split_scalar_config(config)
    self.assertEqual(
        set(scalar_cfg.sources),
        {"bio", "reduced_scalar", "temporal_scalar"},
    )
    self.assertEqual(set(non_scalar_cfg.sources), {"fc_scalar", "patch"})
    self.assertEqual(
        set(config.sources),
        {"bio", "reduced_scalar", "temporal_scalar", "fc_scalar", "patch"},
    )

  def test_im_static_samples_native_projection_and_masks_missing_points(self):
    config = _make_config("bio")
    mock_image = mock.MagicMock()
    sampled_fc = mock.MagicMock()
    sampled_fc.getInfo.return_value = {
        "features": [
            {"properties": {"_idx": 0, "bio01": 12.5, "bio12": 800.0}},
            # Item 1 is omitted by EE sampleRegions (masked pixel).
            {"properties": {"_idx": 2, "bio01": 15.0, "bio12": None}},
        ]
    }
    mock_image.sampleRegions.return_value = sampled_fc

    with mock.patch.object(
        scalar_fetch.pipelines,
        "pipeline_sources",
        return_value={"bio": mock_image},
    ):
      dofn = scalar_fetch.SampleScalarSources("test-project", config)
      dofn.start_bundle()
      inputs = [
          _item(10, 47.0, 8.0),
          _item(11, 47.1, 8.1),
          _item(12, 47.2, 8.2),
      ]
      out = list(dofn.process(inputs))

    for inp in inputs:
      self.assertNotIn("bio", inp)

    mock_image.sampleRegions.assert_called_once()
    call_kwargs = mock_image.sampleRegions.call_args.kwargs
    self.assertIsNone(call_kwargs["projection"])
    self.assertIsNone(call_kwargs["scale"])
    self.assertEqual(call_kwargs["properties"], ["_idx"])
    self.assertFalse(call_kwargs["geometries"])

    self.assertLen(out, 3)
    np.testing.assert_allclose(
        out[0]["bio"], np.array([[[12.5, 800.0]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out[0]["bio_mask"], np.array([[[1, 1]]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        out[0]["bio_bands"], np.array(["bio01", "bio12"])
    )
    np.testing.assert_allclose(
        out[1]["bio"], np.array([[[-9999.0, -9999.0]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out[1]["bio_mask"], np.array([[[0, 0]]], dtype=np.uint8)
    )
    np.testing.assert_allclose(
        out[2]["bio"], np.array([[[15.0, -9999.0]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out[2]["bio_mask"], np.array([[[1, 0]]], dtype=np.uint8)
    )

  def test_ic_reduced_samples_first_then_reduces_by_point_index(self):
    config = _make_config(
        "aef",
        algo="ic_reduced",
        select=["A00", "A01"],
        date_range=("2022-01-01", "2023-01-01"),
    )
    mock_ic = mock.MagicMock()
    mock_filtered_bounds = mock.MagicMock()
    mock_filtered_date = mock.MagicMock()
    mock_mapped = mock.MagicMock()
    mock_flattened = mock.MagicMock()
    mock_reduced_dict = mock.MagicMock()
    mock_groups = mock.MagicMock()
    mock_groups.getInfo.return_value = [
        {"_idx": 0, "A00": 0.25, "A01": -0.5},
        {"_idx": 1, "A00": 0.75, "A01": 0.125},
    ]
    mock_ic.filterBounds.return_value = mock_filtered_bounds
    mock_filtered_bounds.filterDate.return_value = mock_filtered_date
    mock_filtered_date.map.return_value = mock_mapped
    mock_mapped.flatten.return_value = mock_flattened
    mock_flattened.reduceColumns.return_value = mock_reduced_dict
    mock_reduced_dict.get.return_value = mock_groups

    with (
        mock.patch.object(
            scalar_fetch.pipelines,
            "pipeline_sources",
            return_value={"aef": mock_ic},
        ),
        mock.patch.object(scalar_fetch.ee, "Reducer") as mock_reducer_cls,
    ):
      mock_reducer = mock.MagicMock()
      mock_reducer_cls.lastNonNull.return_value = mock_reducer

      dofn = scalar_fetch.SampleScalarSources("test-project", config)
      dofn.start_bundle()
      out = list(dofn.process([_item(0, 47.0, 8.0), _item(1, 47.1, 8.1)]))

    mock_ic.filterBounds.assert_called_once()
    mock_filtered_bounds.filterDate.assert_called_once_with(
        "2022-01-01", "2023-01-01"
    )
    mock_reducer_cls.lastNonNull.assert_called_once_with()
    mock_reducer.forEach.assert_called_once_with(["A00", "A01"])
    mock_flattened.reduceColumns.assert_called_once()
    reduce_kwargs = mock_flattened.reduceColumns.call_args.kwargs
    self.assertEqual(reduce_kwargs["selectors"], ["_idx", "A00", "A01"])

    self.assertLen(out, 2)
    np.testing.assert_allclose(
        out[0]["aef"], np.array([[[0.25, -0.5]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out[0]["aef_mask"], np.array([[[1, 1]]], dtype=np.uint8)
    )
    np.testing.assert_allclose(
        out[1]["aef"], np.array([[[0.75, 0.125]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out[1]["aef_mask"], np.array([[[1, 1]]], dtype=np.uint8)
    )

  def test_ic_reduced_date_range_fn_groups_by_date_range(self):
    config = _make_config(
        "aef",
        algo="ic_reduced",
        select=["A00", "A01"],
        date_range_fn=lambda it: (
            f"{it['year']}-01-01",
            f"{it['year'] + 1}-01-01",
        ),
    )
    mock_ic = mock.MagicMock()

    with (
        mock.patch.object(
            scalar_fetch.pipelines,
            "pipeline_sources",
            return_value={"aef": mock_ic},
        ),
        mock.patch.object(
            scalar_fetch.SampleScalarSources,
            "_sample_and_reduce_ic",
            side_effect=[
                # 2020 group: items 0 and 2.
                {
                    0: {"_idx": 0, "A00": 1.0, "A01": 2.0},
                    2: {"_idx": 2, "A00": 5.0, "A01": 6.0},
                },
                # 2022 group: item 1.
                {
                    1: {"_idx": 1, "A00": 3.0, "A01": 4.0},
                },
            ],
        ) as mock_sample_and_reduce,
    ):
      dofn = scalar_fetch.SampleScalarSources("test-project", config)
      dofn.start_bundle()
      inputs = [
          _item(0, 47.0, 8.0) | {"year": 2020},
          _item(1, 47.1, 8.1) | {"year": 2022},
          _item(2, 47.2, 8.2) | {"year": 2020},
      ]
      out = list(dofn.process(inputs))

    self.assertEqual(mock_sample_and_reduce.call_count, 2)
    self.assertEqual(
        mock_sample_and_reduce.call_args_list[0].kwargs["date_range"],
        ("2020-01-01", "2021-01-01"),
    )
    self.assertEqual(
        mock_sample_and_reduce.call_args_list[1].kwargs["date_range"],
        ("2022-01-01", "2023-01-01"),
    )
    self.assertLen(out, 3)
    np.testing.assert_allclose(
        out[0]["aef"], np.array([[[1.0, 2.0]]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        out[1]["aef"], np.array([[[3.0, 4.0]]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        out[2]["aef"], np.array([[[5.0, 6.0]]], dtype=np.float32)
    )

  def test_ic_temporal_composites_stacks_windows_and_timestamps(self):
    config = _make_config(
        "l8",
        algo="ic_temporal_composites",
        select=["B2", "B3"],
        date_ranges=[
            ("2022-01-01", 3, 0),
            ("2022-04-01", 3, 0),
        ],
        mask_value=-1.0,
    )
    mock_ic = mock.MagicMock()

    with (
        mock.patch.object(
            scalar_fetch.pipelines,
            "pipeline_sources",
            return_value={"l8": mock_ic},
        ),
        mock.patch.object(
            scalar_fetch.SampleScalarSources,
            "_sample_and_reduce_ic",
            side_effect=[
                # Window 0: both items present.
                {
                    0: {"_idx": 0, "B2": 100.0, "B3": 200.0},
                    1: {"_idx": 1, "B2": 300.0, "B3": 400.0},
                },
                # Window 1: only item 0 present (item 1 cloud-masked).
                {
                    0: {"_idx": 0, "B2": 110.0, "B3": 210.0},
                },
            ],
        ) as mock_sample_and_reduce,
    ):
      dofn = scalar_fetch.SampleScalarSources("test-project", config)
      dofn.start_bundle()
      out = list(dofn.process([_item(0, 47.0, 8.0), _item(1, 47.1, 8.1)]))

    self.assertEqual(mock_sample_and_reduce.call_count, 2)
    self.assertEqual(
        mock_sample_and_reduce.call_args_list[0].kwargs["date_range"],
        ("2022-01-01", "2022-04-01"),
    )
    self.assertEqual(
        mock_sample_and_reduce.call_args_list[1].kwargs["date_range"],
        ("2022-04-01", "2022-07-01"),
    )

    self.assertLen(out, 2)
    self.assertEqual(out[0]["l8"].shape, (2, 1, 1, 2))
    self.assertEqual(out[0]["l8_mask"].shape, (2, 1, 1, 2))
    np.testing.assert_allclose(
        out[0]["l8"][:, 0, 0, :],
        np.array([[100.0, 200.0], [110.0, 210.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        out[0]["l8_mask"][:, 0, 0, :],
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
    )
    np.testing.assert_allclose(
        out[1]["l8"][:, 0, 0, :],
        np.array([[300.0, 400.0], [-1.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        out[1]["l8_mask"][:, 0, 0, :],
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
    )
    self.assertEqual(out[0]["l8_timestamps"].dtype, np.int64)
    self.assertLen(out[0]["l8_timestamps"], 2)
    self.assertLess(out[0]["l8_timestamps"][0], out[0]["l8_timestamps"][1])
    np.testing.assert_array_equal(out[0]["l8_bands"], np.array(["B2", "B3"]))

  def test_ic_temporal_composites_date_ranges_fn_groups_by_windows(self):
    config = _make_config(
        "l8",
        algo="ic_temporal_composites",
        select=["B2"],
        date_ranges_fn=lambda it: [
            (f"{it['year']}-01-01", 6, 0),
            (f"{it['year']}-07-01", 6, 0),
        ],
        mask_value=-1.0,
    )
    mock_ic = mock.MagicMock()

    with (
        mock.patch.object(
            scalar_fetch.pipelines,
            "pipeline_sources",
            return_value={"l8": mock_ic},
        ),
        mock.patch.object(
            scalar_fetch.SampleScalarSources,
            "_sample_and_reduce_ic",
            side_effect=[
                # 2020 window 0 (item 0).
                {0: {"_idx": 0, "B2": 10.0}},
                # 2020 window 1 (item 0).
                {0: {"_idx": 0, "B2": 20.0}},
                # 2021 window 0 (item 1).
                {1: {"_idx": 1, "B2": 30.0}},
                # 2021 window 1 (item 1).
                {1: {"_idx": 1, "B2": 40.0}},
            ],
        ) as mock_sample_and_reduce,
    ):
      dofn = scalar_fetch.SampleScalarSources("test-project", config)
      dofn.start_bundle()
      inputs = [
          _item(0, 47.0, 8.0) | {"year": 2020},
          _item(1, 47.1, 8.1) | {"year": 2021},
      ]
      out = list(dofn.process(inputs))

    self.assertEqual(mock_sample_and_reduce.call_count, 4)
    np.testing.assert_allclose(
        out[0]["l8"][:, 0, 0, 0], np.array([10.0, 20.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        out[1]["l8"][:, 0, 0, 0], np.array([30.0, 40.0], dtype=np.float32)
    )
    self.assertLess(out[0]["l8_timestamps"][0], out[1]["l8_timestamps"][0])

  def test_skip_failing_plots_yields_failed_markers(self):
    config = _make_config("bio")
    config.labels.skip_failing_plots = True
    mock_image = mock.MagicMock()
    mock_image.sampleRegions.side_effect = RuntimeError("EE boom")

    with mock.patch.object(
        scalar_fetch.pipelines,
        "pipeline_sources",
        return_value={"bio": mock_image},
    ):
      dofn = scalar_fetch.SampleScalarSources("test-project", config)
      dofn.start_bundle()
      out = list(dofn.process([_item(42, 47.0, 8.0), _item(43, 47.1, 8.1)]))

    self.assertEqual(
        out,
        [
            {"failed": True, "id": 42, "error": "EE boom"},
            {"failed": True, "id": 43, "error": "EE boom"},
        ],
    )


def _check_spatial_cell_batches(actual_batches):
  all_ids = []
  for b in actual_batches:
    assert len(b) <= 2, f"Batch size {len(b)} exceeds maximum 2"
    all_ids.extend(it["id"] for it in b)
    keys = {
        scalar_fetch.spatial_cell_key(it["lat"], it["lon"], cell_size_deg=1.0)
        for it in b
    }
    assert len(keys) == 1, f"Expected 1 spatial cell key per batch, got {keys}"
  assert sorted(all_ids) == [0, 1, 2, 3], f"Unexpected IDs: {sorted(all_ids)}"


if __name__ == "__main__":
  absltest.main()
