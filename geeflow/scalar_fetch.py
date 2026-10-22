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

"""Batched native-grid Earth Engine fetching for scalar (point) sources."""

import collections
from collections.abc import Iterable, Sequence
import copy
import itertools
import math
from typing import Any

from absl import logging
import apache_beam as beam
from geeflow import ee_algo
from geeflow import pipelines
from geeflow import times
from geeflow import utils
from geeflow import utm_lib
import ml_collections
import numpy as np

import ee

_DEFAULT_BATCH_SIZE = 500
_DEFAULT_CELL_SIZE_DEG = 1.0
_DEFAULT_TILE_SCALE = 4
_IDX_KEY = "_idx"

SUPPORTED_SCALAR_ALGOS = frozenset({
    ee_algo.im_static,
    ee_algo.ic_reduced,
    ee_algo.ic_temporal_composites,
})


def split_scalar_config(
    config: ml_collections.ConfigDict,
) -> tuple[ml_collections.ConfigDict, ml_collections.ConfigDict]:
  """Returns `(scalar_config, non_scalar_config)` with `sources` partitioned."""
  scalar_config = copy.copy(config)
  non_scalar_config = copy.copy(config)
  scalar_config.sources = ml_collections.ConfigDict({
      k: v
      for k, v in config.sources.items()
      if v.get("scalar")
      and pipelines.get_algo_from_config(config, k) in SUPPORTED_SCALAR_ALGOS
  })
  non_scalar_config.sources = ml_collections.ConfigDict({
      k: v for k, v in config.sources.items() if k not in scalar_config.sources
  })
  return scalar_config, non_scalar_config


def spatial_cell_key(
    lat: float, lon: float, cell_size_deg: float = _DEFAULT_CELL_SIZE_DEG
) -> str:
  """Returns a spatial cell key combining UTM zone and lat/lon bucket."""
  if -80.0 <= lat <= 84.0:
    _, _, zone_number, zone_letter = utm_lib.from_latlon(lat, lon)
    utm_zone = f"{zone_number}{zone_letter}"
  else:
    utm_zone = "POLAR"
  lat_bin = math.floor(lat / cell_size_deg)
  lon_bin = math.floor(lon / cell_size_deg)
  return f"{utm_zone}:{lat_bin}:{lon_bin}"


class BatchBySpatialCell(beam.PTransform):
  """Groups items into spatial batches keyed by UTM zone and lat/lon bucket."""

  def __init__(
      self,
      batch_size: int = _DEFAULT_BATCH_SIZE,
      cell_size_deg: float = _DEFAULT_CELL_SIZE_DEG,
  ):
    super().__init__()
    self.batch_size = batch_size
    self.cell_size_deg = cell_size_deg

  @classmethod
  def from_config(
      cls, config: ml_collections.ConfigDict
  ) -> "BatchBySpatialCell":
    labels = config.get("labels", {})
    return cls(
        batch_size=labels.get("scalar_batch_size", _DEFAULT_BATCH_SIZE),
        cell_size_deg=labels.get(
            "scalar_batch_cell_size_deg", _DEFAULT_CELL_SIZE_DEG
        ),
    )

  def expand(
      self, pcoll: beam.PCollection[dict[str, Any]]
  ) -> beam.PCollection[list[dict[str, Any]]]:
    cell_size_deg = self.cell_size_deg
    batch_size = self.batch_size
    return (
        pcoll
        | "KeyBySpatialCell"
        >> beam.Map(
            lambda item: (
                spatial_cell_key(item["lat"], item["lon"], cell_size_deg),
                item,
            )
        )
        | "GroupBySpatialCell" >> beam.GroupByKey()
        | "ChunkIntoBatches"
        >> beam.FlatMap(_chunk_grouped_items, batch_size=batch_size)
        # Break fusion so batches from one dense cell spread across workers.
        | "ReshuffleBatches" >> beam.Reshuffle()
    )


def _chunk_grouped_items(
    kv: tuple[str, Iterable[dict[str, Any]]], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
  """Chunks grouped items into lists of at most `batch_size`."""
  it = iter(kv[1])
  while True:
    batch = list(itertools.islice(it, batch_size))
    if not batch:
      break
    yield batch


def _items_to_fc(
    items: Sequence[dict[str, Any]], indices: Sequence[int] | None = None
) -> ee.FeatureCollection:
  """Builds an `ee.FeatureCollection` of points tagged with `_idx`."""
  if indices is None:
    indices = range(len(items))
  return ee.FeatureCollection([
      ee.Feature(
          ee.Geometry.Point([float(items[i]["lon"]), float(items[i]["lat"])]),
          {_IDX_KEY: i},
      )
      for i in indices
  ])


class SampleScalarSources(beam.DoFn):
  """Samples scalar sources on their native projections via EE."""

  def __init__(
      self,
      ee_project: str,
      config: ml_collections.ConfigDict,
  ):
    self.ee_project = ee_project
    self.config = config
    self.num_calls_counter = beam.metrics.Metrics.counter(
        "GEEScalarFetch", "num_query_calls"
    )
    self.num_failures_counter = beam.metrics.Metrics.counter(
        "GEEScalarFetch", "num_query_failures"
    )
    self.num_items_processed = beam.metrics.Metrics.counter(
        "GEEScalarFetch", "num_items_processed"
    )
    self.sources: dict[str, Any] = {}
    self.band_names: dict[str, list[str]] = {}
    self.dtypes: dict[str, np.dtype] = {}

  def start_bundle(self):
    utils.initialize_ee(self.ee_project)
    self.sources = pipelines.pipeline_sources(self.config)
    for name, asset in self.sources.items():
      cfg = self.config.sources[name]
      if bands := (cfg.get("select_final") or cfg.get("select")):
        self.band_names[name] = (
            [bands] if isinstance(bands, str) else list(bands)
        )
      else:
        first_im = (
            asset if isinstance(asset, ee.Image) else ee.Image(asset.first())
        )
        self.band_names[name] = list(
            utils.retry_ee_call(first_im.bandNames().getInfo)
        )
      self.dtypes[name] = np.dtype(cfg.get("dtype") or np.float32)

  def _sample_image(
      self,
      image: ee.Image,
      fc: ee.FeatureCollection,
      tile_scale: int = _DEFAULT_TILE_SCALE,
  ) -> dict[int, dict[str, Any]]:
    """Samples an `ee.Image` on its native projection and scale."""
    sampled = image.sampleRegions(
        collection=fc,
        properties=[_IDX_KEY],
        projection=None,
        scale=None,
        tileScale=tile_scale,
        geometries=False,
    )
    self.num_calls_counter.inc()
    info = utils.retry_ee_call(sampled.getInfo)
    return {
        int(f["properties"][_IDX_KEY]): f["properties"]
        for f in info.get("features", [])
        if _IDX_KEY in f.get("properties", {})
    }

  def _sample_and_reduce_ic(
      self,
      ic: ee.ImageCollection,
      fc: ee.FeatureCollection,
      bands: list[str],
      sampling_kw: dict[str, Any],
      date_range: tuple[str, str] | None = None,
      limit: int | None = None,
  ) -> dict[int, dict[str, Any]]:
    """Samples each image on its native grid, then reduces across time."""
    tile_scale = sampling_kw.get("tile_scale", _DEFAULT_TILE_SCALE)
    reduce_fn = (
        sampling_kw.get("reduce_fn") or sampling_kw.get("reducer") or "mosaic"
    )
    filtered = ee_algo.preprocess_ic(
        roi=fc.geometry().bounds(),
        ic=ic,
        cloud_mask_fn=sampling_kw.get("cloud_mask_fn"),
        sort_by=sampling_kw.get("sort_by"),
        ascending=sampling_kw.get("ascending", True),
        bands=sampling_kw.get("bands"),
        filter_bounds=sampling_kw.get("filter_bounds", True),
        roi_filter_fn=sampling_kw.get("roi_filter_fn"),
        date_range=date_range,
        limit=limit,
        match_season_date=sampling_kw.get("match_season_date"),
        match_season_window_days=sampling_kw.get(
            "match_season_window_days", 45
        ),
        temporal_anchor_ts=sampling_kw.get("temporal_anchor_ts"),
        time_window_days=sampling_kw.get("time_window_days"),
    )

    samples = filtered.map(
        lambda img: img.sampleRegions(
            collection=fc,
            properties=[_IDX_KEY],
            projection=None,
            scale=None,
            tileScale=tile_scale,
            geometries=False,
        )
    ).flatten()
    ee_reducer = ee_algo.get_ic_sample_reducer(reduce_fn)
    grouped = samples.reduceColumns(
        reducer=ee_reducer.forEach(bands).group(
            groupField=0, groupName=_IDX_KEY
        ),
        selectors=[_IDX_KEY, *bands],
    ).get("groups")
    self.num_calls_counter.inc()
    groups = utils.retry_ee_call(grouped.getInfo) or []
    return {int(g[_IDX_KEY]): g for g in groups if _IDX_KEY in g}

  def _fetch_source_for_batch(
      self,
      name: str,
      items: Sequence[dict[str, Any]],
      fc: ee.FeatureCollection,
  ) -> None:
    cfg = self.config.sources[name]
    asset = self.sources[name]
    bands = self.band_names[name]
    dtype = self.dtypes[name]
    mask_value = cfg.get("mask_value", 0)
    sampling_kw = cfg.get("sampling_kw", {})
    limit = cfg.get("limit", sampling_kw.get("limit"))
    if cfg.get("select_final"):
      asset = asset.select(bands)
    bands_arr = np.array(bands)

    match pipelines.get_algo_from_config(self.config, name):
      case ee_algo.im_static:
        tile_scale = sampling_kw.get("tile_scale", _DEFAULT_TILE_SCALE)
        rows_by_idx = self._sample_image(asset, fc, tile_scale=tile_scale)
      case ee_algo.ic_reduced:
        indices_by_range = collections.defaultdict(list)
        for idx, item in enumerate(items):
          indices_by_range[pipelines.resolve_date_range(cfg, item)].append(idx)
        rows_by_idx = {}
        for dr, indices in indices_by_range.items():
          group_fc = (
              fc if len(indices) == len(items) else _items_to_fc(items, indices)
          )
          rows_by_idx |= self._sample_and_reduce_ic(
              asset,
              group_fc,
              bands=bands,
              sampling_kw=sampling_kw,
              date_range=dr,
              limit=limit,
          )
      case ee_algo.ic_temporal_composites:
        indices_by_ranges = collections.defaultdict(list)
        for idx, item in enumerate(items):
          indices_by_ranges[pipelines.resolve_date_ranges(cfg, item)].append(
              idx
          )
        for date_ranges, indices in indices_by_ranges.items():
          group_fc = (
              fc if len(indices) == len(items) else _items_to_fc(items, indices)
          )
          windows, timestamps = times.resolve_date_ranges(date_ranges)
          rows_per_t = [
              self._sample_and_reduce_ic(
                  asset,
                  group_fc,
                  bands=bands,
                  sampling_kw=sampling_kw,
                  date_range=window,
                  limit=limit,
              )
              for window in windows
          ]
          for idx in indices:
            item = items[idx]
            t_vals = []
            t_masks = []
            for rows_by_idx in rows_per_t:
              row = rows_by_idx.get(idx, {})
              vals = [row.get(b) for b in bands]
              t_vals.append([mask_value if v is None else v for v in vals])
              t_masks.append([int(v is not None) for v in vals])
            item[name] = np.array(t_vals, dtype=dtype)[:, None, None, :]
            item[f"{name}_mask"] = np.array(t_masks, dtype=np.uint8)[
                :, None, None, :
            ]
            item[f"{name}_bands"] = bands_arr
            item[f"{name}_timestamps"] = timestamps
        return
      case algo:
        raise ValueError(f"Unsupported scalar algo for {name!r}: {algo}")

    for idx, item in enumerate(items):
      row = rows_by_idx.get(idx, {})
      vals = [row.get(b) for b in bands]
      item[name] = np.array(
          [[[mask_value if v is None else v for v in vals]]], dtype=dtype
      )
      item[f"{name}_mask"] = np.array(
          [[[int(v is not None) for v in vals]]], dtype=np.uint8
      )
      item[f"{name}_bands"] = bands_arr

  def process(
      self, items: Sequence[dict[str, Any]]
  ) -> Iterable[dict[str, Any]]:
    items = [dict(it) for it in items]
    try:
      fc = _items_to_fc(items)
      for name in self.sources:
        self._fetch_source_for_batch(name, items, fc)
      self.num_items_processed.inc(len(items))
      yield from items
    except Exception as e:  # pylint: disable=broad-except
      if self.config.get("labels", {}).get("skip_failing_plots", False):
        logging.error(
            "Failed to sample scalar sources for batch of %d items: %s",
            len(items),
            e,
        )
        self.num_failures_counter.inc(len(items))
        for item in items:
          yield {"failed": True, "id": item.get("id"), "error": str(e)}
      else:
        raise
