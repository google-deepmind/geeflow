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

"""Utils for running EE exports."""

import collections
from collections.abc import Sequence
import functools
import io
import itertools
import math
import re
import time
from typing import Any

from absl import logging
import apache_beam as beam
from geeflow import ee_algo
from geeflow import pipelines
from geeflow import times
from geeflow import utils
import ml_collections
import numpy as np
import toolz

import ee

_LONGITUDE_STEP = 6
_LATITUDE_STEP = 8
_MAX_RETRIES = 5
_INITIAL_RETRY_SLEEP_TIMEOUT_SECS = 0.5
_RETRY_SLEEP_TIMEOUT_MULTIPLIER = 2
_MAX_HEADER_SIZE = 100_000

_CCDC_BANDS = [
    "BLUE",
    "GREEN",
    "NIR",
    "RED",
    "SWIR1",
    "SWIR2",
    "changeProb",
    "numObs",
    "tBreak",
    "tEnd",
    "tStart",
]
_CCDC_PROP = ["", "_coefs", "_magnitude", "_rmse"]
_MAX_CCDC_SEGMENTS = 100
_CCDC_DIM = 8


_USER_MEM_LOG = (
    "User memory limit exceeded for item: %s. Retries number: %d, exception: %s"
)
_GENERIC_LOG = (
    "Encountered an earth engine exception for item: %s. Retries number: %d, "
    "exception: %s"
)


def _should_split_query(err: str) -> bool:
  """Returns whether the query should be split."""
  pattern = "Number of bands (.*) must be less than or equal to 1024."
  return (
      "User memory limit exceeded" in err
      or "Total request size" in err
      or (re.match(pattern, err) is not None)
  )


def get_utm_zones():
  """Returns a list of utm zones."""
  zones = []
  for lat in range(-72, 80, _LATITUDE_STEP):
    for lon in range(-180, 180, _LONGITUDE_STEP):
      start_lon = lon
      start_lat = lat
      lon_step = _LONGITUDE_STEP
      lat_step = _LATITUDE_STEP
      # 31V/32V correction
      if lat == 56:
        if lon == 0:
          lon_step = 3
        elif lon == 6:
          lon_step = 9
          start_lon -= 3
      elif lat == 72:
        # X-zone with 12 lat degrees and 31X, 33X, 35X, 37X corrections.
        lat_step = 12
        if lon == 0:
          lon_step = 9
        elif lon == 6:
          continue
        elif lon == 12:
          start_lon = 9
          lon_step = 12
        elif lon == 18:
          continue
        elif lon == 24:
          start_lon = 12
          lon_step = 21
        elif lon == 30:
          continue
        elif lon == 36:
          start_lon = 33
          lon_step = 9
      zones.append((start_lat, start_lon, lat_step, lon_step))
  return zones


class GetInfo(beam.DoFn):
  """Beam DoFn to extract necessary info and emit features one by one."""

  def __init__(
      self,
      ee_project,
      config: ml_collections.ConfigDict,
      num_calls_counter=None,
      num_failures_counter=None,
      num_items_processed=None,
      num_query_retries=None,
      query_duration_distribution=None,
      query_bytes_distribution=None,
      query_sleep_distribution=None,
      add_retries_counters=False,
  ):
    self.ee_project = ee_project
    self.config = config
    self.num_calls_counter = num_calls_counter
    self.num_failures_counter = num_failures_counter
    self.num_items_processed = num_items_processed
    self.num_query_retries = num_query_retries
    self.query_duration_distribution = query_duration_distribution
    self.query_bytes_distribution = query_bytes_distribution
    self.query_sleep_distribution = query_sleep_distribution
    self.retries_counters = dict() if add_retries_counters else None
    self.min_source_split = dict()
    self.band_names = dict()

  def start_bundle(self):
    # Initialize and authenticate EE on the flume workers.
    utils.initialize_ee(self.ee_project)
    sources = pipelines.pipeline_sources(self.config)
    self.process_fn_dict = pipelines.get_requests_fn(self.config, sources)

  def _query(self, exp):
    res = ee.data.computePixels(dict(expression=exp, fileFormat="npy"))
    if self.query_bytes_distribution:
      self.query_bytes_distribution.update(len(res) / 1e6)
    if self.num_calls_counter:
      self.num_calls_counter.inc()
    res = np.load(io.BytesIO(res), max_header_size=_MAX_HEADER_SIZE)  # pylint: disable=unexpected-keyword-arg
    return {k: res[k] for k in res.dtype.names}

  def _split_query(
      self, exp: ee.Image, bands: Sequence[str], num_splits: int, source: str
  ):
    res = {}
    for split in np.array_split(bands, num_splits):
      res |= self._query(exp.select(list(map(str, split))))
    # The -1 value corresponds to spliting bands individually.
    counter_key = -1 if len(bands) == num_splits else num_splits
    if self.retries_counters is not None:
      self.retries_counters.setdefault(
          (source, counter_key),
          beam.metrics.Metrics.counter(source, f"retry_{counter_key}"),
      ).inc()
    return res

  def _make_request(self, request: pipelines.Request, item: dict[str, Any]):
    exceptions = []
    sleep_timeout = _INITIAL_RETRY_SLEEP_TIMEOUT_SECS
    crs = request.crs or request.roi.projection().crs()
    im = request.image.reproject(crs=crs, scale=request.scale)
    exp = im.clipToBoundsAndScale(geometry=request.roi)
    # Cache band names for each request to avoid repeated calls. If the band
    # names are not fixed, we need to update them each time.
    if not request.fixed_band_names or request.name not in self.band_names:
      self.band_names[request.name] = exp.bandNames().getInfo()
    num_bands = len(band_names := self.band_names[request.name])
    for i in range(_MAX_RETRIES):
      try:
        # If the query has never failed, we try to run it.
        if (request.name, num_bands) not in self.min_source_split:
          return self._query(exp)
        # Otherwise, we try try to run the query with the minimum number of
        # splits that we have previously found to be successful.
        else:
          min_source_split = self.min_source_split[request.name, num_bands]
          return self._split_query(
              exp, band_names, min_source_split, request.name
          )
      except ee.EEException as e:
        err = str(e)
        exceptions.append(err)
        if _should_split_query(err):
          logging.exception(_USER_MEM_LOG, item, i, err)
          # If the query has failed, then we retry the query with a larger
          # number of splits starting from splitting the bands in half.
          min_source_split = (
              self.min_source_split.get((request.name, num_bands), 1) + 1
          )
          for split in range(min_source_split, len(band_names) + 1):
            try:
              res = self._split_query(exp, band_names, split, request.name)
              if _should_split_query(err):
                # This errors means that the query corresponding to this source
                # was too large. This is not a temporary error. So we increase
                # the number of splits for future queries of this source.
                # We also include the number of bands in the min_source_split
                # key to account for time varying ee_algos. This is because when
                # the number of timesteps changes, the number of queries bands
                # changes.
                self.min_source_split[request.name, num_bands] = split
              # The "User memory limit exceeded" is instead a temporary error.
              # So we keep trying without splitting the query further.
              return res
            except ee.EEException as e2:
              if split == len(band_names):
                # The query failed and we already tried to split it as much as
                # possible. So we raise the error.
                raise ValueError(f"Failed to split {request.name}.") from e2
        else:
          # Retry soon.
          logging.exception(_GENERIC_LOG, item, i, e)
          if self.num_query_retries:
            self.num_query_retries.inc()
          time.sleep(sleep_timeout)
          if self.query_sleep_distribution:
            self.query_sleep_distribution.update(sleep_timeout)
          sleep_timeout *= _RETRY_SLEEP_TIMEOUT_MULTIPLIER

    raise RuntimeError(
        f"EE query failed after {_MAX_RETRIES} retries for item: {item} with"
        f" exceptions: {exceptions}."
    )

  def process(self, item):
    start = time.time()
    rois = pipelines.pipeline_item_to_rois(self.config, item)
    requests, metadata = self.process_fn_dict(rois, item)
    if self.num_items_processed:
      self.num_items_processed.inc()
    # b/393557949 - We are making a ee.data.computePixels per image. However,
    # we could group multiple calls together if they have the same roi, crs,
    # and scale. Note this would require to account for the maximum number of
    # bytes per request (~50MB).
    pixels = functools.reduce(
        lambda x, y: x | y, (self._make_request(r, item) for r in requests)
    )
    if time.time() - start > 20:
      logging.info("Long EE query: %s for %s", time.time() - start, item)
    if self.query_duration_distribution:
      self.query_duration_distribution.update(time.time() - start)

    yield _process_ee_query_result(pixels, metadata, self.config)


def _process_ee_query_result(pixels, metadata, config) -> dict[str, np.ndarray]:
  """Format EE query result."""
  # Process CCDC data.
  ccdc_pixels = {}
  for k in config.sources:
    if pipelines.get_algo_from_config(config, k) == ee_algo.get_ccdc:
      for b, c in itertools.product(_CCDC_BANDS, _CCDC_PROP):
        if f"{k}_{b}{c}#0#0" not in pixels:
          continue
        tmp = [
            [
                pixels.pop(f"{k}_{b}{c}#{j}#{i}")
                for j in range(_MAX_CCDC_SEGMENTS)
                if f"{k}_{b}{c}#{j}#{i}" in pixels
            ]
            for i in range(_CCDC_DIM)
            if f"{k}_{b}{c}#0#{i}" in pixels
        ]
        ccdc_pixels[f"{k}_{b}{c}"] = np.transpose(tmp, (2, 3, 1, 0)).squeeze()

  # Concatenate the channels from the same source on the last dimension.
  tmp = collections.defaultdict(list)
  for k in list(pixels):
    name, *_ = k.split("/")
    tmp[name].append(pixels.pop(k))
  pixels = toolz.valmap(np.dstack, tmp)

  # Concatenate timesteps from same source on the first dimension.
  tmp = collections.defaultdict(list)
  for source in sorted(pixels):
    # Non temporal sources are not marked with a "#".
    if "#" not in source:
      tmp[source] = pixels.pop(source)
    # Temporal sources. Note that we are guaranteed the correct temporal
    # ordering since computePixels requests are made in the correct order.
    else:
      tmp[source.split("#")[0]].append(pixels.pop(source))
  pixels = toolz.valmap(np.array, tmp)

  # Ensure that assets with variable number of timesteps are in the right
  # format.
  for k in set(k.replace("_mask", "") for k in pixels):
    if pipelines.get_algo_from_config(config, k) in [
        ee_algo.ic_sample,
        ee_algo.ic_sample_date_ranges,
    ]:
      if pixels[k].ndim == 3:
        pixels[k] = pixels[k][None,]
        pixels[k + "_mask"] = pixels[k + "_mask"][None,]

  # Add empty arrays for missing sources. This happens for ee_algo.ic_sample
  # when there is no data for the given item.
  for key in config.sources:
    is_time_varying_source = (
        pipelines.get_algo_from_config(config, key) == ee_algo.ic_sample
    )
    if key not in pixels and is_time_varying_source:
      for postfix in ["", "_mask", "_timestamps"]:
        pixels[key + postfix] = np.array([])

  return pixels | ccdc_pixels | metadata


def _has_forest_loss(feature):
  return np.max(np.array(feature["hansen"])[..., 1]) > 0


def apply_filters(feature, config):
  export_config = config.get("export", {})
  if export_config.get("filter_empty_forest_loss", False):
    if not _has_forest_loss(feature):
      return False
  if keys := export_config.get("filter_empty_sequences", []):
    keys = [keys] if isinstance(keys, str) else keys
    for key in keys:
      if not feature[key]:
        return False
  return True


def generate_ccdc(data, config, key: str):
  """Generates CCDC data in [T,H,W,C] format."""
  assert f"{key}_tStart" in data
  start_dates = np.array(data[f"{key}_tStart"])
  h = start_dates.shape[0]
  w = start_dates.shape[1]
  year_from = config["from"]
  year_to = config["to"]
  ignore_tstart = config.get("ignore_tstart", False)
  num_bands = 0
  ccdc_features = {}
  for k, v in data.items():
    if not k.startswith(f"{key}_"): continue
    if ignore_tstart and k == f"{key}_tStart": continue
    nv = np.array(v)
    ccdc_features[k] = nv
    if len(nv.shape) == 3:
      num_bands += 1
    else:
      num_bands += nv.shape[-1]
  ccdc = np.zeros((year_to - year_from + 1, h, w, num_bands), dtype=np.float32)
  ccdc_mask = np.ones(ccdc.shape, dtype=np.bool_)

  for x in range(h):
    for y in range(w):
      dates = start_dates[x, y]
      num_segments = dates.shape[0]
      segment_pos = 0
      previous_segment_pos = -1
      for year in range(year_from, year_to + 1):
        while (segment_pos + 1 < num_segments and
               year + 0.5 >= dates[segment_pos + 1] and
               dates[segment_pos + 1] > 0):
          segment_pos += 1
        if dates[segment_pos] == 0:
          # There's no data (value should be fractional year) -> mask out.
          ccdc_mask[year - year_from, x, y] = np.zeros_like(
              ccdc_mask[year - year_from, x, y])
        else:
          if previous_segment_pos == segment_pos:
            # No need to recompute, the segment didn't change.
            # Copy previous value.
            ccdc[year - year_from, x, y] = ccdc[year - year_from - 1, x, y]
          else:
            a = []
            for v in ccdc_features.values():
              if len(v.shape) == 3:
                a.append(v[x, y, segment_pos])
              else:
                a.extend(v[x, y, segment_pos])

            ccdc[year - year_from, x, y] = a
            previous_segment_pos = segment_pos
  if "year_selection" in config:
    ccdc = ccdc[config["year_selection"]]
    ccdc_mask = ccdc_mask[config["year_selection"]]
  return ccdc, ccdc_mask


def apply_transforms(feature, config, cropped_features_counter=None):
  """Apply transforms to feature."""
  for key, value in config.sources.items():
    if key.startswith("ccdc"):
      keys = [x for x in feature.keys() if x.startswith(f"{key}_")]
      feature[key], feature[f"{key}_mask"] = generate_ccdc(
          feature, value.format_config, key)
      for k in keys:
        del feature[k]

  default_image_width = config.labels.img_width_m
  for key, value in config.sources.items():
    if value.get("scalar", False): continue
    image_width = value.get("img_width_m", default_image_width)
    scale = value.get("scale", None)
    if not scale: continue
    assert image_width >= scale, f"{key}: {image_width} < {scale}"
    if isinstance(scale, float):
      s = math.ceil(image_width / scale)
      assert abs(s  * scale - image_width) < 1e-6
    else:
      if config.labels.get("use_utm", True):
        assert image_width % scale == 0, f"{key}: {image_width} % {scale} != 0"
        s = image_width // scale
      else:
        s = math.ceil(image_width / scale)

    for sub_key in [key, f"{key}_mask"]:
      data = np.array(feature[sub_key])
      # The expected data format is [H, W, C] or [T, H, W, C]. Even if C==1.
      if len(data.shape) != 4 and len(data.shape) != 3: continue
      if data.shape[-3] != s or data.shape[-2] != s:
        if cropped_features_counter:
          cropped_features_counter.inc()
        logging.info("Cropping %s from %s to %s", sub_key, data.shape, s)
        assert data.shape[-3] == s or data.shape[-3] == s + 1, sub_key
        assert data.shape[-2] == s or data.shape[-2] == s + 1, sub_key
        feature[sub_key] = data[
            ...,
            data.shape[-3] // 2 - s // 2 : data.shape[-3] // 2 + s - s // 2,
            data.shape[-2] // 2 - s // 2 : data.shape[-2] // 2 + s - s // 2,
            :]
  return feature


def process_example(x, config):
  """Process a single example."""
  for k in config.get("skip_keys", []) + ["split"]:
    x.pop(k, None)
  out = {}
  for k, v in x.items():
    dtype = None
    if k.endswith("_mask") or k == "hr":
      dtype = np.uint8
    elif config.sources.get(k):
      dtype = config.sources[k].get("dtype")
    t = np.array(v, dtype=dtype)
    # Automatically convert data to float (even if array is empty).
    # Exceptions are:
    #  - !"dtype is None" - predetermined types
    #  - !"isinstance(t.flat[0], np.integer)" - don't touch non int types
    #  - !"in ignore_for_float_conversion" - keys in the exception list
    if (dtype is None and
        (not t.size or (isinstance(t.flat[0], np.integer) or
                        isinstance(t.flat[0], np.float64))) and
        k not in config.sources.get("ignore_for_float_conversion", []) and
        k not in config.labels.get("ignore_for_float_conversion", [])):
      t = t.astype(np.float32)
    if t.shape:
      out[k] = t
    else:
      # Keep scalars as they are.
      out[k] = v
  tfds_id_keys = config.labels.get("tfds_id_keys", ("id",))
  key = "-".join(map(str, (x[k] for k in tfds_id_keys)))
  return (key, out)


def process_single_item(
    item: dict[str, Any],
    config: ml_collections.ConfigDict,
    ee_project: str = "skip",
) -> dict[str, Any]:
  """Process a single item outside of a beam pipeline."""
  get_info = GetInfo(ee_project, config)
  get_info.start_bundle()
  feature_data = next(get_info.process(item))
  feature_data = apply_transforms(feature_data, config)
  if post_process_map := config.get("post_process_map"):
    if isinstance(post_process_map, beam.DoFn):
      post_process_map.setup()
      post_process_map.start_bundle()
      # Assuming that the post_process_map.process function takes `flush` makes
      # things significantly simpler for the inference pipe, which constructs
      # batches of data and the finish bundle returns
      # `beam.transforms.window.WindowedValue` objects instead of dict.
      feature_data = next(post_process_map.process(feature_data, flush=True))
    elif callable(post_process_map):
      feature_data = post_process_map(feature_data)
    else:
      raise ValueError(f"Unsupported post_process_map: {post_process_map}")
  _, data = process_example(feature_data, config)
  return data


# TODO: Add support for ee.Image and ee.FeatureCollection.
def fetch_data(
    lat: float,
    lon: float,
    collection: str,
    img_width_m: float,
    resolution: float,
    start_date: str = "",
    end_date: str = "",
    bands: Sequence[str] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
  """Fetches data from Earth Engine for a given location and time range.

  Args:
    lat: Latitude of the center of the region to fetch.
    lon: Longitude of the center of the region to fetch.
    collection: Name of the Earth Engine image collection to use.
    img_width_m: Width of the image in meters.
    resolution: Resolution of the image in meters per pixel.
    start_date: Start date of the time range (format: YYYY-MM-DD). Optional.
    end_date: End date of the time range (format: YYYY-MM-DD). Optional.
    bands: List of bands to select from the image collection. Default is all
      bands.

  Returns:
      A numpy array with the fetched data and a dictionary with the metadata.
  """
  cfg = ml_collections.config_dict.create(
      sources=dict(
          collection=dict(
              out="ic",
              module="CustomIC",
              algo="ic_sample",
              kw=dict(asset_name=collection),
              scale=resolution,
          )
      ),
      labels=dict(img_width_m=img_width_m, max_cell_size_m=resolution),
  )
  if bands:
    cfg.sources.collection.select_final = bands
  if start_date:
    cfg.sources.collection.start_date = start_date
  if end_date:
    cfg.sources.collection.end_date = end_date

  data = process_single_item(dict(lat=lat, lon=lon, id=0), cfg)
  dates = list(map(times.to_datestr, data["collection_timestamps"].astype(int)))
  metadata = dict(mask=data["collection_mask"], dates=dates)
  return data["collection"], metadata
