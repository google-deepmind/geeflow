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
import time
from typing import Any

from absl import logging
import apache_beam as beam
from geeflow import ee_algo
from geeflow import pipelines
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
        if "User memory limit exceeded" in err or "Total request size" in err:
          logging.exception(_USER_MEM_LOG, item, i, err)
          # If the query has failed, then we retry the query with a larger
          # number of splits starting from splitting the bands in half.
          min_source_split = (
              self.min_source_split.get((request.name, num_bands), 1) + 1
          )
          for split in range(min_source_split, len(band_names) + 1):
            try:
              res = self._split_query(exp, band_names, split, request.name)
              if "Total request size" in err:
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
        f"EE query failed after {_MAX_RETRIES} retries for item: {item}."
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
        ee_algo.rgb_ic_sample,
        ee_algo.ic_sample_date_ranges,
    ]:
      if pixels[k].ndim == 3:
        pixels[k] = pixels[k][None,]
        pixels[k + "_mask"] = pixels[k + "_mask"][None,]

  # Add empty arrays for missing sources. This happens for ee_algo.ic_sample
  # when there is no data for the given item.
  for key in config.sources:
    if key not in pixels and pipelines.get_algo_from_config(config, key) in [
        ee_algo.ic_sample,
        ee_algo.rgb_ic_sample,
    ]:
      for postfix in ["", "_mask", "_timestamps"]:
        pixels[key + postfix] = np.array([])

  return pixels | ccdc_pixels | metadata
