# Copyright 2024 DeepMind Technologies Limited.
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

import time
from absl import logging
import apache_beam as beam
from geeflow import pipelines
from geeflow import utils
import ml_collections

import ee

LONGITUDE_STEP = 6
LATITUDE_STEP = 8


def get_utm_zones():
  """Returns a list of utm zones."""
  zones = []
  for lat in range(-72, 80, LATITUDE_STEP):
    for lon in range(-180, 180, LONGITUDE_STEP):
      start_lon = lon
      start_lat = lat
      lon_step = LONGITUDE_STEP
      lat_step = LATITUDE_STEP
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

  def __init__(self,
               ee_project,
               config: ml_collections.ConfigDict,
               num_calls_counter=None, num_failures_counter=None,
               getinfo_duration_distribution=None):
    self.ee_project = ee_project
    self.config = config
    self.num_calls_counter = num_calls_counter
    self.num_failures_counter = num_failures_counter
    self.getinfo_duration_distribution = getinfo_duration_distribution

  def start_bundle(self):
    # Initialize and authenticate EE on the flume workers.
    utils.initialize_ee(self.ee_project)
    sources = pipelines.pipeline_sources(self.config)
    self.process_fn = pipelines.get_roi_sample_fn(self.config, sources)
    self.process_fn_dict = pipelines.get_roi_sample_dict_fn(self.config,
                                                            sources)

  def process(self, items):
    if self.num_calls_counter:
      self.num_calls_counter.inc()
    rois = [pipelines.pipeline_item_to_rois(self.config, item)
            for item in items]
    try:
      start = time.time()
      fc = ee.FeatureCollection([self.process_fn(roi, item)
                                 for roi, item in zip(rois, items)])
      fc_data = fc.getInfo()
      if time.time() - start > 20:
        logging.info("Long getinfo call: %s for %s", time.time() - start, items)
      if self.getinfo_duration_distribution:
        self.getinfo_duration_distribution.update(time.time() - start)
    except ee.EEException as e:
      if self.num_failures_counter:
        self.num_failures_counter.inc()
      logging.info("EEException occurred: %s for %s", e, items)
      if ("User memory limit exceeded." in str(e) or
          "Response size exceeds limit" in str(e)):
        # Try requesting properties separately for a single feature to avoid OOM
        # If there are multiple ee_feature_group_size should have been reduced
        # as a first countermeasure.
        if len(items) == 1:
          data = {}
          for a, b in self.process_fn_dict(rois[0], items[0]).items():
            f = ee.Feature(rois[0], {a: b})
            tmp_data = f.getInfo()
            data.update(tmp_data["properties"])
          yield data
          return
      raise e
    for feature in fc_data["features"]:
      yield feature["properties"]
