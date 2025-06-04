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

"""Data extraction config for a global drivers at 1km model for 2022."""

import functools

from geeflow import ee_algo
from geeflow import ee_data
from geeflow import times
from geeflow import utils
from ml_collections import config_dict as cd


S2_BANDS = ("B3", "B4", "B5", "B7", "B12")


def get_sources_config(scale: int):
  """Returns sources config."""
  c = cd.ConfigDict()

  c.region_ee = utils.get_source_config("CustomFC", "fc")
  c.region_ee.kw.asset_name = (
      "projects/wri-datalab/Drivers_1km/regions_final_update_dissolve_20240701"
  )
  c.region_ee.select = ["gridcode"]
  c.region_ee.scalar = True
  c.region_ee.algo = ee_algo.fc_to_image
  c.region_ee.sampling_kw.reducer = "first"

  c.hansen = utils.get_source_config("Hansen", "im")
  c.hansen.scale = scale
  c.hansen.kw.mode = "2022_v1_10"

  c.tree_cover_loss_due_to_fire = utils.get_source_config(
      "TreeCoverLossDueToFire", "im")
  c.tree_cover_loss_due_to_fire.scale = scale

  c.mining_combined = utils.get_source_config("CustomFC", "fc")
  c.mining_combined.kw.asset_name = "users/radoststanimirova/mining_combined"
  c.mining_combined.scale = scale
  c.mining_combined.select = [ee_algo.FEATURE_EXISTS_INTEGER_KEY]
  c.mining_combined.algo = ee_algo.fc_to_image
  c.mining_combined.sampling_kw.reducer = "first"

  c.s2 = utils.get_source_config("Sentinel2", "filter_by_cloud_percentage")
  c.s2.scale = scale
  c.s2.select = S2_BANDS
  c.s2.kw.mode = "L1C"
  c.s2.out_kw.percentage = 50
  c.s2.sampling_kw.reduce_fn = "mean"
  c.s2.sampling_kw.cloud_mask_fn = ee_data.Sentinel2.im_cloud_score_plus_mask
  c.s2.date_ranges_fn = functools.partial(
      times.adjust_for_hemisphere,
      north=[(f"{y}-06-01", 3, 0) for y in [2018, 2020, 2022]],
      south=[(f"{y-1}-12-01", 3, 0) for y in [2018, 2020, 2022]])

  c.dw = utils.get_source_config("DynamicWorld", "ic")
  c.dw.scale = scale
  c.dw.select = ("grass", "crops", "shrub_and_scrub")
  c.dw.sampling_kw.reduce_fn = "mean"
  c.dw.date_ranges = [(f"{y}-01-01", 12, 0) for y in [2016, 2019, 2022]]

  c.ghs_pop = utils.get_source_config("GHSPop", "ic")
  c.ghs_pop.scale = scale
  c.ghs_pop.select = "population_count"

  c.elevation = utils.get_source_config("FABDEM", "im")
  c.elevation.scale = scale
  c.elevation.select = ("elevation", "slope", "aspect")

  return c


def get_labels_config(labels_path: str, scale_factor: float, scale: int,
                      for_inference: bool):
  """Returns labels config."""
  c = cd.ConfigDict()
  c.default_scale = scale
  c.use_utm = False
  c.img_width_deg = 0.01 * scale_factor
  c.img_width_m = int(1111 * scale_factor)  # Image width in meters.

  c.path = labels_path
  c.meta_keys = None
  if not for_inference:
    c.meta_keys = ("lat", "lon", "label_7", "split", "label_name",
                   "confidence", "hansen_year", "split_5",
                   "hansen_year_deforestation_ratio", "original_id", "region",
                   )
  return c


def get_config(arg=None):
  """Returns config."""
  # scale_factor is used to provide larger context. For example 1 means no
  # additional context (1km plots). 2 means each plot will be 2km in size.
  arg = utils.parse_arg(arg, labels_path="UPDATE", scale_factor=1.5, scale=15,
                        for_inference=False)
  config = cd.ConfigDict()
  config.sources = get_sources_config(scale=arg.scale)
  config.labels = get_labels_config(
      labels_path=arg.labels_path,
      scale_factor=arg.scale_factor,
      scale=arg.scale,
      for_inference=arg.for_inference,
  )
  return config
