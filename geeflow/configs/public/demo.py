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

"""Demo data extraction config.

See geeflow/README.md for example command.
"""

from geeflow import ee_data
from geeflow import times
from geeflow import utils
from ml_collections import config_dict as cd


# 2 annual time ranges, starting at 2018.
FC_DATE_RANGES = times.get_date_ranges("2018-01-01", 2, 12)


def get_sources_config():
  """Returns sources config."""
  c = cd.ConfigDict()

  c.s2 = utils.get_source_config("Sentinel2", "filter_by_cloud_percentage")
  c.s2.scale = 10
  c.s2.select = ["B3", "B2", "B1"]
  c.s2.kw.mode = "L2A"
  c.s2.out_kw.percentage = 50
  c.s2.sampling_kw.reduce_fn = "median"
  c.s2.sampling_kw.cloud_mask_fn = ee_data.Sentinel2.im_cloud_score_plus_mask
  c.s2.date_ranges = FC_DATE_RANGES

  c.s1 = utils.get_source_config("Sentinel1", "ic")
  c.s1.scale = 10
  c.s1.kw = {"mode": "IW", "pols": ("VV", "VH"), "orbit": "both"}
  c.s1.sampling_kw.reduce_fn = "mean"
  c.s1.date_ranges = FC_DATE_RANGES

  c.elevation = utils.get_source_config("NasaDem", "im")
  c.elevation.scale = 30
  c.elevation.select = ("elevation", "slope", "aspect")
  return c


def get_labels_config(labels_path: str, num_examples: int):
  """Returns labels config."""
  c = cd.ConfigDict()
  c.img_width_m = 240  # Image width in meters.
  c.max_cell_size_m = 30  # Max pixel size in meters.
  c.path = labels_path
  c.meta_keys = ("lat", "lon", "split")
  c.num_max_samples = num_examples
  return c


def get_config(arg=None):
  arg = utils.parse_arg(arg, labels_path="data/demo_labels.csv", num=10)
  config = cd.ConfigDict()
  config.sources = get_sources_config()
  config.labels = get_labels_config(arg.labels_path, num_examples=arg.num)
  return config
