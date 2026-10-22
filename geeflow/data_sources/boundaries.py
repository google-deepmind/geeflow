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

"""Boundaries and regions GEE FeatureCollections."""

import dataclasses

from geeflow.data_sources import ee_base

import ee

EeDataFC = ee_base.EeDataFC


@dataclasses.dataclass
class Ecoregions(EeDataFC):
  """Ecoregions FC for 2017."""

  @property
  def asset_name(self) -> str:
    return "RESOLVE/ECOREGIONS/2017"


@dataclasses.dataclass
class Countries(EeDataFC):
  """Country boundaries FC for 2017."""

  mode: str = "simple"
  BANDS = ["abbreviati", "country_co", "country_na", "wld_rgn"]

  @property
  def asset_name(self) -> str:
    sel = "_SIMPLE" if self.mode == "simple" else ""
    return f"USDOS/LSIB{sel}/2017"

  def filter_by_name(self, country_names):
    if isinstance(country_names, str):
      country_names = [country_names]
    country_names = [x.capitalize() for x in country_names]
    return self.fc.filter(ee.Filter.inList("country_na", country_names))


@dataclasses.dataclass
class RPGFrance2023(EeDataFC):
  """French RPG crop parcel boundaries."""

  @property
  def asset_name(self):
    raise NotImplementedError("Asset not publicly available.")

  @property
  def fc(self):
    raise NotImplementedError("Asset not publicly available.")
