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

"""Demographics GEE datasets."""

import dataclasses

from geeflow.data_sources import ee_base

EeData = ee_base.EeData


@dataclasses.dataclass
class FPP(EeData):
  """Forest proximate people (2019)."""

  BANDS = ["FPP_1km", "FPP_5km"]

  @property
  def asset_name(self) -> str:
    return "FAO/SOFO/1/FPP"

  @property
  def im(self):
    return self.ic.first()


@dataclasses.dataclass
class TPP(EeData):
  """Tree proximate people (2019)."""

  BANDS = ["TPP_1km", "TPP_1km_cropland", "TPP_500m", "TPP_500m_cropland"]

  @property
  def asset_name(self) -> str:
    return "FAO/SOFO/1/TPP"

  @property
  def im(self):
    return self.ic.first()


@dataclasses.dataclass
class CIESIN(EeData):
  """The estimated number of persons per 30 arc-second grid cell."""

  mode: str = "Count"

  @property
  def asset_name(self) -> str:
    return f"CIESIN/GPWv411/GPW_Population_{self.mode}"


@dataclasses.dataclass
class WorldPop(EeData):
  """The estimated number of persons per 100m grid cell."""

  @property
  def asset_name(self) -> str:
    return "WorldPop/GP/100m/pop"


@dataclasses.dataclass
class GHSPop(EeData):
  """The estimated number of persons per 100m grid cell."""

  @property
  def asset_name(self) -> str:
    return "JRC/GHSL/P2023A/GHS_POP"
