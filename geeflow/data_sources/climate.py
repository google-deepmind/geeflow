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

"""Climate (also weather, atmosphere) GEE sources."""

import dataclasses

from geeflow.data_sources import ee_base

EeData = ee_base.EeData


@dataclasses.dataclass
class Era5(EeData):
  """Era5."""

  mode: str = "monthly"
  BANDS = [
      "total_precipitation_sum",
      "total_precipitation_min",
      "total_precipitation_max",
      "temperature_2m",
      "temperature_2m_min",
      "temperature_2m_max",
      "dewpoint_temperature_2m",
      "dewpoint_temperature_2m_min",
      "dewpoint_temperature_2m_max",
      "surface_pressure",
      "surface_pressure_min",
      "surface_pressure_max",
  ]

  @property
  def asset_name(self):
    if self.mode == "monthly":
      return "ECMWF/ERA5_LAND/MONTHLY_AGGR"
    if self.mode == "daily":
      return "ECMWF/ERA5_LAND/DAILY_AGGR"
