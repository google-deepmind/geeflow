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

"""MODIS GEE datasets."""

import dataclasses
from geeflow.data_sources import ee_base
import numpy as np

EeData = ee_base.EeData


@dataclasses.dataclass
class ModisTerraVeg(EeData):
  """MODIS Terra Vegetation Indices."""

  mode: str = "250m"

  BANDS = [
      "NDVI",
      "EVI",
      "DetailedQA",
      "sur_refl_b01",
      "sur_refl_b02",
      "sur_refl_b03",
      "sur_refl_b07",
      "ViewZenith",
      "SolarZenith",
      "RelativeAzimuth",
      "DayOfYear",
      "SummaryQA",
  ]
  VIS_BANDS = ["NDVI"]
  PALETTE = [
      "FFFFFF",
      "CE7E45",
      "DF923D",
      "F1B555",
      "FCD163",
      "99B718",
      "74A901",
      "66A000",
      "529400",
      "3E8601",
      "207401",
      "056201",
      "004C00",
      "023B01",
      "012E01",
      "011D01",
      "011301",
  ]

  @property
  def asset_name(self) -> str:
    if self.mode == "250m":
      return "MODIS/061/MOD13Q1"
    elif self.mode == "500m":
      return "MODIS/061/MOD13A1"
    elif self.mode == "1km":
      return "MODIS/061/MOD13A2"
    raise ValueError(f"Unsupported mode: {self.mode}")

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    v_min, v_max = 0, 9000
    img = img[..., 0]
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)


@dataclasses.dataclass
class ModisSurfRefl(EeData):
  """MODIS Terra Surface Reflectance."""

  mode: str = "8d-500m-7b"

  BANDS = [
      "sur_refl_b01",
      "sur_refl_b02",
      "sur_refl_b03",
      "sur_refl_b04",
      "sur_refl_b05",
      "sur_refl_b06",
      "sur_refl_b07",
      "SolarZenith",
      "ViewZenith",
      "RelativeAzimuth",
      "QA",
      "StateQA",
      "DayOfYear",
  ]
  VIS_BANDS = ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"]

  @property
  def asset_name(self) -> str:
    if self.mode == "8d-500m-7b":
      return "MODIS/061/MOD09A1"
    raise ValueError(f"Unsupported mode: {self.mode}")


@dataclasses.dataclass
class ModisGPP(EeData):
  """MODIS Terra Gross Primary Productivity."""

  mode: str = "8d-500mb"

  BANDS = ["Gpp", "PsnNet", "Psn_QC"]
  VIS_BANDS = ["Gpp"]

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MOD17A2H"


@dataclasses.dataclass
class ModisET(EeData):
  """MODIS Terra EvapoTranspiration."""

  mode: str = "8d-500m"

  BANDS = ["ET", "LE", "PET", "PLE", "ET_QC"]
  VIS_BANDS = ["ET"]

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MOD16A2"


@dataclasses.dataclass
class ModisBurn(EeData):
  """MODIS Burned Area."""

  mode: str = "m-500m"

  BANDS = ["BurnDate", "Uncertainty", "QA", "FirstDay", "LastDay"]
  VIS_BANDS = ["BurnDate"]

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MCD64A1"


@dataclasses.dataclass
class ModisFire(EeData):
  """MODIS Thermal Anomalies & Fire."""

  mode: str = "8-day"

  BANDS = ["MaxFRP"]

  @property
  def asset_name(self) -> str:
    if self.mode == "yearly":
      raise NotImplementedError(
          "Preaggregated yearly asset not publicly available."
      )
    postfix = ({"8-day": "2", "daily": "1"})[self.mode]
    return f"MODIS/061/MOD14A{postfix}"


@dataclasses.dataclass
class FIRMS(EeData):
  """FIRMS: Fire Information for Resource Management System."""

  mode: str = "daily"

  @property
  def asset_name(self) -> str:
    if self.mode == "yearly":
      raise NotImplementedError(
          "Preaggregated yearly asset not publicly available."
      )
    return "FIRMS"


@dataclasses.dataclass
class ModisLaiFpar(EeData):
  """MODIS LAI & FPAR."""

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MCD15A3H"
