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

"""Radar GEE datasets."""

from collections.abc import Sequence
import dataclasses

from geeflow.data_sources import ee_base
import numpy as np

import ee

EeData = ee_base.EeData


@dataclasses.dataclass
class Sentinel1(EeData):
  """Sentinel-1 datasets (GRD).

  - It is recommended not to combining ascending and descending data. Usually it
    is location dependent, but ideally one would filter out minority orbit.
  - Most of the Earth was imaged with VV+VH polarization.
  """

  mode: str = "IW"
  pols: Sequence[str] = ("VV", "VH")  # Polarization which should be included.
  orbit: str = "both"  # both/asc/desc

  BANDS = ["VV", "VH", "angle"]
  VIS_BANDS = ["VV", "VH", "VV"]  # Disregard angle from visualization.

  # S1_GRD collection is huge. Due to filterMetadata, generating dummy image
  # was the bottleneck, so we specify it explicitly.
  @property
  def dummy_image_id(self) -> str:
    assert self.orbit in ("asc", "desc", "both"), "Wrong orbit!"
    if self.orbit != "asc":
      return "S1A_IW_GRDH_1SDV_20141003T040550_20141003T040619_002660_002F64_EC04"  # pylint: disable=line-too-long
    return "S1A_IW_GRDH_1SDV_20141003T101018_20141003T101047_002664_002F75_DC99"  # pylint: disable=line-too-long

  @property
  def ic(self):
    assert self.orbit in ("asc", "desc", "both"), "Wrong orbit!"
    ic = (
        ee.ImageCollection(self.asset_name)
        .filterMetadata("instrumentMode", "equals", self.mode)
        .filterMetadata("transmitterReceiverPolarisation", "equals", self.pols)
    )
    if self.orbit == "asc":
      ic = ic.filterMetadata("orbitProperties_pass", "equals", "ASCENDING")
    if self.orbit == "desc":
      ic = ic.filterMetadata("orbitProperties_pass", "equals", "DESCENDING")
    return ic

  @property
  def asset_name(self) -> str:
    return "COPERNICUS/S1_GRD"

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    v_min, v_max = -25, 5
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)


@dataclasses.dataclass
class Alos(EeData):
  """ALOS PALSAR datasets."""

  mode: str = "yearly"
  pols: Sequence[str] = ("HH", "HV")  # Polarization which should be included.
  orbit: str = "both"  # both/asc/desc

  # For ALOS PALSAR ScanSar L2_2: ["HH", "HV", "LIN", "MSK"]
  BANDS = ["HH", "HV", "angle", "date", "qa"]
  VIS_BANDS = ["HH", "HV", "HH"]  # Disregard angle from visualization.

  @property
  def asset_name(self) -> str:
    if self.mode == "yearly":
      return "JAXA/ALOS/PALSAR/YEARLY/SAR"
    elif self.mode == "yearly_v2":
      return "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH"
    elif self.mode == "L2_2":
      return "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
    raise ValueError(f"Unsupported mode: {self.mode}")

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name)
    if self.mode == "L2_2":
      ic = ic.filterMetadata("Polarizations", "equals", self.pols)
      if self.orbit == "asc":
        ic = ic.filterMetadata("PassDirection", "equals", "Ascending")
      if self.orbit == "desc":
        ic = ic.filterMetadata("PassDirection", "equals", "Descending")
    return ic

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    v_min, v_max = 0.0, 10000.0
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)

  @classmethod
  def to_gamma0(cls, img: np.ndarray) -> np.ndarray:
    """Converts 16-bit digital numbers (DN) to gamma-zero backscatter in dB."""
    return 10 * np.log10(img.astype(float) ** 2) - 83.0
