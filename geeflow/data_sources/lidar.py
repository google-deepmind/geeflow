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

"""Lidar-based GEE datasets."""

import dataclasses

from geeflow.data_sources import ee_base

import ee

EeData = ee_base.EeData


@dataclasses.dataclass
class GediRasterCanopyHeight(EeData):
  """GEDI rasterized canopy top height (L2A)."""

  filter_quality: bool = True

  BANDS = [
      "digital_elevation_model",
      "landsat_treecover",
      "landsat_water_persistence",
      "modis_treecover",
      "modis_nonvegetated",
      "urban_proportion",
  ] + [f"rh{p}" for p in range(101)]

  @property
  def asset_name(self) -> str:
    return "LARSE/GEDI/GEDI02_A_002_MONTHLY"

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name)
    if self.filter_quality:

      def quality_mask_fn(im) -> ee.Image:
        return im.updateMask(im.select("quality_flag").eq(1)).updateMask(
            im.select("degrade_flag").eq(0)
        )

      ic = ic.map(quality_mask_fn)
    return ic


@dataclasses.dataclass
class GediRasterCanopyStructure(EeData):
  """GEDI rasterized canopy cover vertical profile metrics (L2B)."""

  filter_quality: bool = True

  BANDS = (
      ["pai", "cover", "fhd_normal"]
      + [f"cover_z{p}" for p in range(31)]
      + [f"pai_z{p}" for p in range(31)]
      + [f"pavd_z{p}" for p in range(31)]
  )

  @property
  def asset_name(self) -> str:
    return "LARSE/GEDI/GEDI02_B_002_MONTHLY"

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name)
    if self.filter_quality:

      def quality_mask_fn(im) -> ee.Image:
        return im.updateMask(im.select("l2b_quality_flag").eq(1)).updateMask(
            im.select("degrade_flag").eq(0)
        )

      ic = ic.map(quality_mask_fn)
    return ic


@dataclasses.dataclass
class GediRasterAGB(EeData):
  """GEDI rasterized aboveground biomass density (L4A)."""

  filter_quality: bool = True

  BANDS = ["agbd", "elev_lowestmode"]

  @property
  def asset_name(self) -> str:
    return "LARSE/GEDI/GEDI04_A_002_MONTHLY"

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name)
    if self.filter_quality:

      def quality_mask_fn(im) -> ee.Image:
        return im.updateMask(im.select("l4_quality_flag").eq(1)).updateMask(
            im.select("degrade_flag").eq(0)
        )

      ic = ic.map(quality_mask_fn)
    return ic
