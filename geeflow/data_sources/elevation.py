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

"""Elevation and DEM data sources."""

import dataclasses

from geeflow.data_sources import ee_base

ee = ee_base.ee


@dataclasses.dataclass
class NasaDem(ee_base.EeData):
  """NASADEM elevation based on reprocessed and improved SRTM."""

  # NOTE: Has no coverage for higher latitudes. Use FABDEM or CopDem instead.

  BANDS = ["elevation", "slope", "aspect"]

  @property
  def asset_name(self) -> str:
    return "NASA/NASADEM_HGT/001"

  @property
  def im(self):
    elevation = ee.Image(self.asset_name).select("elevation")
    # Elevation above geoid in meters, slope/aspect in deg [0..90], [0..360].
    return elevation.addBands(
        ee.Terrain.slope(elevation)
    ).addBands(  # pytype: disable=attribute-error
        ee.Terrain.aspect(elevation)
    )  # pytype: disable=attribute-error

  @property
  def ic(self):
    raise ValueError("This is an Image and not a Collection.")


@dataclasses.dataclass
class FABDEM(ee_base.EeData):
  """FABDEM v1-0 (Forest And Buildings removed Copernicus DEM)."""

  BANDS = ["elevation", "slope", "aspect"]

  @property
  def asset_name(self) -> str:
    return "projects/sat-io/open-datasets/FABDEM"

  @property
  def im(self):
    fabdem = ee.ImageCollection(self.asset_name)
    proj = fabdem.first().projection()
    elevation = fabdem.mosaic().setDefaultProjection(proj)
    slope = ee.Terrain.slope(elevation).setDefaultProjection(proj)  # pytype: disable=attribute-error
    aspect = ee.Terrain.aspect(elevation).setDefaultProjection(proj)  # pytype: disable=attribute-error

    res = ee.Image([elevation, slope, aspect])
    res = res.rename(["elevation", "slope", "aspect"])
    return res

  @property
  def ic(self):
    raise ValueError("This is an Image and not a Collection.")


@dataclasses.dataclass
class CopDem(ee_base.EeData):
  """Copernicus DEM (GLO-30) elevation based on TanDEM-X."""

  BANDS = ["elevation", "slope", "aspect"]

  @property
  def asset_name(self) -> str:
    return "COPERNICUS/DEM/GLO30"

  @property
  def im(self):
    # ImageCollection consists of spatially disjoint patches that we join for
    # a single global map, and return as an ee.Image.
    orig_ic = ee.ImageCollection(self.asset_name)
    orig_csr = orig_ic.first().projection()
    elevation = (
        orig_ic.mosaic()  # Results in 1 deg cells, and requires...
        .reproject(orig_csr)  # ...reprojection for slope/aspect.
        .select("DEM")
        .rename(["elevation"])
    )
    # Elevation above geoid in meters, slope/aspect in deg [0..90], [0..360].
    return elevation.addBands(
        ee.Terrain.slope(elevation)
    ).addBands(  # pytype: disable=attribute-error
        ee.Terrain.aspect(elevation)
    )  # pytype: disable=attribute-error

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")
