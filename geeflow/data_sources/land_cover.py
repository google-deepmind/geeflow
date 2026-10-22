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

"""Land cover GEE sources.

Also currently used for layers related to water, soil, agriculture, nightlights.
"""

import dataclasses

from geeflow.data_sources import ee_base
import matplotlib.colors as mcol
import matplotlib.pylab as plt
import numpy as np

import ee

EeData = ee_base.EeData


@dataclasses.dataclass
class LandCover(EeData):
  """Global Consensus Landcover."""

  # Possible values: barren, cultivated_and_managed_vegetation,
  # deciduous_broadleaf_trees, evergreen-deciduous_needleleaf_trees,
  # evergreen_broadleaf_trees, herbaceous_vegetation, mixed-other_trees,
  # open_water, regularly_flooded_vegetation, shrubs, snow-ice, urban-built-up.
  mode: str = "barren"

  @property
  def asset_name(self) -> str:
    return (
        f"projects/sat-io/open-datasets/global_consensus_landcover/{self.mode}"
    )


@dataclasses.dataclass
class WSF2015(EeData):
  """World Settlement Footprint 2015."""

  @property
  def asset_name(self) -> str:
    return "DLR/WSF/WSF2015/v1"


@dataclasses.dataclass
class TreeCoverLossDueToFire(EeData):
  """Tree cover loss due to fire, Tyukavina et al 2022 (30m)."""

  @property
  def asset_name(self) -> str:
    return "users/sashatyu/2001-2022_fire_forest_loss_annual"

  @property
  def im(self):
    return ee.ImageCollection(self.asset_name).mosaic()

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")


@dataclasses.dataclass
class Hansen(EeData):
  """Hansen Global Forest Change v1.10 (2000-2022)."""

  mode: str = "2022_v1_10"

  BANDS = [
      "treecover2000",
      "loss",
      "gain",
      "first_b30",
      "first_b40",
      "first_b50",
      "first_b70",
      "last_b30",
      "last_b40",
      "last_b50",
      "last_b70",
      "datamask",
      "lossyear",
  ]

  @property
  def asset_name(self) -> str:
    if self.mode == "2025":
      return "projects/glad/GFC/2025/global_forest_change_2025_v1_13_merged"
    if self.mode == "2024":
      return "projects/glad/GFC/2024/global_forest_change_2024_v1_12_merged"
    return "UMD/hansen/global_forest_change_" + self.mode


@dataclasses.dataclass
class RaddAlert(EeData):
  """WUR RADD Alert images (2017-present).

  Website: http://radd-alert.wur.nl
  Data inspection:
  https://code.earthengine.google.com/a70afebdd813722b9f676d9d6c91d656
  """

  BANDS = ("Alert", "Date")

  asset_name: str = "projects/radar-wur/raddalert/v1"

  @property
  def ic(self):
    """Keeps only images carrying the `Alert` band (drops metadata tiles)."""
    ic = ee.ImageCollection(self.asset_name)
    return ic.filter(ee.Filter.listContains("system:band_names", "Alert"))

  @property
  def im(self):
    raise ValueError("This is an ImageCollection and not an Image.")


@dataclasses.dataclass
class QiuDisturbance(EeData):
  """Qiu's US CONUS disturbance data (1988-2022)."""

  ref_year: int = 2020

  BANDS = [
      "logging_events_count",
      "avg_rotation_time",
      "rotations_count",
      "forest_age_after_logging",
      "forest_age_after_disturbance",
      "last_disturbance_driver",
  ]

  @property
  def asset_name(self) -> str:
    return "users/ShiQiu/product/conus/disturbance/v081/APRI"

  @property
  def im(self):
    start_year, num_bands = 1985, 38
    assert start_year < self.ref_year < start_year + num_bands

    years = ee.List([ee.String(str(start_year + i)) for i in range(num_bands)])
    disturbances = ee.ImageCollection(self.asset_name).mosaic().rename(years)

    logging = disturbances.eq(1)
    logging_count = logging.reduce(ee.Reducer.sum())

    indices = ee.Image(ee.Array(ee.List.sequence(0, num_bands - 1))).arrayMask(
        logging.toArray()
    )
    unmasked_indices = indices.unmask(
        ee.Image(ee.Array(ee.List.repeat(0, num_bands)))
    )
    sorted_indices = unmasked_indices.arraySort()
    indices_array = sorted_indices.arrayMask(sorted_indices.lt(num_bands))
    shifted = indices_array.arraySlice(0, 1)
    original = indices_array.arraySlice(0, 0, -1)
    intervals = shifted.subtract(original).subtract(1)
    avg_interval = intervals.arrayReduce(ee.Reducer.mean(), [0]).arrayGet(0)
    num_intervals = intervals.arrayReduce(ee.Reducer.count(), [0]).arrayGet(0)

    selected_years = years.slice(0, self.ref_year - start_year + 1)

    def get_change(band_name):
      change_this_year = disturbances.select([band_name])
      masked_change = change_this_year.updateMask(change_this_year.gt(0))
      return masked_change.rename("last_change")

    latest_driver = ee.ImageCollection(selected_years.map(get_change)).mosaic()

    def get_year_of_last_change(band_name):
      change_this_year = disturbances.select([band_name])
      mask = change_this_year.gt(0)
      year_value = ee.Number.parse(band_name).int16()
      year_image = ee.Image.constant(year_value).int16()
      return year_image.updateMask(mask).rename("year_of_last_change")

    last_change_year = ee.ImageCollection(
        selected_years.map(get_year_of_last_change)
    ).mosaic()
    years_since_last_disturbance = ee.Image.constant(self.ref_year).subtract(
        last_change_year
    )

    def get_year_of_last_logging(band_name):
      year_value = ee.Number.parse(band_name).int16()
      year_image = ee.Image.constant(year_value).int16()
      mask = logging.select([band_name])
      return year_image.updateMask(mask).rename("year_of_last_change")

    last_logging_year = ee.ImageCollection(
        selected_years.map(get_year_of_last_logging)
    ).mosaic()
    years_since_last_logging = ee.Image.constant(self.ref_year).subtract(
        last_logging_year
    )

    return ee.Image.cat([
        logging_count.selfMask(),
        avg_interval.updateMask(logging_count.gte(2)),
        num_intervals.updateMask(logging_count.gte(2)),
        years_since_last_logging,
        years_since_last_disturbance,
        latest_driver,
    ]).rename(self.BANDS)


@dataclasses.dataclass
class Primary(EeData):
  """Primary humid tropical forest (Image) for 2001."""

  mode: str = "2001"

  BANDS = ["Primary_HT_forests"]

  @property
  def asset_name(self) -> str:
    if self.mode == "2001":
      return "UMD/GLAD/PRIMARY_HUMID_TROPICAL_FORESTS/v1"
    elif self.mode == "2020":
      raise NotImplementedError("2020 asset not publicly available.")
    else:
      raise ValueError(f"Unsupported mode: {self.mode}")

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")

  @property
  def im(self):
    if self.mode == "2001":
      return ee.ImageCollection(self.asset_name).first()
    return ee.Image(self.asset_name)


@dataclasses.dataclass
class CocoaAbu(EeData):
  """Cocoa map by Abu."""

  mode: str = "gha"

  BANDS = ["b1"]

  @property
  def asset_name(self) -> str:
    raise NotImplementedError("Asset not publicly available.")

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")


@dataclasses.dataclass
class NLCD(EeData):
  """USGS National Land Cover Database."""

  mode: str = "2019"

  BANDS = ["landcover", "impervious", "impervious_descriptor"]

  @property
  def asset_name(self) -> str:
    return f"USGS/NLCD_RELEASES/{self.mode}_REL/NLCD"

  def get_year_im(self, year):
    return self.ic.filter(f"system:index == {year}").first()


@dataclasses.dataclass
class DynamicWorld(EeData):
  """DynamicWorld ImageCollection."""

  mode: str = "V1"

  BANDS = [
      "water",
      "trees",
      "grass",
      "flooded_vegetation",
      "crops",
      "shrub_and_scrub",
      "built",
      "bare",
      "snow_and_ice",
      "label",
  ]
  VIS_BANDS = ["label"]
  PALETTE = [
      "419BDF",
      "397D49",
      "88B053",
      "7A87C6",
      "E49635",
      "DFC35A",
      "C4281B",
      "A59B8F",
      "B39FE1",
  ]
  PALETTE_11 = [
      "000000",
      "419BDF",
      "397D49",
      "88B053",
      "7A87C6",
      "E49635",
      "DFC35A",
      "C4281B",
      "A59B8F",
      "B39FE1",
      "FFFFFF",
  ]

  @property
  def asset_name(self) -> str:
    return f"GOOGLE/DYNAMICWORLD/{self.mode}"

  @classmethod
  def cmap(cls):
    return mcol.ListedColormap([f"#{c}" for c in cls.PALETTE])

  @classmethod
  def cmap_names(cls) -> tuple[str, ...]:
    return tuple(cls.BANDS[:-1])

  @classmethod
  def imshow(cls, *args, **kwargs):
    plt.imshow(
        *args,
        vmin=0,
        vmax=9,
        cmap=cls.cmap(),
        interpolation="nearest",
        **kwargs,
    )

  @classmethod
  def colorbar(cls, *args, **kwargs):
    cbar = plt.colorbar(
        *args, ticks=np.arange(len(cls.PALETTE)) + 0.5, **kwargs
    )
    cbar.ax.set_yticklabels(f"{i}: {c}" for i, c in enumerate(cls.BANDS[:-1]))


@dataclasses.dataclass
class WorldCover(EeData):
  """ESA WorldCover (Image) for 2020 and 2021."""

  mode: str = "2020"

  BANDS = ["Map"]

  @property
  def asset_name(self) -> str:
    modes = {"2020": "v100", "2021": "v200"}
    return f"ESA/WorldCover/{modes[self.mode]}"

  @property
  def im(self):
    return self.ic.first()


@dataclasses.dataclass
class CCDC(EeData):
  """Google Global Landsat-based CCDC Segments (1999-2019)."""

  mode: str = "V1"
  BANDS = [
      "tStart",
      "tEnd",
      "tBreak",
      "numObs",
      "changeProb",
      "BLUE_coefs",
      "GREEN_coefs",
      "RED_coefs",
      "NIR_coefs",
      "SWIR1_coefs",
      "SWIR2_coefs",
      "BLUE_rmse",
      "GREEN_rmse",
      "RED_rmse",
      "NIR_rmse",
      "SWIR1_rmse",
      "SWIR2_rmse",
      "BLUE_magnitude",
      "GREEN_magnitude",
      "RED_magnitude",
      "NIR_magnitude",
      "SWIR1_magnitude",
      "SWIR2_magnitude",
  ]

  @property
  def asset_name(self) -> list[str]:  # pyrefly: ignore[bad-override]
    if self.mode == "V1":
      return ["GOOGLE/GLOBAL_CCDC/V1"]
    elif self.mode == "V2":
      return ["projects/CCDC/measures/v1_overlap", "projects/CCDC/measures/v1"]
    elif self.mode == "V4":
      return ["projects/CCDC/v4"]
    else:
      raise ValueError(f"Unsupported mode: {self.mode}")

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name[0])
    for asset_name in self.asset_name[1:]:
      ic = ic.merge(ee.ImageCollection(asset_name))
    return ic
