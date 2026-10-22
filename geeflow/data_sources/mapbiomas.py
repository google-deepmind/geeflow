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

"""MapBiomas data sources for Earth Engine extraction.

This module provides GeeFlow EeData classes and helper functions for MapBiomas
data collections, including:

1. MapBiomas Brazil Landsat Surface Reflectance Mosaics:
   - 30m annual Landsat 5/7/8/9 surface reflectance mosaics (1985-2024).
   - Median spectral bands (blue, green, red, nir, swir1, swir2).
   - Asset ID: `projects/nexgenmap/MapBiomas2/LANDSAT/BRAZIL/mosaics-2`

2. MapBiomas Atmosfera Atlas (Collection 1):
   - Annual climate/atmospheric variables downscaled across Brazil (1985-2024).
   - Minimum, mean, and maximum air temperature (°C).
   - Annual total precipitation (mm/year).
   - Annual mean land surface temperature / LST (°C).
   - Annual mean vapor pressure deficit / VPD (kPa).
   - Annual number of rain-free days (days/year, 1998-2024).
   - Asset folder: `projects/mapbiomas-public/assets/brazil/atmosphere/`

3. MapBiomas Solo Atlas (Collection 1 / 3):
   - Soil physical/chemical properties downscaled to 30m (depth layers).
   - Soil organic carbon stock (t/ha, 1985-2024 annual time series).
   - Granulometry fractions: Clay, Sand, Silt content (10 depth slices).
   - Soil stoniness: volumetric rock fragments.
   - Asset folder: `projects/mapbiomas-public/assets/brazil/soil/collection3/`

4. MapBiomas LULC Brazil (Collection 9/10):
   - Annual 30m pixel-level land use and land cover classification (1985-2024).
   - Asset ID: `projects/mapbiomas-public/assets/brazil/lulc/v1`
"""

import dataclasses
from typing import Any

from geeflow.data_sources import ee_base

import ee


def add_system_time_from_year(ic: ee.ImageCollection) -> ee.ImageCollection:
  """Sets 'system:time_start' on each image from its 'year' property."""

  def _add_time(img: ee.Image) -> ee.Image:
    year = ee.Number(img.get("year"))
    return ee.Image(
        img.set("system:time_start", ee.Date.fromYMD(year, 1, 1).millis())
    )

  return ic.map(_add_time)


def mapbiomas_mosaic_ic(asset_name: str) -> ee.ImageCollection:
  """Loads MapBiomas Landsat mosaics and sets system:time_start from year."""
  return add_system_time_from_year(ee.ImageCollection(asset_name))


def annual_multiband_to_ic(
    asset_name: str,
    band_prefix: str,
    out_band: str,
    start_year: int = 1985,
    end_year: int = 2024,
) -> ee.ImageCollection:
  """Converts a single multi-annual Image into a time-series ImageCollection."""
  img = ee.Image(asset_name)
  s_yr = max(start_year, 1985)
  e_yr = min(end_year, 2024)
  if s_yr > e_yr:
    return ee.ImageCollection([])

  def _make_year_img(y):
    y = ee.Number(y)
    bname = ee.String(band_prefix).cat(y.format("%d"))
    date = ee.Date.fromYMD(y, 1, 1)
    return (
        img.select([bname], [out_band])
        .set("system:time_start", date.millis())
        .set("year", y)
    )

  return ee.ImageCollection(ee.List.sequence(s_yr, e_yr).map(_make_year_img))


# =============================================================================
# 1. Landsat Mosaics
# =============================================================================


@dataclasses.dataclass
class MapBiomasLandsat(ee_base.EeData):
  """MapBiomas Brazil Landsat Surface Reflectance Mosaics."""

  DEFAULT_BANDS = (
      "blue_median", "green_median", "red_median", "nir_median", "swir1_median",
      "swir2_median")  # fmt: skip
  BANDS = (
      "blue_median", "blue_median_wet", "blue_median_dry", "blue_min",
      "blue_stdDev",
      "green_median", "green_median_dry", "green_median_wet",
      "green_median_texture", "green_min", "green_stdDev",
      "red_median", "red_median_dry", "red_min", "red_median_wet", "red_stdDev",
      "nir_median", "nir_median_dry", "nir_median_wet", "nir_min", "nir_stdDev",
      "swir1_median", "swir1_median_dry", "swir1_median_wet", "swir1_min",
      "swir1_stdDev",
      "swir2_median", "swir2_median_wet", "swir2_median_dry", "swir2_min",
      "swir2_stdDev",
      "ndvi_median_dry", "ndvi_median_wet", "ndvi_median", "ndvi_amp",
      "ndvi_stdDev",
      "ndwi_median", "ndwi_median_dry", "ndwi_median_wet", "ndwi_amp",
      "ndwi_stdDev",
      "evi2_median", "evi2_median_dry", "evi2_median_wet", "evi2_amp",
      "evi2_stdDev",
      "savi_median_dry", "savi_median_wet", "savi_median", "savi_stdDev",
      "pri_median_dry", "pri_median", "pri_median_wet",
      "gcvi_median", "gcvi_median_dry", "gcvi_median_wet", "gcvi_stdDev",
      "hallcover_median", "hallcover_stdDev",
      "cai_median", "cai_median_dry", "cai_stdDev",
      "gv_median", "gv_median_dry", "gv_median_wet", "gv_max", "gv_min",
      "gv_amp", "gv_stdDev",
      "gvs_median", "gvs_median_dry", "gvs_median_wet", "gvs_max", "gvs_min",
      "gvs_amp", "gvs_stdDev",
      "npv_median", "npv_median_dry", "npv_median_wet", "npv_max", "npv_min",
      "npv_amp", "npv_stdDev",
      "soil_median", "soil_median_dry", "soil_median_wet", "soil_max",
      "soil_min", "soil_amp", "soil_stdDev",
      "cloud_median", "cloud_median_dry", "cloud_median_wet", "cloud_max",
      "cloud_min", "cloud_amp", "cloud_stdDev",
      "shade_median", "shade_median_dry", "shade_median_wet", "shade_max",
      "shade_min", "shade_amp", "shade_stdDev",
      "ndfi_median", "ndfi_median_dry", "ndfi_median_wet", "ndfi_max",
      "ndfi_min", "ndfi_amp", "ndfi_stdDev",
      "sefi_median", "sefi_stdDev", "sefi_median_dry",
      "wefi_median", "wefi_median_wet", "wefi_amp", "wefi_stdDev",
      "slope")  # fmt: skip
  VIS_BANDS = ("red_median", "green_median", "blue_median")

  asset_name: str = "projects/nexgenmap/MapBiomas2/LANDSAT/BRAZIL/mosaics-2"

  @property
  def ic(self) -> ee.ImageCollection:
    return mapbiomas_mosaic_ic(self.asset_name)

  @property
  def im(self) -> ee.Image:
    raise ValueError("This is an ImageCollection, use .ic instead.")


# =============================================================================
# 2. Atmosfera Atlas (Collection 1)
# =============================================================================

MAPBIOMAS_ATMOSPHERE_ASSETS = {
    "air_temperature_min": (
        (
            "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/"
            "mapbiomas_brazil_collection1_air_temperature_annual_v2/"
            "mapbiomas_brazil_collection1_air_temperature_min_annual_v2"
        ),
        "temperature_",
    ),
    "air_temperature_mean": (
        (
            "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/"
            "mapbiomas_brazil_collection1_air_temperature_annual_v2/"
            "mapbiomas_brazil_collection1_air_temperature_mean_annual_v2"
        ),
        "temperature_",
    ),
    "air_temperature_max": (
        (
            "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/"
            "mapbiomas_brazil_collection1_air_temperature_annual_v2/"
            "mapbiomas_brazil_collection1_air_temperature_max_annual_v2"
        ),
        "temperature_",
    ),
    "precipitation": (
        (
            "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/"
            "mapbiomas_brazil_collection1_precipitation_annual_v2"
        ),
        "precipitation_",
    ),
    "surface_temperature": (
        (
            "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/"
            "mapbiomas_brazil_collection1_surface_temperature_annual_v2"
        ),
        "surface_temperature_",
    ),
    "vpd": (
        (
            "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/"
            "mapbiomas_brazil_collection1_vapor_pressure_deficit_annual_v2"
        ),
        "vapor_pressure_deficit_",
    ),
    "rain_free_days": (
        (
            "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/"
            "mapbiomas_brazil_collection1_rain_free_days_annual_v2"
        ),
        "rain_free_days_",
    ),
}


def mapbiomas_atmosphere_ic(
    start_year: int = 1985,
    end_year: int = 2024,
    include_rain_free_days: bool = True,
) -> ee.ImageCollection:
  """Constructs combined multi-band ImageCollection for MapBiomas Atmosfera."""
  tmin_img = ee.Image(MAPBIOMAS_ATMOSPHERE_ASSETS["air_temperature_min"][0])
  tmean_img = ee.Image(MAPBIOMAS_ATMOSPHERE_ASSETS["air_temperature_mean"][0])
  tmax_img = ee.Image(MAPBIOMAS_ATMOSPHERE_ASSETS["air_temperature_max"][0])
  precip_img = ee.Image(MAPBIOMAS_ATMOSPHERE_ASSETS["precipitation"][0])
  stemp_img = ee.Image(MAPBIOMAS_ATMOSPHERE_ASSETS["surface_temperature"][0])
  vpd_img = ee.Image(MAPBIOMAS_ATMOSPHERE_ASSETS["vpd"][0])
  if include_rain_free_days:
    rfd_img = ee.Image(MAPBIOMAS_ATMOSPHERE_ASSETS["rain_free_days"][0])

  def _make_year_img(y: Any) -> ee.Image:
    y = ee.Number(y)
    ystr = y.format("%d")
    date = ee.Date.fromYMD(y, 1, 1)

    img = (
        tmin_img.select(
            [ee.String("temperature_").cat(ystr)], ["air_temperature_min"]
        )
        .addBands(
            tmean_img.select(
                [ee.String("temperature_").cat(ystr)], ["air_temperature_mean"]
            )
        )
        .addBands(
            tmax_img.select(
                [ee.String("temperature_").cat(ystr)], ["air_temperature_max"]
            )
        )
        .addBands(
            precip_img.select(
                [ee.String("precipitation_").cat(ystr)], ["precipitation"]
            )
        )
        .addBands(
            stemp_img.select(
                [ee.String("surface_temperature_").cat(ystr)],
                ["surface_temperature"],
            )
        )
        .addBands(
            vpd_img.select(
                [ee.String("vapor_pressure_deficit_").cat(ystr)], ["vpd"]
            )
        )
    )

    if include_rain_free_days:
      # rain_free_days: 1998-2024; masked out for earlier years.
      rfd_band_name = ee.Algorithms.If(
          y.gte(1998),
          ee.String("rain_free_days_").cat(ystr),
          "rain_free_days_1998",
      )
      rfd_band = rfd_img.select([rfd_band_name], ["rain_free_days"])
      rfd_band = ee.Image(
          ee.Algorithms.If(y.gte(1998), rfd_band, rfd_band.updateMask(0))
      )
      img = img.addBands(rfd_band)

    return img.set("system:time_start", date.millis()).set("year", y)

  return ee.ImageCollection(
      ee.List.sequence(start_year, end_year).map(_make_year_img)
  )


@dataclasses.dataclass
class MapBiomasAtmosphere(ee_base.EeData):
  """MapBiomas Atmosfera Collection 1 - Combined Annual Atmospheric Variables.

  Combines 7 annual atmospheric layers (1000m resolution) into a single
  ImageCollection:
    - air_temperature_min: Annual minimum air temp 2m (°C, ERA5-Land/CHIRTS)
    - air_temperature_mean: Annual mean air temp 2m (°C, ERA5-Land/CHIRTS)
    - air_temperature_max: Annual maximum air temp 2m (°C, ERA5-Land/CHIRTS)
    - precipitation: Annual total precipitation (mm/yr, CHIRPS)
    - surface_temperature: Annual land surface temp (°C, MODIS/Landsat thermal)
    - vpd: Annual vapor pressure deficit (kPa)
    - rain_free_days: Annual rain-free days (TRMM/GPM >= 1998; masked < 1998)
  """

  start_year: int = 1985
  end_year: int = 2024
  include_rain_free_days: bool = True
  asset_name: str = (
      "projects/mapbiomas-public/assets/brazil/atmosphere/collection1"
  )

  DEFAULT_BANDS = (
      "air_temperature_min",
      "air_temperature_mean",
      "air_temperature_max",
      "precipitation",
      "surface_temperature",
      "vpd",
      "rain_free_days",
  )
  BANDS = DEFAULT_BANDS

  @property
  def ic(self) -> ee.ImageCollection:
    return mapbiomas_atmosphere_ic(
        start_year=self.start_year,
        end_year=self.end_year,
        include_rain_free_days=self.include_rain_free_days,
    )

  @property
  def im(self) -> ee.Image:
    raise ValueError("This is an annual series, use .ic instead.")


# =============================================================================
# 3. Soil Atlas (Collection 1 / 3)
# =============================================================================


@dataclasses.dataclass
class MapBiomasSoil(ee_base.EeData):
  """MapBiomas Soil Collection 3 - Physical Soil Properties (32 bands, 30m).

  Combines 4 static depth-sliced soil assets into a single static Image:
    - clay fraction: 10 depth intervals (0-100 cm)
    - sand fraction: 10 depth intervals (0-100 cm)
    - silt fraction: 10 depth intervals (0-100 cm)
    - stoniness: 2 depth thresholds (50 cm, 90 cm)
  """

  DEFAULT_BANDS = (
      # Clay fraction (10 depth intervals: 0-100 cm)
      "clay_fraction_000_010_v1_clay_fraction",
      "clay_fraction_010_020_v1_clay_fraction",
      "clay_fraction_020_030_v1_clay_fraction",
      "clay_fraction_030_040_v1_clay_fraction",
      "clay_fraction_040_050_v1_clay_fraction",
      "clay_fraction_050_060_v1_clay_fraction",
      "clay_fraction_060_070_v1_clay_fraction",
      "clay_fraction_070_080_v1_clay_fraction",
      "clay_fraction_080_090_v1_clay_fraction",
      "clay_fraction_090_100_v1_clay_fraction",
      # Sand fraction (10 depth intervals: 0-100 cm)
      "sand_fraction_000_010_v1_sand_fraction",
      "sand_fraction_010_020_v1_sand_fraction",
      "sand_fraction_020_030_v1_sand_fraction",
      "sand_fraction_030_040_v1_sand_fraction",
      "sand_fraction_040_050_v1_sand_fraction",
      "sand_fraction_050_060_v1_sand_fraction",
      "sand_fraction_060_070_v1_sand_fraction",
      "sand_fraction_070_080_v1_sand_fraction",
      "sand_fraction_080_090_v1_sand_fraction",
      "sand_fraction_090_100_v1_sand_fraction",
      # Silt fraction (10 depth intervals: 0-100 cm)
      "silt_fraction_000_010_v1_silt_fraction",
      "silt_fraction_010_020_v1_silt_fraction",
      "silt_fraction_020_030_v1_silt_fraction",
      "silt_fraction_030_040_v1_silt_fraction",
      "silt_fraction_040_050_v1_silt_fraction",
      "silt_fraction_050_060_v1_silt_fraction",
      "silt_fraction_060_070_v1_silt_fraction",
      "silt_fraction_070_080_v1_silt_fraction",
      "silt_fraction_080_090_v1_silt_fraction",
      "silt_fraction_090_100_v1_silt_fraction",
      # Stoniness (2 depth thresholds: 50 cm, 90 cm)
      "soil_depth_stoniness_50_v2_soil_depth_stoniness",
      "soil_depth_stoniness_90_v2_soil_depth_stoniness",
  )
  BANDS = DEFAULT_BANDS

  asset_name: str = "projects/mapbiomas-public/assets/brazil/soil/collection3"

  @property
  def im(self) -> ee.Image:
    clay_ic = ee.ImageCollection(
        f"{self.asset_name}/mapbiomas_brazil_collection3_soil_clay_fraction_v1"
    )
    sand_ic = ee.ImageCollection(
        f"{self.asset_name}/mapbiomas_brazil_collection3_soil_sand_fraction_v1"
    )
    silt_ic = ee.ImageCollection(
        f"{self.asset_name}/mapbiomas_brazil_collection3_soil_silt_fraction_v1"
    )
    stone_ic = ee.ImageCollection(
        f"{self.asset_name}/mapbiomas_brazil_collection3_soil_stoniness_v2"
    )
    return (
        clay_ic.toBands()
        .addBands(sand_ic.toBands())
        .addBands(silt_ic.toBands())
        .addBands(stone_ic.toBands())
    )

  @property
  def ic(self) -> ee.ImageCollection:
    raise ValueError(
        "This is a static depth-sliced Image, use .im to obtain bands."
    )


@dataclasses.dataclass
class MapBiomasSoilCarbon(ee_base.EeData):
  """MapBiomas Soil Collection 3 - Soil Organic Carbon Stock (t/ha)."""

  start_year: int = 1985
  end_year: int = 2024

  DEFAULT_BANDS = ("carbon",)
  BANDS = DEFAULT_BANDS

  asset_name: str = (
      "projects/mapbiomas-public/assets/brazil/soil/collection3/"
      "mapbiomas_brazil_collection3_soil_carbon_v1"
  )

  @property
  def ic(self) -> ee.ImageCollection:
    return annual_multiband_to_ic(
        asset_name=self.asset_name,
        band_prefix="carbon_",
        out_band="carbon",
        start_year=self.start_year,
        end_year=self.end_year,
    )

  @property
  def im(self) -> ee.Image:
    raise ValueError("This is an annual series, use .ic instead.")


# =============================================================================
# 4. Fire Frequency (Collection 5)
# =============================================================================


@dataclasses.dataclass
class MapBiomasFire(ee_base.EeData):
  """MapBiomas Fire Collection 5 - Cumulative Fire Frequency (30m)."""

  start_year: int = 1985
  end_year: int = 2024

  DEFAULT_BANDS = ("fire_frequency",)
  BANDS = DEFAULT_BANDS

  asset_name: str = (
      "projects/mapbiomas-public/assets/brazil/fire/collection5/"
      "mapbiomas_fire_collection5_fire_frequency_v1"
  )

  @property
  def ic(self) -> ee.ImageCollection:
    return annual_multiband_to_ic(
        asset_name=self.asset_name,
        band_prefix="fire_frequency_1985_",
        out_band="fire_frequency",
        start_year=self.start_year,
        end_year=self.end_year,
    )

  @property
  def im(self) -> ee.Image:
    raise ValueError("This is an annual series, use .ic instead.")


# =============================================================================
# 5. Land Use and Land Cover (LULC)
# =============================================================================


@dataclasses.dataclass
class MapBiomasLulcBrazil(ee_base.EeData):
  """MapBiomas Brazil Land Use / Land Cover (Collection 9 / 10)."""

  DEFAULT_BANDS = ("classification",)
  BANDS = DEFAULT_BANDS

  asset_name: str = "projects/mapbiomas-public/assets/brazil/lulc/v1"

  @property
  def ic(self) -> ee.ImageCollection:
    return ee.ImageCollection(self.asset_name)

  @property
  def im(self) -> ee.Image:
    raise ValueError("This is an ImageCollection, use .ic instead.")


def mapbiomas_lulc_soam_im_fn(unused_asset_name: str) -> ee.Image:
  """Annual BR+BO+EC MapBiomas LULC mosaic with classification_{2017..2024}."""
  assets = (
      "projects/mapbiomas-public/assets/brazil/lulc/v1",
      "projects/mapbiomas-public/assets/bolivia/lulc/v1",
      "projects/mapbiomas-public/assets/ecuador/lulc/v1",
  )
  images = []
  for y in range(2017, 2025):
    ims = [
        ee.ImageCollection(a)
        .filter(ee.Filter.eq("year", y))
        .select(["classification"])
        .mosaic()
        for a in assets
    ]
    images.append(
        ee.ImageCollection(ims).mosaic().rename(f"classification_{y}")
    )
  return ee.Image.cat(*images)
