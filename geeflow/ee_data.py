# Copyright 2025 DeepMind Technologies Limited.
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

"""EE datasets."""

import abc
from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any

from geeflow import ee_algo
import matplotlib.colors as mcol
import matplotlib.pylab as plt
import numpy as np

import ee


class EeData(abc.ABC):
  """Abstract class for EE ImageCollection and Image datasets."""

  BANDS = []  # Ordered list of bands.
  VIS_BANDS = []  # Bands for RGB visualization.

  @property
  @abc.abstractmethod
  def asset_name(self) -> str:
    """Returns asset name."""

  @classmethod
  def stack(cls, data, vis=False) -> np.ndarray:
    """Returns HWC numpy array with C=(R,G,B,[N])."""
    bands = cls.VIS_BANDS if vis else cls.BANDS
    if isinstance(data, dict):
      d = [data[band] for band in bands if band in data]
      return np.stack(d, axis=2)  # (H,W,C)
    raise ValueError(f"Unsupported data type: {type(data)}")

  @classmethod
  def vis(cls, data, **kwargs) -> np.ndarray:
    """Returns RGB or single-channel image ready for visualization."""
    img = cls.stack(data, vis=True)
    return cls.vis_norm(img, **kwargs)

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    """Visualization normalization. Can be overwriting by inheriting classes."""
    return img

  @property
  def ic(self):
    return ee.ImageCollection(self.asset_name)

  @property
  def im(self):
    return ee.Image(self.asset_name)

  @property
  def image(self):
    return ee.Image(self.asset_name)


class EeDataFC(EeData, abc.ABC):
  """Abstract class for EE FeatureCollection datasets."""

  @property
  def fc(self):
    return ee.FeatureCollection(self.asset_name)

  @property
  def ic(self):
    raise ValueError("This is a FeatureCollection.")

  @property
  def im(self):
    raise ValueError("This is a FeatureCollection.")


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
    v_min, v_max = 0., 10000.
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)

  @classmethod
  def to_gamma0(cls, img: np.ndarray) -> np.ndarray:
    """Converts 16-bit digital numbers (DN) to gamma-zero backscatter in dB."""
    return 10 * np.log10(img.astype(float)**2) - 83.


@dataclasses.dataclass
class Sentinel2(EeData):
  """Sentinel-2 datasets (harmonized)."""
  mode: str = "L2A"  # L2A, L1C

  BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10",
           "B11", "B12", "QA10", "QA20", "QA60"]
  VIS_BANDS = ["B4", "B3", "B2"]

  @property
  def asset_name(self) -> str:
    if self.mode == "L2A":
      return "COPERNICUS/S2_SR_HARMONIZED"
    elif self.mode == "L1C":
      return "COPERNICUS/S2_HARMONIZED"
    raise ValueError(f"Unsupported mode: {self.mode}")

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    v_min, v_max = 0, 3000
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)

  @property
  def ic(self):
    # Temporal fix to avoid (internal link)
    # This will pass all valid images and filter out corrupted ones.
    return super().ic.filter("CLOUDY_PIXEL_PERCENTAGE <= 100")

  def filter_by_cloud_percentage(self, percentage):
    # return self.ic.filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "lt", percentage)
    return self.ic.filter(f"CLOUDY_PIXEL_PERCENTAGE < {percentage}")

  def add_cloud_probability_band(self):
    """Adds `cloud_probability` band to S2 collection."""
    # Has "probability" band (range 0 to 100).
    s2c = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").map(
        lambda x: x.rename(["cloud_probability"]))
    # Saves first cloud_prob as property. Drops s2 without cloud_prob.
    s2_fc = ee.Join.saveFirst("cloud_prob").apply(
        self.ic, s2c, ee.Filter.equals(  # pytype: disable=attribute-error
            leftField="system:index", rightField="system:index"))
    s2 = ee.ImageCollection(s2_fc).map(
        lambda x: x.addBands(ee.Image(x.get("cloud_prob"))).set(  # pylint: disable=g-long-lambda
            "cloud_prob", "deleted"))  # At the end, resetting the property.
    return s2

  @classmethod
  def im_cloud_mask(cls, s2_image):
    """S2 official image cloud masking from the data catalog."""
    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED
    qa = s2_image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    is_not_cloud = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
                    .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return is_not_cloud  # At 60m (QA60 band) resolution.

  @classmethod
  def im_cloud_mask2(cls, s2_image,
                     cloud_prob_min=65, cdi_max=-0.5, cirrus_min=0.01):
    """Cloud mask based on cfb@ DW_v1 processing."""
    # Cloud Displacement Index from S2 L1C.
    cdi = ee.Algorithms.Sentinel2.CDI(s2_image)
    # Should have been added in add_cloud_probability_band.
    cloud_prob = s2_image.select("cloud_probability")
    cirrus = s2_image.select("B10").multiply(0.0001)
    is_cloud = (cloud_prob.gt(cloud_prob_min)
                .And(cdi.lt(cdi_max))
                .Or(cirrus.gt(cirrus_min)))
    return is_cloud.Not()  # Image mask at 10m resolution.

  @classmethod
  def im_cloud_score_plus_mask(cls, s2_image, cloud_prob_min=40, band="cs_cdf"):
    """Cloud mask (0: cloud, 1: clear) based on cloud score+."""
    # CS ranges: 0: not clear(occluded), 1: clear.
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    clear_threshold = 1 - cloud_prob_min / 100.
    is_clear = (s2_image
                .linkCollection(cs, [band])
                .select(band)
                .gte(clear_threshold))
    return is_clear  # Image mask at 10m resolution.


@dataclasses.dataclass
class Landsat7(EeData):
  """Landsat7 SR (Surface Reflectance) datasets (1999-2022)."""
  mode: str = "T1_L2"  # "T2_L2"

  # TOA (Top of Atmosphere) bands have different names (no SR (Surface Refl.)).
  BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7",
           "SR_ATMOS_OPACITY", "SR_CLOUD_QA", "ST_B6", "ST_ATRAN", "ST_CDIST",
           "ST_DRAD", "ST_EMIS", "ST_EMSD", "ST_QA", "ST_TRAD", "ST_URAD",
           "QA_PIXEL", "QA_RADSAT"]
  VIS_BANDS = ["SR_B3", "SR_B2", "SR_B1"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/LE07/C02/{self.mode}"

  @classmethod
  def vis_norm(cls, img: np.ndarray, max_f=1., gamma=1.) -> np.ndarray:
    # Official:
    v_min, v_max = 0.2 / 2.75e-5, 0.5 / 2.75e-5
    # Reduce max for better visualization:
    if max_f:
      v_max /= max_f
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)**gamma

  def filter_by_cloud_percentage(self, percentage):
    # TODO What is the difference to CLOUD_COVER_LAND?
    valid_cc_filter = ee.Filter.gte("CLOUD_COVER", 0)
    cloud_percentage_filter = ee.Filter.lt("CLOUD_COVER", percentage)
    combined_filter = ee.Filter.And(valid_cc_filter, cloud_percentage_filter)
    return self.ic.filter(combined_filter)

  @classmethod
  def im_cloud_mask(cls, image):
    """L7/8/9/5 image cloud mask."""
    # https://developers.google.com/earth-engine/guides/ic_visualization#compositing
    qa_bitmask = int("11111", 2)  # Fill and cloud bits.
    is_cloud = image.select("QA_PIXEL").bitwiseAnd(qa_bitmask).neq(0)
    is_saturated = image.select("QA_RADSAT").neq(0)
    is_cloud_or_saturated = is_cloud.Or(is_saturated)
    return is_cloud_or_saturated.Not()  # @30m. {0: bad, 1: good}.


@dataclasses.dataclass
class Landsat8(Landsat7):
  """Landsat8 raw datasets (2013-03 to now)."""
  mode: str = "T1"
  calibrate_radiance: bool = False

  BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
           "B11", "QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"]
  VIS_BANDS = ["B4", "B3", "B2"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/LC08/C02/{self.mode}"

  @property
  def ic(self):
    def _landsat_calibrated_radiance(image):
      """Calculates calibrated radiance."""
      metadata = image.select(
          ["QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"]
      )
      radiance = ee.Algorithms.Landsat.calibratedRadiance(image)
      return metadata.addBands(radiance)

    ic = ee.ImageCollection(self.asset_name)
    if self.mode == "T1" and self.calibrate_radiance:
      # Raw data to calibrated radiance
      ic = ic.map(_landsat_calibrated_radiance).select(self.BANDS)
    return ic


@dataclasses.dataclass
class Landsat9(Landsat7):
  """Landsat9 raw datasets (2021-10 to now)."""

  mode: str = "T1"
  calibrate_radiance: bool = False
  BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
           "B11", "QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"]
  VIS_BANDS = ["B4", "B3", "B2"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/LC09/C02/{self.mode}"

  @property
  def ic(self):
    def _landsat_calibrated_radiance(image):
      """Calculates calibrated radiance."""
      metadata = image.select(
          ["QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"]
      )
      radiance = ee.Algorithms.Landsat.calibratedRadiance(image)
      return metadata.addBands(radiance)

    ic = ee.ImageCollection(self.asset_name)
    if self.mode == "T1" and self.calibrate_radiance:
      # Raw data to calibrated radiance
      ic = ic.map(_landsat_calibrated_radiance).select(self.BANDS)
    return ic


@dataclasses.dataclass
class NAIP(EeData):
  """NAIP datasets."""
  # https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ

  BANDS = ["B", "G", "R", "N"]
  VIS_BANDS = ["R", "G", "B"]

  @property
  def asset_name(self) -> str:
    return "USDA/NAIP/DOQQ"


@dataclasses.dataclass
class Nicfi(EeData):
  """NICFI datasets."""
  mode: str = "asia"  # {asia, americas, africa}.

  BANDS = ["B", "G", "R", "N"]
  VIS_BANDS = ["R", "G", "B"]

  @property
  def asset_name(self) -> str:
    return f"projects/planet-nicfi/assets/basemaps/{self.mode}"

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    v_min, v_max, gamma = 64, 5454, 1.8
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)**gamma

  def filter_by_cadence(self, cadence):
    if cadence not in {"biannual", "monthly"}:
      raise ValueError(f"Unrecognized cadence: {cadence}")
    return self.ic.filterMetadata("cadence", "equals", cadence)


@dataclasses.dataclass
class ModisTerraVeg(EeData):
  """MODIS Terra Vegetation Indices."""
  mode: str = "250m"  # {250m, 500m, 1km}.

  BANDS = ["NDVI", "EVI", "DetailedQA", "sur_refl_b01", "sur_refl_b02",
           "sur_refl_b03", "sur_refl_b07", "ViewZenith", "SolarZenith",
           "RelativeAzimuth", "DayOfYear", "SummaryQA"]
  VIS_BANDS = ["NDVI"]
  PALETTE = [
      "FFFFFF", "CE7E45", "DF923D", "F1B555", "FCD163", "99B718", "74A901",
      "66A000", "529400", "3E8601", "207401", "056201", "004C00", "023B01",
      "012E01", "011D01", "011301"]

  @property
  def asset_name(self) -> str:
    if self.mode == "250m":  # dim: [172800, 67200], crs: "SR-ORG:6974"
      return "MODIS/061/MOD13Q1"  # 16-day composite 250m.
    elif self.mode == "500m":
      return "MODIS/061/MOD13A1"  # 16-day composite 500m.
    elif self.mode == "1km":
      return "MODIS/061/MOD13A2"  # 16-day composite 1km.
    raise ValueError(f"Unsupported mode: {self.mode}")

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    v_min, v_max = 0, 9000
    img = img[..., 0]  # A single NDVI band by default.
    return np.clip((img - v_min) / (v_max - v_min), 0, 1)


@dataclasses.dataclass
class ModisSurfRefl(EeData):
  """MODIS Terra Surface Reflectance."""
  # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1
  mode: str = "8d-500m-7b"  # {8d-500m-7b, 8d-250m-2b}.

  BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04",
           "sur_refl_b05", "sur_refl_b06", "sur_refl_b07",
           "SolarZenith", "ViewZenith", "RelativeAzimuth",
           "QA", "StateQA", "DayOfYear"]
  VIS_BANDS = ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"]  # R,G,B.

  @property
  def asset_name(self) -> str:
    if self.mode == "8d-500m-7b":  # Default.
      return "MODIS/061/MOD09A1"  # 8-days composite, 500m res, 7-bands
    raise ValueError(f"Unsupported mode: {self.mode}")


@dataclasses.dataclass
class ModisGPP(EeData):
  """MODIS Terra Gross Primary Productivity."""
  # https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD17A2H
  mode: str = "8d-500mb"

  BANDS = ["Gpp", "PsnNet", "Psn_QC"]
  VIS_BANDS = ["Gpp"]

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MOD17A2H"  # 8-days composite, 500m res.


@dataclasses.dataclass
class ModisET(EeData):
  """MODIS Terra EvapoTranspiration."""
  # https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2
  mode: str = "8d-500m"

  BANDS = ["ET", "LE", "PET", "PLE", "ET_QC"]
  VIS_BANDS = ["ET"]

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MOD16A2"  # 8-days composite, 500m res.


@dataclasses.dataclass
class ModisBurn(EeData):
  """MODIS Burned Area."""
  # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD64A1
  mode: str = "m-500m"

  BANDS = ["BurnDate", "Uncertainty", "QA", "FirstDay", "LastDay"]
  # Use spatial nn sampling, not bilinear/cubic.
  VIS_BANDS = ["BurnDate"]

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MCD64A1"  # Monthly, 500m res.


@dataclasses.dataclass
class ModisFire(EeData):
  """MODIS Thermal Anomalies & Fire."""
  # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1
  # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A2
  mode: str = "8-day"  # {8-day, daily, yearly}.

  BANDS = ["MaxFRP"]  # Maximum fire radiative power.
  # Other band: "FireMask" (9/8/7: fire high/medium/low confidence)

  @property
  def asset_name(self) -> str:
    if self.mode == "yearly":
      # It is based on the daily (https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1)  # pylint: disable=line-too-long
      # dataset with max (across a year) temporal reduction.
      return "projects/computing-engine-190414/assets/arbaro/preaggregated/MODIS_061_MOD14A1_yearly"
    postfix = ({"8-day": "2", "daily": "1"})[self.mode]
    return f"MODIS/061/MOD14A{postfix}"  # 1km, global (2000-present).


@dataclasses.dataclass
class FIRMS(EeData):
  """FIRMS: Fire Information for Resource Management System."""
  # https://developers.google.com/earth-engine/datasets/catalog/FIRMS
  mode: str = "daily"  # {daily, yearly}

  @property
  def asset_name(self) -> str:
    if self.mode == "yearly":
      # It is based on the daily dataset (FIRMS) with max (across a year)
      # temporal reduction.
      return "projects/computing-engine-190414/assets/arbaro/preaggregated/FIRMS_yearly"
    return "FIRMS"


@dataclasses.dataclass
class LandCover(EeData):
  """Global Consensus Landcover."""
  # https://gee-community-catalog.org/projects/gcl/
  # Possible values are: barren, cultivated_and_managed_vegetation,
  # deciduous_broadleaf_trees, evergreen-deciduous_needleleaf_trees,
  # evergreen_broadleaf_trees, herbaceous_vegetation, mixed-other_trees,
  # open_water, regularly_flooded_vegetation, shrubs, snow-ice, urban-built-up.
  mode: str = "barren"

  @property
  def asset_name(self) -> str:
    return (
        f"projects/sat-io/open-datasets/global_consensus_landcover/{self.mode}")


@dataclasses.dataclass
class WSF2015(EeData):
  """World Settlement Footprint 2015."""
  # https://developers.google.com/earth-engine/datasets/catalog/DLR_WSF_WSF2015_v1

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
  # https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2022_v1_10
  mode: str = "2022_v1_10"

  BANDS = ["treecover2000",  # Canopy cover percentage (trees > 5m) in 2000.
           "loss", "gain",  # {0: no loss/gain, 1: loss/gain occurred}.
           "first_b30", "first_b40", "first_b50", "first_b70",
           "last_b30", "last_b40", "last_b50", "last_b70",
           # Red, NIR, SWIR-1, SWIR-2 bands for first and last years.
           "datamask",  # {0: no data, 1: land, 2: permanent water}.
           "lossyear",  # {0: no loss, 1-22: 2001-2022}.
           ]

  @property
  def asset_name(self) -> str:
    # TODO: Update/remove this when the data is properly published.
    if self.mode == "2024":
      return "projects/glad/GFC/2024/global_forest_change_2024_v1_12_merged"
    return "UMD/hansen/global_forest_change_" + self.mode


@dataclasses.dataclass
class ModisLaiFpar(EeData):
  """MODIS LAI & FPAR."""

  @property
  def asset_name(self) -> str:
    return "MODIS/061/MCD15A3H"  # 4-day composite 500m.


@dataclasses.dataclass
class NasaDem(EeData):
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
    return (elevation
            .addBands(ee.Terrain.slope(elevation))  # pytype: disable=attribute-error
            .addBands(ee.Terrain.aspect(elevation)))  # pytype: disable=attribute-error

  @property
  def ic(self):
    raise ValueError("This is an Image and not a Collection.")


@dataclasses.dataclass
class FABDEM(EeData):
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
class CopDem(EeData):
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
    elevation = (orig_ic
                 .mosaic()  # Results in 1 deg cells, and requires...
                 .reproject(orig_csr)  # ...reprojection for slope/aspect.
                 .select("DEM")
                 .rename(["elevation"]))
    # Elevation above geoid in meters, slope/aspect in deg [0..90], [0..360].
    return (elevation
            .addBands(ee.Terrain.slope(elevation))  # pytype: disable=attribute-error
            .addBands(ee.Terrain.aspect(elevation)))  # pytype: disable=attribute-error

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")


@dataclasses.dataclass
class GediRasterCanopyHeight(EeData):
  """GEDI rasterized canopy top height (L2A)."""
  # Monthly from 2019-04 to present; 25m resolution; -51.6 to 51.6 deg latitude.
  # 8 beams sample ~25m footprints spaced approximately every 60m along-track.
  filter_quality: bool = True

  # Suggested: rh25, rh95, rh98, rh100, canopy ratio ((rh98-rh25)/rh98).
  BANDS = ["digital_elevation_model", "landsat_treecover",
           "landsat_water_persistence", "modis_treecover", "modis_nonvegetated",
           "urban_proportion"] + [f"rh{p}" for p in range(101)]

  @property
  def asset_name(self) -> str:
    return "LARSE/GEDI/GEDI02_A_002_MONTHLY"

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name)
    if self.filter_quality:
      # Based on the example in
      # https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY
      # https://code.earthengine.google.com/?scriptPath=Examples%3ADatasets%2FLARSE%2FLARSE_GEDI_GEDI02_A_002_MONTHLY
      def quality_mask_fn(im) -> ee.Image:
        return (im
                .updateMask(im.select("quality_flag").eq(1))
                .updateMask(im.select("degrade_flag").eq(0)))
      ic = ic.map(quality_mask_fn)
    return ic


@dataclasses.dataclass
class GediRasterCanopyStructure(EeData):
  """GEDI rasterized canopy cover vertical profile metrics (L2B)."""
  # Monthly from 2019-04 to present; 25m resolution.
  filter_quality: bool = True

  # Suggested: pai (total plant area index), cover (canopy cover),
  # fhd_normal (foliage height height distribution).
  BANDS = (["pai", "cover", "fhd_normal"] +
           [f"cover_z{p}" for p in range(31)] +
           [f"pai_z{p}" for p in range(31)] +
           [f"pavd_z{p}" for p in range(31)])

  @property
  def asset_name(self) -> str:
    return "LARSE/GEDI/GEDI02_B_002_MONTHLY"

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name)
    if self.filter_quality:
      # Based on the example in
      # https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_B_002_MONTHLY
      def quality_mask_fn(im) -> ee.Image:
        return (im
                .updateMask(im.select("l2b_quality_flag").eq(1))
                .updateMask(im.select("degrade_flag").eq(0)))
      ic = ic.map(quality_mask_fn)
    return ic


@dataclasses.dataclass
class GediRasterAGB(EeData):
  """GEDI rasterized aboveground biomass density (L4A)."""
  # Monthly from 2019-04 to present; 25m resolution.
  filter_quality: bool = True

  BANDS = ["agbd", "elev_lowestmode"]

  @property
  def asset_name(self) -> str:
    return "LARSE/GEDI/GEDI04_A_002_MONTHLY"

  @property
  def ic(self):
    ic = ee.ImageCollection(self.asset_name)
    if self.filter_quality:
      # Based on the example in
      # https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI04_A_002_MONTHLY
      def quality_mask_fn(im) -> ee.Image:
        return (im
                .updateMask(im.select("l4_quality_flag").eq(1))
                .updateMask(im.select("degrade_flag").eq(0)))
      ic = ic.map(quality_mask_fn)
    return ic


@dataclasses.dataclass
class QiuDisturbance(EeData):
  """Qiu's US CONUS disturbance data (1988-2022).

    Qiu, et al. 2025. “A Shift from Human-Directed to Undirected Wild Land
    Disturbances in the USA.” Nature Geoscience 18 (10): 989-96.

    US CONUS disturbance drivers at 30m resolution for 38 bands (1985-2022),
    with driver values (0: no change, 1: Logging, 2: Construction,
    3: Agriculture, 4: Stress, 5: Wind / geohazard, 6: Fire, 7: Water).
    Note that predictions are only for 1985-2022; the first three years are used
    only for time series initialization.
  """
  ref_year: int = 2020  # The year to use as a reference for forest age.

  BANDS = [
      "logging_events_count",  # Independent of the ref_year.
      "avg_rotation_time",  # Independent of the ref_year.
      "rotations_count",  # Independent of the ref_year.
      "forest_age_after_logging",  # Age at ref_year.
      "forest_age_after_disturbance",  # Age at ref_year.
      "last_disturbance_driver",  # The last disturbance driver <= ref_year.
  ]

  @property
  def asset_name(self) -> str:
    return "users/ShiQiu/product/conus/disturbance/v081/APRI"

  @property
  def im(self):
    start_year, num_bands = 1985, 38
    assert (start_year < self.ref_year < start_year + num_bands)

    years = ee.List([ee.String(str(start_year + i)) for i in range(num_bands)])
    # 38 bands (1985-2022), with disturbance driver values.
    disturbances = ee.ImageCollection(self.asset_name).mosaic().rename(years)

    # 1. Number of logging events
    logging = disturbances.eq(1)
    logging_count = logging.reduce(ee.Reducer.sum())

    # 2-4. Rotation lengths & Number of rotations
    indices = ee.Image(ee.Array(
        ee.List.sequence(0, num_bands - 1))).arrayMask(logging.toArray())
    unmasked_indices = indices.unmask(
        ee.Image(ee.Array(ee.List.repeat(0, num_bands))))
    sorted_indices = unmasked_indices.arraySort()
    indices_array = sorted_indices.arrayMask(sorted_indices.lt(num_bands))
    shifted = indices_array.arraySlice(0, 1)
    original = indices_array.arraySlice(0, 0, -1)
    intervals = shifted.subtract(original).subtract(1)
    avg_interval = intervals.arrayReduce(ee.Reducer.mean(), [0]).arrayGet(0)
    num_intervals = intervals.arrayReduce(ee.Reducer.count(), [0]).arrayGet(0)

    # 6. Get the driver of the latest disturbance event <= ref_year
    selected_years = years.slice(0, self.ref_year-start_year + 1)

    def get_change(band_name):
      change_this_year = disturbances.select([band_name])
      masked_change = change_this_year.updateMask(change_this_year.gt(0))
      return masked_change.rename("last_change")

    latest_driver = ee.ImageCollection(selected_years.map(get_change)).mosaic()

    # 5. Get the forest age in ref_year
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
        last_change_year)

    def get_year_of_last_logging(band_name):
      year_value = ee.Number.parse(band_name).int16()
      year_image = ee.Image.constant(year_value).int16()
      mask = logging.select([band_name])
      return year_image.updateMask(mask).rename("year_of_last_change")

    last_logging_year = ee.ImageCollection(
        selected_years.map(get_year_of_last_logging)
    ).mosaic()
    years_since_last_logging = ee.Image.constant(self.ref_year).subtract(
        last_logging_year)

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
  # 30m, global.
  mode: str = "2001"  # {2001, 2020}

  BANDS = ["Primary_HT_forests"]

  @property
  def asset_name(self) -> str:
    if self.mode == "2001":
      return "UMD/GLAD/PRIMARY_HUMID_TROPICAL_FORESTS/v1"
    elif self.mode == "2020":
      # Created offline with:
      # hansen = ee.Image("UMD/hansen/global_forest_change_2020_v1_8")
      # primary_2020 = primary_2001.multiply(
      #   hansen.select(["loss"]).multiply(-1).add(1)).mask(primary_2020.eq(1))
      # Colab:
      #   (internal link)?resourcekey=0-S7QHnkUbIaW_fUxrZPAKWg
      return "projects/computing-engine-190414/assets/arbaro/suso/primary_forests_2020_v2"
    else:
      raise ValueError(f"Unsupported mode: {self.mode}")

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")

  @property
  def im(self):
    if self.mode == "2001":  # There is an only a single image.
      return ee.ImageCollection(self.asset_name).first()
    return ee.Image(self.asset_name)  # 2020 is correctly a single Image.


@dataclasses.dataclass
class CocoaAbu(EeData):
  """Cocoa map by Abu."""
  mode: str = "gha"  # {gha, civ}

  BANDS = ["b1"]  # {0: no cocoa, 1: cocoa}

  @property
  def asset_name(self) -> str:
    return (
        f"projects/computing-engine-190414/assets/arbaro/cocoa/abu_{self.mode}")

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")


@dataclasses.dataclass
class NLCD(EeData):
  """USGS National Land Cover Database."""
  # US only, 30m resolution, 20 classes.
  # 2019 release years: 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019.
  # 2016 release years: 1992, 2001, 2004, 2006, 2008, 2011, 2013, 2016.
  mode: str = "2019"

  BANDS = ["landcover", "impervious", "impervious_descriptor"]

  @property
  def asset_name(self) -> str:
    # Individual year images can be retrieved with:
    # f"USGS/NLCD_RELEASES/{self.mode}_REL/NLCD/{year}"
    return f"USGS/NLCD_RELEASES/{self.mode}_REL/NLCD"

  def get_year_im(self, year):
    """Returns a single image for the given year."""
    return self.ic.filter(f"system:index == {year}").first()


@dataclasses.dataclass
class DynamicWorld(EeData):
  """DynamicWorld ImageCollection."""
  # 10m, global, 9 land cover classes.
  mode: str = "V1"

  BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
           "shrub_and_scrub", "built", "bare", "snow_and_ice", "label"]
  VIS_BANDS = ["label"]
  PALETTE = ["419BDF", "397D49", "88B053", "7A87C6", "E49635",
             "DFC35A", "C4281B", "A59B8F", "B39FE1"]
  # Palette (11 colors, manual annotations): [void] + BANDS[:-1] + [clouds]
  PALETTE_11 = ["000000", "419BDF", "397D49", "88B053", "7A87C6", "E49635",
                "DFC35A", "C4281B", "A59B8F", "B39FE1", "FFFFFF"]

  @property
  def asset_name(self) -> str:
    return f"GOOGLE/DYNAMICWORLD/{self.mode}"

  @classmethod
  def cmap(cls):
    return mcol.ListedColormap([f"#{c}" for c in cls.PALETTE])

  @classmethod
  def cmap_names(cls) -> tuple[str, ...]:
    # For 11 classes (manual annotations):
    # return ("void",) + tuple(cls.BANDS[:-1]) + ("clouds")
    return tuple(cls.BANDS[:-1])

  @classmethod
  def imshow(cls, *args, **kwargs):
    # If data is downloaded via adapted download function, then the orientation
    # is probably correct. Otherwise, set origin="lower".
    plt.imshow(*args, vmin=0, vmax=9, cmap=cls.cmap(), interpolation="nearest",
               **kwargs)

  @classmethod
  def colorbar(cls, *args, **kwargs):
    cbar = plt.colorbar(*args, ticks=np.arange(len(cls.PALETTE))+0.5, **kwargs)
    cbar.ax.set_yticklabels(f"{i}: {c}" for i, c in enumerate(cls.BANDS[:-1]))


@dataclasses.dataclass
class WorldCover(EeData):
  """ESA WorldCover (Image) for 2020 and 2021."""
  # https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200
  # 10m, global, 11 land cover classes, single image (uint8).
  # Band values are not consecutive.
  mode: str = "2020"  # {v100: 2020, v200: 2021}

  BANDS = ["Map"]

  @property
  def asset_name(self) -> str:
    modes = {"2020": "v100", "2021": "v200"}
    return f"ESA/WorldCover/{modes[self.mode]}"

  @property
  def im(self):
    # Band values are not consecutive.
    return self.ic.first()  # There is only a single image.


@dataclasses.dataclass
class FPP(EeData):
  """Forest proximate people (2019)."""
  # Number of people living in or within 1km or 5km of forests.
  # Global, 100m resolution. It's only a single image in an ImageCollection.

  BANDS = ["FPP_1km", "FPP_5km"]  # Num people per ha.

  @property
  def asset_name(self) -> str:
    return "FAO/SOFO/1/FPP"

  @property
  def im(self):
    return self.ic.first()  # There is only a single image.


@dataclasses.dataclass
class TPP(EeData):
  """Tree proximate people (2019)."""
  # Number of people living in or within 1km or 500m from croplands/agricultural
  # lands with at least 10% of tree cover. Global, 100m resolution.

  BANDS = ["TPP_1km", "TPP_1km_cropland", "TPP_500m", "TPP_500m_cropland"]

  @property
  def asset_name(self) -> str:
    return "FAO/SOFO/1/TPP"

  @property
  def im(self):
    return self.ic.first()  # There is only a single image.


@dataclasses.dataclass
class CIESIN(EeData):
  """The estimated number of persons per 30 arc-second grid cell."""
  # A distribution of global human population for the years 2000, 2005, 2010,
  # 2015, and 2020 on 30 arc-second (approximately 1km) grid cells.
  mode: str = "Count"  # Also "Density" is supported

  @property
  def asset_name(self) -> str:
    return f"CIESIN/GPWv411/GPW_Population_{self.mode}"


@dataclasses.dataclass
class WorldPop(EeData):
  """The estimated number of persons per 100m grid cell."""
  # WorldPop Global Project Population Data: Estimated Residential Population
  # per 100x100m Grid Square.

  @property
  def asset_name(self) -> str:
    return "WorldPop/GP/100m/pop"


@dataclasses.dataclass
class GHSPop(EeData):
  """The estimated number of persons per 100m grid cell."""
  # GHS-POP R2023A - GHS population grid multitemporal (1975-2030)

  @property
  def asset_name(self) -> str:
    return "JRC/GHSL/P2023A/GHS_POP"


@dataclasses.dataclass
class CCDC(EeData):
  """Google Global Landsat-based CCDC Segments (1999-2019)."""
  mode: str = "V1"
  BANDS = ["tStart", "tEnd", "tBreak", "numObs", "changeProb", "BLUE_coefs",
           "GREEN_coefs", "RED_coefs", "NIR_coefs", "SWIR1_coefs",
           "SWIR2_coefs", "BLUE_rmse", "GREEN_rmse", "RED_rmse", "NIR_rmse",
           "SWIR1_rmse", "SWIR2_rmse", "BLUE_magnitude", "GREEN_magnitude",
           "RED_magnitude", "NIR_magnitude", "SWIR1_magnitude",
           "SWIR2_magnitude"]

  @property
  def asset_name(self) -> list[str]:
    """Returns asset name."""
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


@dataclasses.dataclass
class Ecoregions(EeDataFC):
  """Ecoregions FC for 2017."""

  @property
  def asset_name(self) -> str:
    return "RESOLVE/ECOREGIONS/2017"


@dataclasses.dataclass
class CustomFC(EeDataFC):
  """A custom source that loads a FeatureCollection.

  An example:
    c.gedi = copy.deepcopy(all_sources.CustomFC)
    c.gedi.kw.asset_name =
        "projects/computing-engine-190414/assets/gedi_nico_all_predictions"
    c.gedi.select = ["canopy_height"]
    c.gedi.scale = 10
    c.gedi.algo = ee_algo.fc_to_image

  Attributes:
    asset_name: A name or a list of names of the assets to load.
    filters: A list of filters to apply to the asset.
    buffer_points: How many meters to buffer the point features.
    buffer: How many meters to buffer all features (on top of buffer_points).
    use_bounds: Whether to use bounds instead of actual geometries.
    set_property: A tuple of (property_name, property_value) to set on the
      features.
  """
  asset_name: Sequence[str] | str = ""  # Needs to be specified.
  filters: Sequence[tuple[str, Any]] | None = None
  buffer_points: int = 0
  buffer: int = 0
  # NOTE: Currently ".bounds" methiod is very slow and could incurr very
  # significant slowdown. For more context, see:
  # (internal link)
  # Only use on small collections.
  use_bounds: bool = False
  set_property: tuple[str, Any] | None = None

  @property
  def fc(self):
    if isinstance(self.asset_name, (tuple, list)):
      fc = ee.FeatureCollection(
          [ee.FeatureCollection(x) if isinstance(x, str) else CustomFC(**x).fc
           for x in self.asset_name])
      fc = fc.flatten()
    else:
      fc = ee.FeatureCollection(self.asset_name)
    if self.filters:
      for k, v in self.filters:
        if isinstance(v, (tuple, list)):
          if k.startswith("!"):
            fc = fc.filter(ee.Filter.inList(k[1:], v).Not())
          else:
            fc = fc.filter(ee.Filter.inList(k, v))
        else:
          if k.startswith("<="):
            fc = fc.filter(ee.Filter.lte(k[2:], v))
          elif k.startswith("<"):
            fc = fc.filter(ee.Filter.lt(k[1:], v))
          elif k.startswith(">="):
            fc = fc.filter(ee.Filter.gte(k[2:], v))
          elif k.startswith(">"):
            fc = fc.filter(ee.Filter.gt(k[1:], v))
          elif k.startswith("!~"):
            fc = fc.filter(ee.Filter.stringContains(k[2:], v).Not())
          elif k.startswith("~"):
            fc = fc.filter(ee.Filter.stringContains(k[1:], v))
          elif k.startswith("!"):
            fc = fc.filter(ee.Filter.neq(k[1:], v))
          else:
            fc = fc.filter(ee.Filter.eq(k, v))
    if self.buffer_points > 0:
      fc_points = fc.filter(ee.Filter.hasType(".geo", "Point"))
      fc_not_points = fc.filter(ee.Filter.hasType(".geo", "Point").Not())
      fc_points = fc_points.map(lambda x: x.buffer(self.buffer_points))
      if self.use_bounds:
        fc_points = fc_points.map(lambda x: x.bounds())
      fc = ee.FeatureCollection([fc_points, fc_not_points]).flatten()
    # NOTE: We allow for negative values too.
    if self.buffer:
      fc = fc.map(lambda x: x.buffer(self.buffer))
    if self.set_property:
      fc = fc.map(lambda x: x.set(self.set_property[0], self.set_property[1]))
    return fc


@dataclasses.dataclass
class CustomImage(EeData):
  """A custom source that loads an image.

  An example:
    c.primary_forest = copy.deepcopy(all_sources.CustomImage)
    c.primary_forest.kw.asset_name =
        "projects/computing-engine-190414/assets/arbaro/suso/primary_forests"
  """
  asset_name: str = ""  # Needs to be specified.
  im_fn: Callable[[str], ee.Image] | None = None

  @property
  def im(self):
    if self.im_fn:
      return self.im_fn(self.asset_name)
    return ee.Image(self.asset_name)

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")


@dataclasses.dataclass
class CustomIC(EeData):
  """A custom source that loads an IC.

  An example:
    c.google = copy.deepcopy(all_sources.CustomImage)
    c.google.kw.asset_name =
        "projects/satellite-segmentation/assets/labels"
  """
  # asset_name could refer to:
  #  - str: a single ImageCollection (merge should be False)
  #  - list/tuple: list of Image assets (merge should be False)
  #  - list/tuple: list of ImageCollection assets (merge should be True)
  asset_name: str | list[str] = ""  # Needs to be specified.
  merge: bool = False
  ic_fn: Callable[[str | list[str]], ee.ImageCollection] | None = None

  @property
  def im(self):
    raise ValueError("This is considered as an IC and not an Image.")

  @property
  def ic(self):
    if self.ic_fn:
      assert not self.merge, "merge should be handled by ic_fn if ic_fn given"
      return self.ic_fn(self.asset_name)
    if self.merge and not isinstance(self.asset_name, str):
      ic = ee.ImageCollection(self.asset_name[0])
      for asset_name in self.asset_name[1:]:
        ic = ic.merge(ee.ImageCollection(asset_name))
      return ic
    return ee.ImageCollection(self.asset_name)


@dataclasses.dataclass
class Countries(EeDataFC):
  """Country boundaries FC for 2017."""
  mode: str = "simple"  # {simple, detailed}
  BANDS = ["abbreviati", "country_co", "country_na", "wld_rgn"]

  @property
  def asset_name(self) -> str:
    sel = "_SIMPLE" if self.mode == "simple" else ""
    return f"USDOS/LSIB{sel}/2017"

  def filter_by_name(self, country_names):
    if isinstance(country_names, str):
      country_names = [country_names]
    country_names = [x.capitalize() for x in country_names]
    return self.fc.filter(ee.Filter.inList(
        "country_na", country_names))


@dataclasses.dataclass
class Era5(EeData):
  """Era5."""

  mode: str = "monthly"  # {monthly, daily}
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
