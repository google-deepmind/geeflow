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

"""Optical GEE datasets."""

import dataclasses
import functools

from geeflow import ee_algo
from geeflow.data_sources import ee_base
import numpy as np

import ee

EeData = ee_base.EeData


@dataclasses.dataclass
class Sentinel2(EeData):
  """Sentinel-2 datasets (harmonized)."""

  mode: str = "L2A"  # L2A, L1C

  BANDS = [
      "B1",
      "B2",
      "B3",
      "B4",
      "B5",
      "B6",
      "B7",
      "B8",
      "B8A",
      "B9",
      "B10",
      "B11",
      "B12",
      "QA10",
      "QA20",
      "QA60",
  ]
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

  def ic_with_cloud_score(self):
    """S2 IC with Cloud Score+ band pre-linked (collection-level join)."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    return self.ic.linkCollection(cs, ["cs_cdf"])

  def filter_by_cloud_percentage(self, percentage):
    return self.ic.filter(f"CLOUDY_PIXEL_PERCENTAGE < {percentage}")

  def add_cloud_probability_band(self):
    """Adds `cloud_probability` band to S2 collection."""
    s2c = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").map(
        lambda x: x.rename(["cloud_probability"])
    )
    s2_fc = ee.Join.saveFirst("cloud_prob").apply(
        self.ic,
        s2c,
        ee.Filter.equals(
            leftField="system:index", rightField="system:index"
        ),
    )
    s2 = ee.ImageCollection(s2_fc).map(
        lambda x: x.addBands(ee.Image(x.get("cloud_prob"))).set(  # pylint: disable=g-long-lambda
            "cloud_prob", "deleted"
        )
    )
    return s2

  @classmethod
  def im_cloud_mask(cls, s2_image):
    """S2 official image cloud masking from the data catalog."""
    qa = s2_image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    is_not_cloud = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return is_not_cloud

  @classmethod
  def im_cloud_mask2(
      cls, s2_image, cloud_prob_min=65, cdi_max=-0.5, cirrus_min=0.01
  ):
    """Cloud mask based on cfb@ DW_v1 processing."""
    cdi = ee.Algorithms.Sentinel2.CDI(s2_image)
    cloud_prob = s2_image.select("cloud_probability")
    cirrus = s2_image.select("B10").multiply(0.0001)
    is_cloud = (
        cloud_prob.gt(cloud_prob_min)
        .And(cdi.lt(cdi_max))
        .Or(cirrus.gt(cirrus_min))
    )
    return is_cloud.Not()

  @classmethod
  def im_cloud_score_plus_mask(cls, s2_image, cloud_prob_min=40, band="cs_cdf"):
    # TODO: Deprecate in favor of the IC-linked version.
    """Cloud mask (0: cloud, 1: clear) based on cloud score+."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    clear_threshold = 1 - cloud_prob_min / 100.0
    is_clear = (
        s2_image.linkCollection(cs, [band]).select(band).gte(clear_threshold)
    )
    return is_clear

  @classmethod
  def im_cloud_score_plus_mask_prelinked(
      cls, s2_image, cloud_prob_min=40
  ):
    """Cloud mask (0: cloud, 1: clear) assuming collection-level link."""
    clear_threshold = 1 - cloud_prob_min / 100.0
    return s2_image.select("cs_cdf").gte(clear_threshold)


@dataclasses.dataclass
class HLS(EeData):
  """Harmonized Landsat Sentinel-2 (HLS) datasets (by NASA)."""

  mode: str = "HLSS30/v002"

  BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
  VIS_BANDS = ["B4", "B3", "B2"]

  @property
  def asset_name(self) -> str:
    return f"NASA/HLS/{self.mode}"

  @classmethod
  def im_cloud_mask(cls, image):
    """HLS Fmask cloud mask."""
    bitmask = int("1110", 2)
    is_cloud = image.select("Fmask").bitwiseAnd(bitmask).neq(0)
    return is_cloud.Not()


@dataclasses.dataclass
class LandsatComposite(EeData):
  """Landsat (4-9) composite datasets computed on the fly (by EE team).

  From 1984. Has annual, 32-day, 8-day composites.
  https://developers.google.com/earth-engine/datasets/tags/landsat-composite
  """

  mode: str = "T1_L2_ANNUAL"
  BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/COMPOSITES/C02/{self.mode}"


@dataclasses.dataclass
class LandsatTimelapse(EeData):
  """Landsat Timelapse composite dataset (1985-2022 only)."""

  BANDS = ["red", "green", "blue"]

  @property
  def asset_name(self) -> str:
    return "projects/ee-earth-timelapse/assets/annual"


@dataclasses.dataclass
class Landsat7(EeData):
  """Landsat7 SR (Surface Reflectance) datasets (1999-2022)."""

  mode: str = "T1_L2"

  BANDS = [
      "SR_B1",
      "SR_B2",
      "SR_B3",
      "SR_B4",
      "SR_B5",
      "SR_B7",
      "SR_ATMOS_OPACITY",
      "SR_CLOUD_QA",
      "ST_B6",
      "ST_ATRAN",
      "ST_CDIST",
      "ST_DRAD",
      "ST_EMIS",
      "ST_EMSD",
      "ST_QA",
      "ST_TRAD",
      "ST_URAD",
      "QA_PIXEL",
      "QA_RADSAT",
  ]
  VIS_BANDS = ["SR_B3", "SR_B2", "SR_B1"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/LE07/C02/{self.mode}"

  @classmethod
  def vis_norm(cls, img: np.ndarray, max_f=1.0, gamma=1.0) -> np.ndarray:
    v_min, v_max = 0.2 / 2.75e-5, 0.5 / 2.75e-5
    if max_f:
      v_max /= max_f
    return np.clip((img - v_min) / (v_max - v_min), 0, 1) ** gamma

  def filter_by_cloud_percentage(self, percentage):
    valid_cc_filter = ee.Filter.gte("CLOUD_COVER", 0)
    cloud_percentage_filter = ee.Filter.lt("CLOUD_COVER", percentage)
    combined_filter = ee.Filter.And(valid_cc_filter, cloud_percentage_filter)
    return self.ic.filter(combined_filter)

  @classmethod
  def im_cloud_mask(cls, image):
    """L7/8/9/5 image cloud mask."""
    qa_bitmask = int("11111", 2)
    is_cloud = image.select("QA_PIXEL").bitwiseAnd(qa_bitmask).neq(0)
    is_saturated = image.select("QA_RADSAT").neq(0)
    is_cloud_or_saturated = is_cloud.Or(is_saturated)
    return is_cloud_or_saturated.Not()


@dataclasses.dataclass
class Landsat8(Landsat7):
  """Landsat8 raw datasets (2013-03 to now)."""

  mode: str = "T1"
  calibrate_radiance: bool = False

  BANDS = [
      "B1",
      "B2",
      "B3",
      "B4",
      "B5",
      "B6",
      "B7",
      "B8",
      "B9",
      "B10",
      "B11",
      "QA_PIXEL",
      "QA_RADSAT",
      "SAA",
      "SZA",
      "VAA",
      "VZA",
  ]
  VIS_BANDS = ["B4", "B3", "B2"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/LC08/C02/{self.mode}"

  @property
  def ic(self):
    def _landsat_calibrated_radiance(image):
      metadata = image.select(
          ["QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"]
      )
      radiance = ee.Algorithms.Landsat.calibratedRadiance(image)
      return metadata.addBands(radiance)

    ic = ee.ImageCollection(self.asset_name)
    if self.mode == "T1" and self.calibrate_radiance:
      ic = ic.map(_landsat_calibrated_radiance).select(self.BANDS)
    return ic


@dataclasses.dataclass
class Landsat9(Landsat7):
  """Landsat9 raw datasets (2021-10 to now)."""

  mode: str = "T1"
  calibrate_radiance: bool = False
  BANDS = [
      "B1",
      "B2",
      "B3",
      "B4",
      "B5",
      "B6",
      "B7",
      "B8",
      "B9",
      "B10",
      "B11",
      "QA_PIXEL",
      "QA_RADSAT",
      "SAA",
      "SZA",
      "VAA",
      "VZA",
  ]
  VIS_BANDS = ["B4", "B3", "B2"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/LC09/C02/{self.mode}"

  @property
  def ic(self):
    def _landsat_calibrated_radiance(image):
      metadata = image.select(
          ["QA_PIXEL", "QA_RADSAT", "SAA", "SZA", "VAA", "VZA"]
      )
      radiance = ee.Algorithms.Landsat.calibratedRadiance(image)
      return metadata.addBands(radiance)

    ic = ee.ImageCollection(self.asset_name)
    if self.mode == "T1" and self.calibrate_radiance:
      ic = ic.map(_landsat_calibrated_radiance).select(self.BANDS)
    return ic


@dataclasses.dataclass
class Landsat5(Landsat7):
  """Landsat5 raw datasets (1984-03 to 2012-05)."""

  mode: str = "T1_L2"
  calibrate_radiance: bool = False
  BANDS = [
      "SR_B1",
      "SR_B2",
      "SR_B3",
      "SR_B4",
      "SR_B5",
      "SR_B7",
      "QA_PIXEL",
      "QA_RADSAT",
  ]
  VIS_BANDS = ["SR_B3", "SR_B2", "SR_B1"]

  @property
  def asset_name(self) -> str:
    return f"LANDSAT/LT05/C02/{self.mode}"


@dataclasses.dataclass
class NAIP(EeData):
  """NAIP datasets."""

  BANDS = ["B", "G", "R", "N"]
  VIS_BANDS = ["R", "G", "B"]

  @property
  def asset_name(self) -> str:
    return "USDA/NAIP/DOQQ"


@dataclasses.dataclass
class Nicfi(EeData):
  """NICFI datasets."""

  mode: str = "asia"

  BANDS = ["B", "G", "R", "N"]
  VIS_BANDS = ["R", "G", "B"]

  @property
  def asset_name(self) -> str:
    return f"projects/planet-nicfi/assets/basemaps/{self.mode}"

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    v_min, v_max, gamma = 64, 5454, 1.8
    return np.clip((img - v_min) / (v_max - v_min), 0, 1) ** gamma

  def filter_by_cadence(self, cadence):
    if cadence not in {"biannual", "monthly"}:
      raise ValueError(f"Unrecognized cadence: {cadence}")
    return self.ic.filterMetadata("cadence", "equals", cadence)


@dataclasses.dataclass
class CandidRgbGeo3d(EeData):
  """Candid RGB."""

  @property
  def asset_name(self) -> str:
    raise NotImplementedError("Asset not publicly available.")

  @property
  def ic(self):
    return (
        ee.ImageCollection(self.asset_name)
        .filter(ee.Filter.lte("pixelSizeMeters", 0.25))
        .filter(ee.Filter.eq("platform", "candid"))
    )

  @property
  def im(self):
    raise ValueError("This is considered as an IC and not an Image.")

  @property
  def image(self):
    raise ValueError("This is considered as an IC and not an Image.")


@dataclasses.dataclass
class CandidRgbMS(EeData):
  """Candid RGB."""

  normalizer: float = 13e3

  @property
  def asset_name(self) -> str:
    return "GOOGLE/GEO/ALL/AERIAL/CANDID/ORTHO/MULTISPECTRAL"

  @property
  def ic(self):

    def pansharpen(image):
      rgb = image.select(["R", "G", "B"]).divide(self.normalizer).clamp(0, 1)
      p = image.select(["P"]).divide(self.normalizer).clamp(0, 1)

      hsv = rgb.rgbToHsv()
      hue = hsv.select("hue")
      saturation = hsv.select("saturation")

      pan = ee.Image.cat([hue, saturation, p]).hsvToRgb()
      return pan.rename(["R", "G", "B"]).multiply(255).toUint8()

    return ee.ImageCollection(self.asset_name).map(pansharpen)

  @property
  def im(self):
    raise ValueError("This is considered as an IC and not an Image.")

  @property
  def image(self):
    raise ValueError("This is considered as an IC and not an Image.")


@dataclasses.dataclass
class PNeoRGB(EeData):
  """Pleiades Neo satellite imagery."""

  @property
  def asset_name(self) -> str:
    return "GOOGLE/GEO/ALL/SATELLITE/PNEO/ORTHO/RGB"

  @property
  def ic(self):
    return ee.ImageCollection(self.asset_name).filter(
        ee.Filter.stringContains(
            "productionID", "earth_airbus_neo_archive_ds"
        ).Not()
    )


class PNeoMultiSpectral(PNeoRGB):
  """Pleiades Neo satellite imagery."""

  @property
  def asset_name(self) -> str:
    return "GOOGLE/GEO/ALL/SATELLITE/PNEO/ORTHO/MULTISPECTRAL"


@dataclasses.dataclass
class Magrathean(EeData):
  """Magrathean datasets (pretty high-res images)."""

  BANDS = ["R", "G", "B"]
  VIS_BANDS = ["R", "G", "B"]

  @property
  def asset_name(self) -> str:
    raise NotImplementedError("Asset not publicly available.")

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    return img

  def filter_and_sort(
      self,
      zoom_level=17,
      project=100,
      ref_date=None,
      max_years_difference=None,
  ):
    """Filters image collection by zoom, project membership, date, and roi."""
    if max_years_difference is not None:
      assert ref_date, "Reference date is required for date filtering."

    ic = self.ic
    filter_project_fn = ee.Filter.inList(
        opt_leftValue=project, opt_rightField="project_membership"
    )
    filter_zoom_fn = ee.Filter.expression(
        f"max_magrathean_zoom_level >= {zoom_level}"
    )
    timestamp_fn = ee.Filter.gt("system:time_start", 0)
    filters = [filter_project_fn, filter_zoom_fn, timestamp_fn]

    if ref_date is not None:
      ref_date = ee.Date(ref_date)
      abs_time_diff_map_fn = functools.partial(
          ee_algo.add_abs_time_difference, ref_date=ref_date
      )
      ic = ic.map(abs_time_diff_map_fn)
      sorting_key = "abs_time_difference"
      sort_ascending = True

      if max_years_difference is not None:
        filter_date_fn = ee.Filter.date(
            ref_date.advance((-1) * max_years_difference / 2, "years"),
            ref_date.advance(max_years_difference / 2, "years"),
        )
        filters.append(filter_date_fn)
    else:
      sorting_key = "max_magrathean_zoom_level"
      sort_ascending = False

    all_filters = ee.Filter(filters)
    return ic.filter(all_filters).sort(sorting_key, sort_ascending)

  def filter_by_zoom_level(self, zoom_level):
    return self.ic.filter(f"max_magrathean_zoom_level >= {zoom_level}")

  @classmethod
  def sort_by_zoom_level(cls, ic):
    return ic.sort("max_magrathean_zoom_level", False)

  @classmethod
  def filter_by_project(cls, ic, project=100):
    filter_fn = ee.Filter.inList(
        opt_leftValue=project, opt_rightField="project_membership"
    )
    return ic.filter(filter_fn)

  @classmethod
  def filter_by_roi_validity(cls, ic, roi, min_validity=0.5, scale=None):
    map_fn = functools.partial(
        ee_algo.add_roi_validity, band="R", roi=roi, scale=scale
    )
    ic = ic.map(map_fn)
    return ic.filter(f"validity >= {min_validity}")
