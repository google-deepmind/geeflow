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

# pylint: disable=line-too-long
"""Pre-configured source presets for GeeFlow.

This module provides ready-to-use source configurations for common data sources
(satellites, DEMs, land cover, etc.). Each preset bundles a data source class
with sensible defaults for scale, bands, sampling algorithm, and other
parameters.
You can then extend and override these presets as needed.

Usage::

    from geeflow.data_sources import presets

    # Get a single preset:
    c.sources.s2 = presets.get('Sentinel2')
    c.sources.s2.date_ranges = my_date_ranges

    # Or get all presets:
    all_presets = presets.get_all()
    c.sources.s2 = all_presets.Sentinel2
"""

import functools

from geeflow import ee_data
from ml_collections import config_dict as cd


S2_BANDS_NO_QA10_QA20 = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10",
    "B11", "B12", "QA60")
# Added "B8A" (4/5/23).
S2_FINAL_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
L7_FINAL_BANDS = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7")
L5_FINAL_BANDS = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7")
# Removed B8, B9, B10, B11 (4/5/23).
L8_FINAL_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9")
L9_FINAL_BANDS = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7")
# Removed "DetailedQA", "SummaryQA" (4/5/23).
MODIS_VEG_BANDS = (
    "NDVI", "EVI", "sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
    "sur_refl_b07", "ViewZenith", "SolarZenith", "RelativeAzimuth")


@functools.cache
def get_all():
  """Returns a ConfigDict with all pre-configured source presets.

  This is the replacement for ``ref_sources.get_config()``.
  Each entry provides sensible defaults for a data source class (scale, bands,
  algorithm, etc.) that can be overridden in project configs.

  Returns:
    A ConfigDict where each key is a source name mapping to a preset config.
  """
  d = cd.ConfigDict()

  def add(name, *, kw={}, out="ic", out_kw={}, select=None,  # pylint: disable=dangerous-default-value
          select_final=None, sampling_kw={}, scale=None, module=None,
          dtype: str | None = None, algo: str | None = None):
    """Registers a single source preset into a ConfigDict.

    Args:
      name: The name of the preset (e.g. "Sentinel2").
      kw: Keyword arguments for the source class constructor.
      out: The output method name (e.g. "ic", "im", "fc").
      out_kw: Keyword arguments for the output method.
      select: Bands/properties to select from the source.
      select_final: Final band selection after sampling.
      sampling_kw: Keyword arguments for the sampling algorithm.
      scale: Spatial resolution in meters.
      module: The source class name in ee_data (defaults to name).
      dtype: Output data type (e.g. "uint8").
      algo: Sampling algorithm name (e.g. "ic_temporal_composites").
    """
    scale = float(scale) if scale is not None else None
    module = module or name
    assert hasattr(ee_data, module)
    d[name] = cd.create(
        module=module, kw=kw, out=out, out_kw=out_kw, select_final=select_final,
        select=select, sampling_kw=sampling_kw, scale=scale, dtype=dtype,
        algo=algo)
    m = getattr(ee_data, module)(**kw)
    if hasattr(m, "dummy_image_id"):
      # Using explicit dummy_image_id allows EE to better optimize data
      # processing and singificantly improve the speed for some special cases.
      d[name].dummy_image_id = m.dummy_image_id
      # In order for dummy_image_id to work, we need to disable filter_date. The
      # data will still be filtered by date inside ic_temporal_composites,
      # should have no effect.
      d[name].filter_date = False

  # Image Collections.
  add(
      "Sentinel1",
      kw={"mode": "IW", "pols": ("VV", "VH"), "orbit": "both"},
      scale=10,
      algo="ic_temporal_composites",
  )
  add(
      "Sentinel1Asc",
      kw={"mode": "IW", "pols": ("VV", "VH"), "orbit": "asc"},
      scale=10,
      module="Sentinel1",
      algo="ic_temporal_composites",
  )
  add(
      "Sentinel1Desc",
      kw={"mode": "IW", "pols": ("VV", "VH"), "orbit": "desc"},
      scale=10,
      module="Sentinel1",
      algo="ic_temporal_composites",
  )
  add(
      "Sentinel2",
      kw={"mode": "L1C"},
      out="filter_by_cloud_percentage",
      out_kw={"percentage": 50},
      select=S2_BANDS_NO_QA10_QA20,
      select_final=S2_FINAL_BANDS,
      scale=10,
      algo="ic_temporal_composites",
  )
  add(
      "Landsat7",
      select_final=L7_FINAL_BANDS,
      out="filter_by_cloud_percentage",
      out_kw={"percentage": 50},
      scale=30,
      algo="ic_temporal_composites",
  )
  add(
      "Landsat5",
      select_final=L5_FINAL_BANDS,
      out="filter_by_cloud_percentage",
      out_kw={"percentage": 50},
      scale=30,
      algo="ic_temporal_composites",
  )
  add(
      "Landsat8",
      select_final=L8_FINAL_BANDS,
      out="filter_by_cloud_percentage",
      out_kw={"percentage": 50},
      scale=30,
      algo="ic_temporal_composites",
  )
  add(
      "Landsat9",
      select_final=L9_FINAL_BANDS,
      scale=30,
      algo="ic_temporal_composites",
  )
  add(
      "LandsatComposite",
      select_final=("blue", "green", "red", "nir", "swir1", "swir2", "thermal"),
      scale=30,
      algo="ic_temporal_composites",
  )
  add(
      "LandsatTimelapse",
      select_final=("red", "green", "blue"),
      scale=30,
      algo="ic_temporal_composites",
  )
  add("NAIP", scale=1, algo="ic_stack")

  # Image Collection - global mosaiced.
  add(
      "Nicfi",
      kw={"mode": "americas"},
      out="filter_by_cadence",
      out_kw={"cadence": "biannual"},
      sampling_kw={"filter_bounds": False},
      scale=5,
      algo="ic_stack",
  )
  add(
      "Alos",
      kw={"mode": "yearly"},
      select=("HH", "HV", "angle"),
      scale=25,
      algo="ic_temporal_composites",
  )
  add(
      "ModisTerraVeg",
      select=MODIS_VEG_BANDS,
      scale=250,
      algo="ic_temporal_composites",
  )
  add(
      "ModisSurfRefl",
      select=[
          "sur_refl_b01",
          "sur_refl_b02",
          "sur_refl_b03",
          "sur_refl_b04",
          "sur_refl_b05",
          "sur_refl_b06",
          "sur_refl_b07",
      ],
      scale=250,
      algo="ic_temporal_composites",
  )
  add(
      "ModisGPP",
      select=["Gpp", "PsnNet"],
      scale=250,
      algo="ic_temporal_composites",
  )
  add(
      "ModisET",
      select=["ET", "LE", "PET", "PLE"],
      scale=250,
      algo="ic_temporal_composites",
  )
  add(
      "ModisBurn",
      select=["BurnDate", "Uncertainty"],
      scale=250,
      algo="ic_temporal_composites",
  )
  add("ModisFire", select=["MaxFRP"], scale=1000, algo="ic_temporal_composites")
  add(
      "CIESIN", out="ic", select="population_count", scale=1000, algo="ic_stack"
  )
  add(
      "WorldPop",
      out="ic",
      select="population",
      scale=100,
      algo="ic_temporal_composites",
  )
  add("GHSPop", out="ic", select="population_count", scale=100, algo="ic_stack")
  add(
      "CCDC",
      out="ic",
      select=ee_data.CCDC.BANDS,
      scale=30,
      sampling_kw={"num_segments": 1},
      algo="ccdc_coefficients",
  )
  add(
      "FIRMS",
      out="ic",
      select=["T21"],
      scale=1000,
      algo="ic_temporal_composites",
  )
  add(
      "DynamicWorld",
      out="ic",
      select=("label",),
      scale=10,
      algo="ic_temporal_composites",
  )
  add("GediRasterCanopyHeight",
      out="ic",
      select=["rh25", "rh95", "rh98", "rh100"],
      scale=30,  # Originally at 25m.
      )
  add("GediRasterCanopyStructure",
      out="ic",
      select=["pai", "cover", "fdh_normal",
              "pai_z0", "pai_z2", "pai_z5", "pai_z10",
              "cover_z0", "cover_z2", "cover_z5", "cover_z10"],
      scale=30,  # Originally at 25m.
      )
  add("GediRasterAGB",
      out="ic",
      select=["agbd"],
      scale=30,  # Originally at 25m.
      )
  add("CustomIC",
      out="ic")

  # Image.
  add(
      "NasaDem",
      out="im",
      select=("elevation", "slope", "aspect"),
      scale=30,
      algo="im_static",
  )
  add(
      "CopDem",
      out="im",
      select=("elevation", "slope", "aspect"),
      scale=30,
      algo="im_static",
  )
  add(
      "FABDEM",
      out="im",
      select=("elevation", "slope", "aspect"),
      scale=30,
      algo="im_static",
  )
  add("WorldCover", out="im", scale=10, algo="im_static")
  add(
      "FPP",
      out="im",
      select="FPP_1km",  # {FPP_1km, FPP_5km} num people within 1/5km of forest.
      scale=100,
      algo="im_static",
  )
  add("Hansen", out="im", algo="im_static")
  add("LandCover", scale=1000, out="im", algo="im_static")
  add(
      "TPP",
      out="im",
      select="TPP_1km",  # TPP_1km, TPP_1km_cropland, TPP_500m, TPP_500m_cropland
      scale=100,
      algo="im_static",
  )
  add("WSF2015", scale=10, out="im", algo="im_static")
  add("TreeCoverLossDueToFire", scale=30, out="im", algo="im_static")
  add(
      "Era5",
      kw={"mode": "monthly"},
      scale=11132,  # ~0.1 arc deg.
      algo="ic_temporal_composites",
  )
  add("Primary", scale=30, out="im", algo="im_static")
  add("CustomImage", out="im", algo="im_static")

  # Feature Collection.
  add(
      "Countries",
      out="fc",
      select=ee_data.Countries.BANDS,
      algo="fc_properties",
  )
  add("CustomFC", out="fc")
  add(
      "MapBiomasLandsat",
      scale=30,
      algo="ic_sample_date_ranges",
      sampling_kw={"reduce_fn": "mosaic"},
  )
  add(
      "MapBiomasAtmosphere",
      scale=1000,
      algo="ic_sample_date_ranges",
      sampling_kw={"reduce_fn": "mean"},
  )
  add(
      "MapBiomasSoil",
      out="im",
      scale=30,
      algo="im_static",
  )
  add(
      "MapBiomasSoilCarbon",
      scale=30,
      algo="ic_sample_date_ranges",
      sampling_kw={"reduce_fn": "mean"},
  )
  add(
      "MapBiomasFire",
      scale=30,
      algo="ic_sample_date_ranges",
      sampling_kw={"reduce_fn": "mosaic"},
  )
  add(
      "MapBiomasLulcBrazil",
      scale=30,
      algo="ic_sample_date_ranges",
      sampling_kw={"reduce_fn": "mosaic"},
  )

  return d


def get(name: str) -> cd.ConfigDict:
  """Returns a single source preset by name."""
  all_presets = get_all()
  if name not in all_presets:
    available = sorted(all_presets.keys())
    raise KeyError(
        f"Unknown source preset: {name!r}. Available presets: {available}"
    )
  return all_presets[name]
