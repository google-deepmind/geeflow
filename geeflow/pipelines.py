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

"""Pipelines for processing EE Geo data."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import io
import os
import tempfile
from typing import Any

from absl import logging
from geeflow import coords
from geeflow import ee_algo
from geeflow import ee_data
from geeflow import times
from geeflow import utils
import ml_collections as mlc
import pandas as pd

from tensorflow.io import gfile
import ee

EeAssetType = ee.ImageCollection | ee.Image | ee.FeatureCollection
ConfigDict = mlc.ConfigDict | dict[str, Any]


# TODO.
IC_SAMPLE_DATE_RANGES = ["Landsat7", "Landsat8", "Sentinel1", "Sentinel2",
                         "Alos", "ModisTerraVeg", "ModisSurfRefl", "ModisGPP",
                         "ModisET", "ModisBurn", "ModisFire", "FIRMS",
                         "WorldPop", "DynamicWorld"]
IC_SAMPLE = ["Nicfi", "CIESIN", "GHSPop", "NAIP"]
SAMPLE_ROI = ["NasaDem", "WorldCover", "FPP", "TPP", "Hansen", "LandCover",
              "WSF2015", "TreeCoverLossDueToFire", "CopDem", "FABDEM",
              "CustomImage", "Primary", "EnglandLidarDem"]
FC_GET = ["Countries"]

ALGO_MAP = {k: ee_algo.ic_sample_date_ranges for k in IC_SAMPLE_DATE_RANGES}
ALGO_MAP |= {k: ee_algo.ic_sample for k in IC_SAMPLE}
ALGO_MAP |= {k: ee_algo.sample_roi for k in SAMPLE_ROI}
ALGO_MAP |= {k: ee_algo.fc_get for k in FC_GET}
ALGO_MAP["CCDC"] = ee_algo.get_ccdc


@dataclasses.dataclass(frozen=True)
class Request:
  """A data request for Earth Engine."""
  image: ee.Image
  roi: ee.Geometry
  scale: float
  name: str
  crs: str | None = None
  fixed_band_names: bool = True


def save_df_to_file(df: pd.DataFrame, path: str) -> None:
  """Saves a pandas DataFrame as a csv or parquet file."""
  path_suffix = "." + path.split(".")[-1]
  tmp_fd, tmp_path = tempfile.mkstemp(suffix=path_suffix)
  if path.endswith(".csv"):
    with gfile.GFile(tmp_path, "w") as f:
      df.to_csv(f, index=False)
  elif path.endswith(".parquet"):
    with gfile.GFile(tmp_path, "wb") as f:
      parquet = df.to_parquet(index=False)
      f.write(parquet)
  else:
    raise ValueError("Not supported labels file format for file " + path)
  os.close(tmp_fd)

  basedir = os.path.dirname(path)
  os.umask(0o022); gfile.makedirs(basedir)
  gfile.Copy(tmp_path, path, overwrite=True)


def read_file_to_df(path: str) -> pd.DataFrame:
  """Reads a pandas DataFrame from a csv or parquet file."""
  if path.endswith(".csv"):
    with gfile.GFile(path, "r") as f:
      df = pd.read_csv(f)
  elif path.endswith(".parquet"):
    with gfile.GFile(path, "rb") as f:
      bytes_io = io.BytesIO(f.read())
      df = pd.read_parquet(bytes_io)
  else:
    raise ValueError("Not supported labels file format for file " + path)
  return df


def get_labels_df(config: mlc.ConfigDict, cache: bool = False) -> pd.DataFrame:
  """Returns pd.DataFrame with labels."""
  path = utils.cache_data(config.labels.path) if cache else config.labels.path
  df = read_file_to_df(path)
  if config.labels.get("num_max_samples"):
    df = df.iloc[:int(config.labels.num_max_samples)]
  return df


def pipeline_labels(
    config: mlc.ConfigDict, df: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
  """Returns a dict with extracted/processed label attributes."""
  if df is None:
    df = get_labels_df(config, config.labels.get("cache"))
  meta_keys = list(config.labels.meta_keys or df.columns)
  if set(meta_keys) - set(df.columns):
    raise ValueError(f"Some meta keys ({meta_keys}) are "
                     f"not in data columns ({df.columns}).")
  df = df[meta_keys]
  if "id" not in df.columns:
    df["id"] = range(len(df))
  return df.to_dict("records")


def pipeline_item_to_roi(
    config: mlc.ConfigDict,
    item: dict[str, Any],
    img_width_m: int | None = None,
    img_width_deg: float | None = None,
) -> ee.Geometry:
  """Converts an item to a Feature."""
  if config.labels.get("use_utm", True):
    img_width = img_width_m or config.labels.img_width_m
    max_cell_size = config.labels.max_cell_size_m
    img_size = img_width // max_cell_size
    if all(x in item for x in ["utm_x_min", "utm_x_max",
                               "utm_y_min", "utm_y_max", "utm_zone"]):
      assert item["utm_x_max"] - item["utm_x_min"] == img_width
      assert item["utm_y_max"] - item["utm_y_min"] == img_width
      roi = coords.UtmGridMapping(item["utm_zone"], max_cell_size, img_size,
                                  img_size, item["utm_x_min"],
                                  item["utm_y_min"])
    elif all(x in item for x in ["utm_x", "utm_y", "utm_zone"]):
      roi = coords.UtmGridMapping(item["utm_zone"],
                                  max_cell_size, img_size, img_size,
                                  item["utm_x"] - img_width / 2,
                                  item["utm_y"] - img_width / 2)
    else:
      roi = coords.UtmGridMapping.from_latlon_center(
          item["lat"], item["lon"], max_cell_size, img_size)
    roi = roi.to_ee(utm=True)
  else:
    roi = coords.get_lat_lon_roi(
        lat=item["lat"], lon=item["lon"],
        width_m=img_width_m or config.labels.get("img_width_m"),
        width_deg=img_width_deg or config.labels.get("img_width_deg"))

  return roi


def pipeline_item_to_rois(
    config: mlc.ConfigDict, item: dict[str, Any]
) -> dict[str, ee.Geometry]:
  """Converts an item to a Feature."""
  # `_default_roi` contains ROI for the default image size.
  # Sources with non-default image sizes are added separately.
  rois = {
      "_default_roi": pipeline_item_to_roi(config, item)
  }
  for k, v in config.sources.items():
    if v.get("img_width_m") or v.get("img_width_deg"):
      rois[k] = pipeline_item_to_roi(config, item,
                                     img_width_deg=v.get("img_width_deg"),
                                     img_width_m=v.get("img_width_m"))
  return rois


def pipeline_sources(config: mlc.ConfigDict) -> dict[str, EeAssetType]:
  """Returns a dict with source assets."""
  c = config.sources

  p = {}
  for k, v in c.items():
    logging.info("Setup source %s with %s", k, v)
    if isinstance(v.module, EeAssetType):
      # In config or colab, we can pass a GEE asset directly.
      sat = v.module
    else:
      sat = getattr(ee_data, v.module)(**v.get("kw", {}))
      out_method = v.get("out") or _infer_out_method(sat)
      sat = getattr(sat, out_method)
      if out_method not in ["ic", "im", "fc"]:
        sat = sat(**v.get("out_kw", {}))
    if v.get("select"):
      _validate_select_choices(k, sat, v.select)
      sat = sat.select(v.select)
    if v.get("cast"):
      sat = sat.cast(*v.cast)
    if v.get("filter_date", True) and hasattr(sat, "filterDate"):
      if "date_ranges" in v:
        start, end = times.outer_dates(v.date_ranges)
      else:
        start, end = v.get("start_date"), v.get("end_date")
      if start:  # start is required, end is optional.
        sat = sat.filterDate(start, end)
    p[k] = sat
  return p


def _infer_out_method(sat: EeAssetType) -> str:
  """Infers output method from the satellite object."""
  if isinstance(sat, ee_data.EeDataFC):
    return "fc"
  assert isinstance(sat, ee_data.EeData)
  try:
    sat.ic  # Check that it doesn't fail (it's cheap).  pylint: disable=pointless-statement
    return "ic"
  except ValueError:
    return "im"


def _validate_select_choices(source_name: str, source: EeAssetType,
                             select: Sequence[str]):
  """Validates that `select` is a subset of available property/band names."""
  if isinstance(source, ee.FeatureCollection):
    # Using toList makes sure we do not need to iterate over the whole FC,
    # which speeds ups things tremendously for large collections.
    if not source.toList(1).size().getInfo():
      # Maybe during filtering it got empty.
      raise ValueError(f"FeatureCollection already empty: {source_name}")
    available_names = set(source.first().propertyNames().getInfo())
    available_names.add(ee_algo.FEATURE_EXISTS_INTEGER_KEY)
  elif isinstance(source, ee.ImageCollection):
    available_names = source.first().bandNames().getInfo()
  elif isinstance(source, ee.Image):
    available_names = source.bandNames().getInfo()
  else:
    raise ValueError(f"Unsupported source type: {type(source)}")
  if isinstance(select, str):
    select = [select]
  if set(select) - set(available_names):
    raise ValueError(
        f"Some selected property/band names `{select}` are not "
        f"available in source `{source_name}`: `{available_names}`.\n"
        "For FeatureCollections created from shapefiles, this might be due to "
        "shapefile property names being clipped at 10 characters.")


def realign_geometry_scale(g: ee.Geometry, scale: int) -> ee.Geometry:
  """Clips geometry to closest grid points at given scale."""
  realigned = ee.Geometry.Polygon(g.coordinates().map(
      lambda x: ee.List(x).map(
          lambda y: ee.List(y).map(
              lambda z: ee.Number(z).divide(scale).round().multiply(scale)
              )
          )
      ), proj=g.projection(), geodesic=g.geodesic())
  return realigned


def get_algo_from_config(
    config: mlc.ConfigDict, source_name: str
) -> Callable[..., Any]:
  """Returns sampling algorithm function for a given source name."""
  cfg = config.sources[source_name]
  if not ("algo" in cfg or cfg.module in ALGO_MAP):
    raise ValueError(f"No valid algo for {source_name}")
  algo_fn = cfg.get("algo", ALGO_MAP.get(cfg.module, None))
  if isinstance(algo_fn, str):
    algo_fn = getattr(ee_algo, algo_fn)
  return algo_fn


def _get_dummy_im(asset: EeAssetType, cfg: ConfigDict) -> ee.Image:
  dummy_asset = asset
  if "dummy_image_id" in cfg:
    dummy_asset = dummy_asset.filterMetadata(
        "system:index", "equals", cfg["dummy_image_id"])
  dummy_im = ee_algo.get_dummy_image(dummy_asset.first())
  if ("sampling_kw" in cfg and
      cfg["sampling_kw"].get("reduce_fn", "") == "mean"):
    # Running "mean" reducer will make type float, so we need to convert
    # dummy_im to float as well.
    dummy_im = dummy_im.toFloat()
  return dummy_im


def get_requests_fn(
    config: mlc.ConfigDict, sources: ConfigDict
) -> Callable[..., tuple[Sequence[Request], dict[str, Any]]]:
  """Constructs a sampling function."""
  # Reference scale for geometry.
  ref_scale = config.labels.get("max_cell_size_m")

  def _request_fn(rois: dict[str, ee.Geometry], item: dict[str, Any]):
    requests = []
    metadata = dict(item)
    default_roi = rois["_default_roi"]
    if "lon" not in metadata:
      logging.warning("No lat/lon in meta_keys. Extracting from feature.")
      metadata["lon"] = default_roi.centroid(0.1).coordinates().get(0)
      metadata["lat"] = default_roi.centroid(0.1).coordinates().get(1)

    for name, asset in sources.items():
      cfg = config.sources.get(name)
      scale = cfg.get("scale") or config.labels.default_scale
      scalar = cfg.get("scalar", False)
      mask_value = cfg.get("mask_value", 0)
      kw = dict(cfg.get("sampling_kw", {}))

      roi = rois.get(name, default_roi)
      roi = roi.centroid(1) if scalar else roi

      # If scale and ref_scale grids don't overlap, realign to closest.
      # This is to to ensure equal gridded image sizes.
      # TODO: Find a way how to get constant shapes for lat/lon
      # based ROIs. Likely requires passing lat/lon scale to .reproject somehow.
      if (config.labels.get("use_utm", True) and not scalar and scale and
          (scale > ref_scale or ref_scale % scale != 0)):
        roi = realign_geometry_scale(roi, scale)

      if cfg.get("filter_date", True) and hasattr(asset, "filterDate"):
        # Usage example:
        # c.hr.start_date_fn = functools.partial(
        #     times.get_date_from_year, year_key="year", add_years=-3)
        # c.hr.end_date_fn = functools.partial(
        #     times.get_date_from_year, year_key="year", add_years=3)
        start, end = cfg.get("start_date_fn"), cfg.get("end_date_fn")
        if start:  # start is required, end is optional.
          start = start(item)
          if end: end = end(item)
          asset = asset.filterDate(start, end)

      algo = get_algo_from_config(config, name)
      algo_name = algo.__name__  # pytype: disable=attribute-error
      split_dates_into_separate_requests = cfg.get(
          "split_dates_into_separate_requests", False)
      asset_ims = []
      if algo_name == "get_ccdc":
        if (("year_selection" in (fmt := cfg.get("format_config", {}))) and
            (len(fmt["year_selection"]) != fmt["to"] - fmt["from"] + 1)):
          raise ValueError("`year_selection` mask should include all years "
                           "between `from` and `to`.")
        ccdc_im = algo(asset, roi, bands=cfg.get("select"), **kw)
        rename_fn = functools.partial(
            lambda b, n: ee.String(n + "_").cat(ee.String(b)), n=name
        )
        ccdc_im = ccdc_im.rename(ccdc_im.bandNames().map(rename_fn))
        asset_ims.append(ccdc_im)
      elif algo_name == "sample_roi":
        asset_ims.append(_add_mask_and_rename(asset, name, "", mask_value))
      elif algo_name in ["ic_sample", "ic_sample_date_ranges"]:
        # TODO: Cleanup the code so that we only have one 'limit'
        # setting.
        kw["limit"] = cfg.get("limit", kw.get("limit"))
        kw_name = "date_range" if algo_name == "ic_sample" else "date_ranges"
        value = cfg.get(kw_name)
        if fn := cfg.get(f"{kw_name}_fn"):
          if value:
            raise ValueError(f"Both {kw_name} and {kw_name}_fn are set.")
          # Usage example for `ic_sample_date_ranges`:
          # c.l7_v2.date_ranges_fn = functools.partial(
          #   times.get_date_ranges_from_year, year_key="year", n=4, months=3)
          value = fn(item)
        kw[kw_name] = value

        if algo_name == "ic_sample_date_ranges":
          kw["dummy_im"] = _get_dummy_im(asset, cfg)
          kw["bands"] = cfg.get("select_final")
          kw["scale"] = scale

        ims, ims_metadata = algo(roi, asset, **kw)
        metadata |= {f"{name}_{k}": v for k, v in ims_metadata.items()}
        # "#{t}" is used to concatenate along the time dimension.
        asset_ims += [
            _add_mask_and_rename(im, name, f"#{t}", mask_value)
            for t, im in enumerate(ims)
        ]
      elif algo_name == "ic_sample_reduced":
        if fn := cfg.get("date_range_fn"):
          # Usage example for `ic_sample_date_ranges`:
          # c.l7_v2.date_range_fn = functools.partial(
          #   times.get_date_ranges_from_year, year_key="year", n=4, months=3)
          kw["date_range"] = fn(item)

        dummy_im = _get_dummy_im(asset, cfg)
        im = algo(roi, asset, dummy_im=dummy_im, scale=scale, **kw)
        asset_ims.append(_add_mask_and_rename(im, name, "", mask_value))
      elif algo_name == "fc_get":
        metadata |= {f"{name}_{p}": v for p, v in
                     algo(asset, roi, cfg["select"]).items()}
      elif algo_name == "fc_to_image":
        im = algo(asset, roi, props=cfg.get("select"), **kw)
        asset_ims.append(_add_mask_and_rename(im, name, "", mask_value))
      else:
        raise ValueError(f"Unsupported algo: {algo_name}")

      if asset_ims:
        if split_dates_into_separate_requests:
          for im in asset_ims:
            requests.append(Request(im, roi, scale, name, cfg.get("crs")))
        else:
          # b/393557949 - This currently combines timesteps for the same asset
          # in a single ee.data.computePixels call. See ee_export_utils.py for
          # more details on how this could be better optimized.
          asset_im = functools.reduce(
              lambda im, im_acc: im_acc.addBands(im), asset_ims
          )
          # When we don't split dates into separate requests and we use ee_algo
          # without a fixed number of timesteps, the names of the bands changes
          # to include the timestep index, meaning that the names of the bands
          # are not fixed for each request.
          fixed_bands = algo != ee_algo.ic_sample
          requests.append(
              Request(asset_im, roi, scale, name, cfg.get("crs"), fixed_bands)
          )
    assert requests, f"No requests generated for {item}."
    return requests, metadata

  return _request_fn


def _add_mask_and_rename(
    im: ee.Image, name: str, idx: str, mask_value: int
) -> ee.Image:
  """Adds mask and renames bands."""
  out = im.rename(
      im.bandNames().map(lambda x: ee.String(f"{name}{idx}/").cat(x))
  )
  return out.unmask(mask_value, False).addBands(
      out.mask()
      .toUint8()
      .rename(
          im.bandNames().map(lambda x: ee.String(f"{name}_mask{idx}/").cat(x))
      )
  )
