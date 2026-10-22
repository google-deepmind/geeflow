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

"""Pipelines for processing EE Geo data."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import io
import os
import tempfile
from typing import Any
import warnings

from absl import logging
from geeflow import coords
from geeflow import ee_algo
from geeflow import ee_data
from geeflow import times
from geeflow import utils
from geeflow import validation
import ml_collections as mlc
import pandas as pd

from tensorflow.io import gfile
import ee

EeAssetType = ee.ImageCollection | ee.Image | ee.FeatureCollection
ConfigDict = mlc.ConfigDict | dict[str, Any]


# Maps deprecated sampling algorithm names to their new names.
_DEPRECATED_ALGO_NAMES: dict[str, str] = {
    "ic_sample": "ic_stack",
    "ic_sample_reduced": "ic_reduced",
    "ic_sample_date_ranges": "ic_temporal_composites",
    "sample_roi": "im_static",
    "get_ccdc": "ccdc_coefficients",
    "fc_get": "fc_properties",
    "fc_to_image": "fc_rasterize",
    "get_image_from_item": "im_from_item",
}


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
  meta_keys = list(config.labels.get("meta_keys", []) or df.columns)
  if set(meta_keys) - set(df.columns):
    raise ValueError(f"Some meta keys ({meta_keys}) are "
                     f"not in data columns ({df.columns}).")
  df = df[meta_keys].reset_index(drop=True)
  if "id" not in df.columns:
    df["id"] = range(len(df))
  return df.to_dict("records")


def utm_item_to_mapping(
    item: dict[str, Any], img_width_m: float, max_cell_size_m: float
) -> coords.UtmGridMapping:
  """Converts an item to a UTM geometry."""
  img_size = int(img_width_m // max_cell_size_m)
  if all(
      x in item
      for x in ["utm_x_min", "utm_x_max", "utm_y_min", "utm_y_max", "utm_zone"]
  ):
    assert item["utm_x_max"] - item["utm_x_min"] == img_width_m
    assert item["utm_y_max"] - item["utm_y_min"] == img_width_m
    return coords.UtmGridMapping(
        item["utm_zone"],
        max_cell_size_m,
        img_size,
        img_size,
        item["utm_x_min"],
        item["utm_y_min"],
    )
  if all(x in item for x in ["utm_x", "utm_y", "utm_zone"]):
    return coords.UtmGridMapping(
        item["utm_zone"],
        max_cell_size_m,
        img_size,
        img_size,
        item["utm_x"] - img_width_m / 2,
        item["utm_y"] - img_width_m / 2,
    )
  return coords.UtmGridMapping.from_latlon_center(
      item["lat"], item["lon"], max_cell_size_m, img_size
  )


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
    roi = utm_item_to_mapping(item, img_width, max_cell_size).to_ee(True)
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
  validation.validate_config(config)
  c = config.sources

  p = {}
  for k, v in c.items():
    logging.info("Setup source %s with %s", k, v)
    if isinstance(v.module, EeAssetType):
      # In config or colab, we can pass a GEE asset directly.
      sat = v.module
    else:
      source_cls = validation.resolve_source_class(v.module)
      sat = source_cls(**v.get("kw", {}))
      out_method = v.get("out") or _infer_out_method(sat)
      sat = getattr(sat, out_method)
      if out_method not in ["ic", "im", "fc"]:
        sat = sat(**v.get("out_kw", {}))
    if v.get("select"):
      if v.algo not in ("im_from_item", "get_image_from_item",
                        ee_algo.get_image_from_item):
        # TODO: For this algo, we do not enforce specifying the
        # image collection, so validating the choices would raise an error.
        # Consider enforcing the specification of the image collection.
        _validate_select_choices(k, sat, v.select)
        sat = sat.select(v.select)
    if v.get("cast"):
      sat = sat.cast(*v.cast)  # pyrefly: ignore[missing-attribute]
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
    available_names = set(source.first().propertyNames().getInfo())  # pyrefly: ignore[bad-argument-type]
    available_names.add(ee_algo.FEATURE_EXISTS_INTEGER_KEY)
  elif isinstance(source, ee.ImageCollection):
    available_names = source.first().bandNames().getInfo()
  elif isinstance(source, ee.Image):
    available_names = source.bandNames().getInfo()
  else:
    raise ValueError(f"Unsupported source type: {type(source)}")
  if isinstance(select, str):
    select = [select]
  if set(select) - set(available_names):  # pyrefly: ignore[bad-argument-type]
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
  if "algo" not in cfg or cfg.algo is None:
    raise ValueError(f"No valid algo for {source_name}")
  algo_fn = cfg.algo
  if isinstance(algo_fn, str):
    if algo_fn in _DEPRECATED_ALGO_NAMES:
      new_name = _DEPRECATED_ALGO_NAMES[algo_fn]
      warnings.warn(
          f"Algorithm name '{algo_fn}' is deprecated for source "
          f"'{source_name}'. Use '{new_name}' instead.",
          DeprecationWarning,
          stacklevel=2,
      )
    algo_fn = getattr(ee_algo, algo_fn)
  return algo_fn


def _get_dummy_im(asset: EeAssetType, cfg: ConfigDict,
                  roi: ee.Geometry) -> ee.Image:
  """Returns an empty dummy image for the given asset."""
  dummy_asset = asset
  if "dummy_image_id" in cfg:
    dummy_asset = dummy_asset.filterMetadata(  # pyrefly: ignore[missing-attribute]
        "system:index", "equals", cfg["dummy_image_id"])
  dummy_im = ee_algo.get_dummy_image(dummy_asset.first())  # pyrefly: ignore[bad-argument-type, missing-argument]
  if ("sampling_kw" in cfg and
      cfg["sampling_kw"].get("reduce_fn", "") == "mean"):
    # Running "mean" reducer will make type float, so we need to convert
    # dummy_im to float as well.
    dummy_im = dummy_im.toFloat()
  if cfg.get("select_final", None):
    dummy_im = dummy_im.select(cfg["select_final"])
  # This will avoid "Reprojection output too large (XXXXxXXXX pixels)." error.
  dummy_im = dummy_im.clip(roi.centroid(0.1))
  dummy_im = dummy_im.reproject(roi.projection())
  return dummy_im


def _handle_get_ccdc(
    algo: Callable[..., Any],
    asset: EeAssetType,
    roi: ee.Geometry,
    name: str,
    cfg: ConfigDict,
    kw: dict[str, Any],
    **unused_kwargs,
) -> tuple[list[ee.Image], dict[str, Any]]:
  """Handles the 'get_ccdc' sampling algorithm."""
  if ("year_selection" in (fmt := cfg.get("format_config", {}))) and (
      len(fmt["year_selection"]) != fmt["to"] - fmt["from"] + 1
  ):
    raise ValueError(
        "`year_selection` mask should include all years "
        "between `from` and `to`."
    )
  ccdc_im = algo(asset, roi, bands=cfg.get("select"), **kw)
  rename_fn = functools.partial(
      lambda b, n: ee.String(n + "_").cat(ee.String(b)), n=name
  )
  ccdc_im = ccdc_im.rename(ccdc_im.bandNames().map(rename_fn))
  return [ccdc_im], {}


def _handle_sample_roi(
    asset: EeAssetType,
    name: str,
    mask_value: int,
    **unused_kwargs,
) -> tuple[list[ee.Image], dict[str, Any]]:
  """Handles the 'sample_roi' sampling algorithm."""
  return [_add_mask_and_rename(asset, name, "", mask_value)], {}  # pyrefly: ignore[bad-argument-type]


def resolve_date_range(
    cfg: ConfigDict, item: dict[str, Any]
) -> tuple[str, str] | None:
  """Resolves `(start_date, end_date)` for an item from `cfg`."""
  value = cfg.get("date_range") or cfg.get("sampling_kw", {}).get("date_range")
  if fn := cfg.get("date_range_fn"):
    if value:
      raise ValueError("Both date_range & date_range_fn are set.")
    value = fn(item)
  return tuple(value) if value is not None else None


def resolve_date_ranges(
    cfg: ConfigDict, item: dict[str, Any]
) -> tuple[tuple[Any, ...], ...] | None:
  """Resolves `date_ranges` sequence for an item from `cfg`."""
  value = cfg.get("date_ranges") or cfg.get("sampling_kw", {}).get(
      "date_ranges"
  )
  if fn := cfg.get("date_ranges_fn"):
    if value:
      raise ValueError("Both date_ranges & date_ranges_fn are set.")
    value = fn(item)
  return tuple(tuple(e) for e in value) if value is not None else None


def _handle_ic_sample(
    algo: Callable[..., Any],
    algo_name: str,
    asset: EeAssetType,
    roi: ee.Geometry,
    name: str,
    cfg: ConfigDict,
    scale: float,
    mask_value: int,
    item: dict[str, Any],
    kw: dict[str, Any],
    processed_ee_sources: dict[str, list[ee.Image]],
    **unused_kwargs,
) -> tuple[list[ee.Image], dict[str, Any]]:
  """Handles the 'ic_stack' and 'ic_temporal_composites' algorithms."""
  if temporal_anchor := cfg.get("temporal_anchor") or kw.get("temporal_anchor"):
    if temporal_anchor not in processed_ee_sources:
      raise ValueError(
          f"Temporal anchor {temporal_anchor} not in "
          f"{processed_ee_sources.keys()}"
      )
    anchor_im = processed_ee_sources[temporal_anchor]
    if isinstance(anchor_im, list):
      anchor_im = anchor_im[0] if anchor_im else None
    if anchor_im is not None:
      kw["temporal_anchor_ts"] = ee.Image(anchor_im).get(
          "system:time_start"
      )
    kw["time_window_days"] = cfg.get(
        "time_window_days", kw.get("time_window_days")
    )

  # TODO: Cleanup the code so that we only have one `limit` setting.
  kw["limit"] = cfg.get("limit", kw.get("limit"))

  if algo_name in ("ic_sample", "ic_stack"):
    if (dr := resolve_date_range(cfg, item)) is not None:
      kw["date_range"] = dr
  elif (drs := resolve_date_ranges(cfg, item)) is not None:
    # Usage example for `ic_sample_date_ranges`:
    # c.l7_v2.date_ranges_fn = functools.partial(
    #   times.get_date_ranges_from_year, year_key="year", n=4, months=3)
    kw["date_ranges"] = drs

  if cfg.get("select_final"):
    kw["bands"] = cfg.get("select_final")
  if algo_name in ("ic_sample_date_ranges", "ic_temporal_composites"):
    kw["dummy_im"] = _get_dummy_im(asset, cfg, roi)
    kw["scale"] = scale

  if cfg.get("match_season"):
    kw["match_season_date"] = item.get("timestamp")
    kw["match_season_window_days"] = cfg.get(
        "match_season_window_days", kw.get("match_season_window_days", 45)
    )

  ims, ims_metadata = algo(roi, asset, **kw)
  metadata_updates = {f"{name}_{k}": v for k, v in ims_metadata.items()}
  # "#{t}" is used to concatenate along the time dimension.
  asset_ims = [
      _add_mask_and_rename(im, name, f"#{t}", mask_value)
      for t, im in enumerate(ims)
  ]
  return asset_ims, metadata_updates


def _handle_ic_sample_reduced(
    algo: Callable[..., Any],
    asset: EeAssetType,
    roi: ee.Geometry,
    name: str,
    cfg: ConfigDict,
    scale: float,
    mask_value: int,
    item: dict[str, Any],
    kw: dict[str, Any],
    **unused_kwargs,
) -> tuple[list[ee.Image], dict[str, Any]]:
  """Handles the 'ic_sample_reduced' sampling algorithm."""
  if (dr := resolve_date_range(cfg, item)) is not None:
    # Usage example for `ic_sample_date_ranges`:
    # c.l7_v2.date_range_fn = functools.partial(
    #   times.get_date_ranges_from_year, year_key="year", n=4, months=3)
    kw["date_range"] = dr

  dummy_im = _get_dummy_im(asset, cfg, roi)
  im = algo(roi, asset, dummy_im=dummy_im, scale=scale, **kw)
  return [_add_mask_and_rename(im, name, "", mask_value)], {}


def _handle_fc_get(
    algo: Callable[..., Any],
    asset: EeAssetType,
    roi: ee.Geometry,
    name: str,
    cfg: ConfigDict,
    **unused_kwargs,
) -> tuple[list[ee.Image], dict[str, Any]]:
  """Handles the 'fc_get' sampling algorithm."""
  metadata_updates = {
      f"{name}_{p}": v for p, v in algo(asset, roi, cfg["select"]).items()
  }
  return [], metadata_updates


def _handle_fc_to_image(
    algo: Callable[..., Any],
    asset: EeAssetType,
    roi: ee.Geometry,
    name: str,
    cfg: ConfigDict,
    mask_value: int,
    kw: dict[str, Any],
    **unused_kwargs,
) -> tuple[list[ee.Image], dict[str, Any]]:
  """Handles the 'fc_to_image' sampling algorithm."""
  props = (cfg.get("select")
           if cfg.get("select_final") is None else cfg.get("select_final"))
  im = algo(asset, roi, props=props, **kw)
  return [_add_mask_and_rename(im, name, "", mask_value)], {}


def _handle_get_image_from_item(
    name: str,
    cfg: ConfigDict,
    mask_value: int,
    item: dict[str, Any],
    kw: dict[str, Any] | None = None,
    **unused_kwargs,
) -> tuple[list[ee.Image], dict[str, Any]]:
  """Handles the 'get_image_from_item' sampling algorithm."""
  select = cfg.get("select", None)
  cloud_mask_fn = (kw or {}).get("cloud_mask_fn", None)
  full_im = ee.Image(item[cfg["asset_id_key"]])
  if cloud_mask_fn:
    full_im = full_im.updateMask(cloud_mask_fn(full_im))
  im = full_im.select(select) if select else full_im
  return [_add_mask_and_rename(im, name, "", mask_value)], {}


_ALGO_HANDLERS: dict[
    str,
    Callable[..., tuple[list[ee.Image], dict[str, Any]]],
] = {
    # Canonical names.
    "ic_stack": _handle_ic_sample,
    "ic_reduced": _handle_ic_sample_reduced,
    "ic_temporal_composites": _handle_ic_sample,
    "im_from_item": _handle_get_image_from_item,
    "im_static": _handle_sample_roi,
    "fc_properties": _handle_fc_get,
    "fc_rasterize": _handle_fc_to_image,
    "ccdc_coefficients": _handle_get_ccdc,
    # Deprecated aliases (kept for backwards compatibility).
    "get_ccdc": _handle_get_ccdc,
    "sample_roi": _handle_sample_roi,
    "ic_sample": _handle_ic_sample,
    "ic_sample_date_ranges": _handle_ic_sample,
    "ic_sample_reduced": _handle_ic_sample_reduced,
    "fc_get": _handle_fc_get,
    "fc_to_image": _handle_fc_to_image,
    "get_image_from_item": _handle_get_image_from_item,
}


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

    processed_ee_sources = {}
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
      algo_str = cfg.algo if isinstance(cfg.algo, str) else algo.__name__
      algo_name = _DEPRECATED_ALGO_NAMES.get(algo_str, algo_str)
      split_dates_into_separate_requests = cfg.get(
          "split_dates_into_separate_requests", False)

      if (handler := _ALGO_HANDLERS.get(algo_name)) is None:
        raise ValueError(f"Unsupported algo: {algo_name}")

      asset_ims, metadata_updates = handler(
          algo=algo,
          algo_name=algo_name,
          asset=asset,
          roi=roi,
          name=name,
          cfg=cfg,
          scale=scale,
          mask_value=mask_value,
          item=item,
          kw=kw,
          processed_ee_sources=processed_ee_sources,
      )
      processed_ee_sources[name] = asset_ims
      metadata |= metadata_updates

      if asset_ims:
        if split_dates_into_separate_requests:
          for im in asset_ims:
            requests.append(Request(im, roi, scale, name, cfg.get("crs")))
        else:
          # Note: This currently combines timesteps for the same asset
          # in a single ee.data.computePixels call. See ee_export_utils.py for
          # more details on how this could be better optimized.
          asset_im = functools.reduce(
              lambda im, im_acc: im_acc.addBands(im), asset_ims
          )
          # When we don't split dates into separate requests and we use ee_algo
          # without a fixed number of timesteps, the names of the bands changes
          # to include the timestep index, meaning that the names of the bands
          # are not fixed for each request.
          fixed_bands = algo_name != "ic_stack"
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
