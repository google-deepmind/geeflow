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

r"""Utility functions to export inference result into EE."""

import collections
import dataclasses
import math
import operator
import os
import tempfile
import time
from typing import Any, Callable, Iterable, Optional
import uuid

from absl import logging
import einops
from geeflow import coords
from geeflow import ee_export_utils
from geeflow import pipelines
from geeflow import utils
import geopandas as gpd
import numpy as np
try:
  from osgeo import gdal
  from osgeo import gdalconst
  from osgeo import ogr
  from osgeo import osr
except ImportError:
  gdal = None
  gdalconst = None
  ogr = None
  osr = None
import pandas as pd

from tensorflow.io import gfile
import ee

UTM_ZONE_KEYWORD = "{utm_zone}"


@dataclasses.dataclass(frozen=True)
class PlotData:
  """The data for a given plot."""

  @property
  def key(self) -> tuple[str, int, int]:
    return self.utm_zone, self.x_split, self.y_split

  @property
  def coords(self) -> tuple[int, int, int, int]:
    return self.utm_x, self.utm_y, self.dx, self.dy

  utm_zone: str
  x_split: int
  y_split: int
  utm_x: int
  utm_y: int
  dx: int
  dy: int
  preds: np.ndarray | dict[str, np.ndarray]
  mask: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class SplitData:
  """The data for a given split."""

  utm_zone: str
  x_split: int
  y_split: int
  np_data: np.ndarray
  np_data_mask: np.ndarray
  geotransform: tuple[float, float, float, float, float, float]
  epsg: str
  mask: np.ndarray | None = None


def get_numpy_data(
    inference_data: dict[tuple[int, int, int, int], list[np.ndarray]],
    lat: float,
    lon: float,
    lat_end: float,
    lon_end: float,
    x_split: int,
    y_split: int,
    num_splits: int,
    grid_spacing_m: float,
    cell_size: float,
    plot_size_m: float,
    border_mode: str,
    normalize: bool,
    clip_min: float | None,
    clip_max: float | None,
    discretization_factor: float | None,
    add_argmax: str,
    shift_argmax: int,
    output_type: str,
    inference_masks: (
        dict[tuple[int, int, int, int], list[np.ndarray | None]] | None
    ) = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns numpy array representing inference results."""

  inference_masks = inference_masks or {}
  if inference_masks:
    if set(inference_masks.keys()) != set(inference_data.keys()):
      raise ValueError(
          "Inference masks must have the same keys as inference data."
      )
    for key in inference_data:
      if len(inference_data[key]) != len(inference_masks[key]):
        raise ValueError(
            f"List length mismatch for key {key}: len(inference_data) is"
            f" {len(inference_data[key])} but len(inference_masks) is"
            f" {len(inference_masks[key])}."
        )

  x_splits, y_splits = get_info(
      lat,
      lon,
      lat_end,
      lon_end,
      grid_spacing_m,
      cell_size,
      plot_size_m,
      num_splits,
  )
  tile_x_slice = slice(x_splits[x_split], x_splits[x_split + 1])
  tile_y_slice = slice(y_splits[y_split], y_splits[y_split + 1])

  # Get one example to determine num_bands and dtype.
  preds = next(iter(inference_data.values()))[0]
  num_bands = 1
  if len(preds.shape) == 3 or len(preds.shape) == 1:
    num_bands = preds.shape[-1]
  npdtype = preds.dtype
  if preds.dtype in [np.float32, np.float64] or border_mode in [
      "uniform_avg",
      "l2_dist_avg",
      "l1_border_dist_avg",
  ]:
    npdtype = np.float32

  # Compute boundaries for the current tile.
  # Use round() instead of int() to avoid floating-point truncation errors
  # (e.g. 23.99999 would become 23 with int(), but correctly becomes 24 with
  # round()).
  tile_width = round((tile_x_slice.stop - tile_x_slice.start) / cell_size)
  tile_height = round((tile_y_slice.stop - tile_y_slice.start) / cell_size)
  # This is the global accumulator/aggregator.
  tile = np.zeros((tile_height, tile_width, num_bands), dtype=npdtype)
  # This is global accumulated weights for normalization.
  tile_weights = np.zeros((tile_height, tile_width, 1), dtype=npdtype)
  # Fixed weights for border averaging.
  w_mask = get_border_averaging_weights(border_mode, preds.shape[0], npdtype)

  for (utm_x, utm_y, dx, dy), predictions_list in inference_data.items():
    masks_list = inference_masks.get((utm_x, utm_y, dx, dy), None)

    # Calculate the slice of the predictions within the split tile.
    plot_shape = predictions_list[0].shape
    x_start = utm_x + dx * cell_size - plot_size_m // 2
    preds_x_slice = slice(x_start, x_start + plot_shape[1] * cell_size)
    y_end = utm_y - dy * cell_size + plot_size_m // 2
    preds_y_slice = slice(y_end - plot_shape[0] * cell_size, y_end)

    # Check if the plot slice is within the tile slice, if not continue.
    if (
        preds_y_slice.stop <= tile_y_slice.start
        or preds_y_slice.start >= tile_y_slice.stop
        or preds_x_slice.stop <= tile_x_slice.start
        or preds_x_slice.start >= tile_x_slice.stop
    ):
      continue

    for i, preds in enumerate(predictions_list):
      preds = preds.astype(npdtype)
      preds = preds[::-1]  # Y-axis is upside down.

      # Get mask for this prediction if available.
      mask = None
      if masks_list is not None and i < len(masks_list):
        m = masks_list[i]
        if m is not None:
          if m.ndim not in (2, 3) or (m.ndim == 3 and m.shape[-1] != 1):
            raise ValueError(
                "Expected 2D spatial mask of shape (H, W) or single-channel"
                f" (H, W, 1), but got shape {m.shape}."
            )
          mask = np.atleast_3d(m[::-1]) > 0  # Y-axis is upside down.

      # Update the split tile with the model corresponding model predictions.
      shared_args = (
          preds_y_slice,
          preds_x_slice,
          tile_x_slice,
          tile_y_slice,
          cell_size,
      )
      replace_fn = lambda a, b: b
      if border_mode in [
          "uniform_avg",
          "l2_dist_avg",
          "l1_border_dist_avg",
          "any",
      ]:
        if border_mode == "any":
          sliced_update(tile, *shared_args, preds, op=replace_fn, mask=mask)
          sliced_update(
              tile_weights, *shared_args, w_mask, op=replace_fn, mask=mask
          )
        else:
          sliced_update(
              tile, *shared_args, preds * w_mask, op=operator.add, mask=mask
          )
          sliced_update(
              tile_weights, *shared_args, w_mask, op=operator.add, mask=mask
          )
      else:
        sliced_update(tile, *shared_args, preds, op=replace_fn, mask=mask)
        sliced_update(
            tile_weights, *shared_args, w_mask, op=replace_fn, mask=mask
        )

  if border_mode in ["uniform_avg", "l2_dist_avg", "l1_border_dist_avg"]:
    tile = tile / np.maximum(tile_weights, 1e-10)

  if normalize:
    tile = tile / np.maximum(np.sum(tile, axis=-1, keepdims=True), 1e-10)

  if clip_min is not None and clip_max is not None:
    tile = np.clip(tile, clip_min, clip_max)
  if discretization_factor is not None:
    tile = tile * discretization_factor

  if add_argmax:
    # NOTE: The following two lines that are commented out, are used to generate
    # the argmax for ForTy models. A special handling is necessary because we do
    # not read (and do not export) probabilities for "other" class (to save
    # resources). That's why it has to be added here explicitly.
    # tile = tile.astype(np.uint8)
    # tile = np.concatenate([tile, 250 - np.sum(tile, axis=-1, keepdims=True)],
    #                       axis=-1)
    argmax_tile = np.argmax(tile, axis=-1, keepdims=True)
    if add_argmax == "add":
      tile = np.concatenate(
          [(argmax_tile + shift_argmax).astype(tile.dtype), tile], axis=-1
      )
    elif add_argmax == "only":
      tile = argmax_tile + shift_argmax
    else:
      raise ValueError(f"Unsupported add_argmax mode: {add_argmax}")

  if output_type == "int16":
    tile = tile.astype(np.int16)
  if output_type == "int8":
    tile = tile.astype(np.int8)
  if output_type == "uint8":
    tile = tile.astype(np.uint8)

  tile_mask = tile_weights[..., 0] > 0
  return tile, tile_mask


def get_linspace(start: float, end: float, cell_size: float, num_splits: int):
  """Split the [start, end] range in meters into num_splits sub-ranges."""
  # The formula below has an extra "16" factor. This is necessary to make
  # each subsplit divisible by 16. Which in turn is needed to avoid a bug
  # in EE that creates strange display artifacts (vertical and horizontal
  # stripes).
  return (
      np.linspace(
          int(start // cell_size // 16),
          int(math.ceil(int(end // cell_size) / 16)),
          num_splits + 1,
          dtype=int,
      )
      * cell_size
      * 16
  )


def get_info(
    lat: float,
    lon: float,
    lat_end: float,
    lon_end: float,
    grid_spacing_m: float,
    cell_size: float,
    plot_size_m: float,
    num_splits: int,
):
  """Return information based on image size."""
  x_num, y_num = utils.get_utm_grid_size(
      lat, lon, lat_end, lon_end, grid_spacing_m
  )
  # Image width in pixels.
  image_width = grid_spacing_m / cell_size
  if abs(image_width - round(image_width)) > 1e-10:
    raise ValueError("grid_spacing_m / cell_size must be an exact integer.")
  # Skip overlapping border.
  border_skip = (plot_size_m - grid_spacing_m) / (2 * cell_size)
  if abs(border_skip - round(border_skip)) > 1e-10:
    raise ValueError(
        "(plot_size_m - grid_spacing_m) / (2 * cell_size) must be an exact"
        " integer."
    )
  roi = coords.UtmGridMapping.from_latlon_center(
      (lat + lat_end) / 2, (lon + lon_end) / 2, grid_spacing_m, x_num, y_num
  )
  x_splits = get_linspace(roi.bbox[0], roi.bbox[2], cell_size, num_splits)
  y_splits = get_linspace(roi.bbox[1], roi.bbox[3], cell_size, num_splits)
  return x_splits, y_splits


def get_ee_asset_id(ee_project: str, ee_asset_id: str) -> str:
  return f"projects/{ee_project.split(':')[0]}/assets/{ee_asset_id}"


def get_border_averaging_weights(mode: str, size: int, dtype):
  """Returns weights for border averaging."""
  if mode == "l2_dist_avg":
    w = np.array([
        [[math.hypot(i - size / 2, j - size / 2)] for j in range(size)]
        for i in range(size)
    ])
    # We normalize by the minimal distance from the center to the border.
    w_max = math.hypot(0, size / 2)
    w = np.maximum(1 - w / w_max, 0.001)
    return w
  if mode == "l1_border_dist_avg":
    assert size % 2 == 0, "size expected to be even"
    # TODO - Generalize border handling.
    border = size // 4
    assert border <= size // 2, "not tested for border > size//2"
    m = np.tile(
        np.array(list(range(size // 2 - 1, -1, -1)) + list(range(size // 2)))[
            None
        ],
        (size, 1),
    )
    m = np.maximum(m, m.T)  # L1 (Mahalanobis) distance from center.
    w = m.max() - m + 1  # Weights starting at 1.
    if border:
      w = np.minimum(w, border + 1)
    w = w / w.max()  # Normalize to max weight 1.
    w = np.expand_dims(w, -1)
    return w
  return np.ones((size, size, 1), dtype=dtype)


def sliced_update(
    tile: np.ndarray,
    preds_y_slice: slice,
    preds_x_slice: slice,
    tile_x_slice: slice,
    tile_y_slice: slice,
    cell_size: float,
    preds: np.ndarray,
    op: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mask: np.ndarray | None = None,
):
  """Updates an inference tile with model predictions using a specified operation.

  Args:
    tile: The large tile where model predictions are added.
    preds_y_slice: The global y-slice corresponding to the predictions.
    preds_x_slice: The global x-slice corresponding to the predictions.
    tile_x_slice: The global x-slice of the current tile.
    tile_y_slice: The global y-slice of the current tile.
    cell_size: The size of the cells in the tile.
    preds: The numpy array containing the values to update with.
    op: A function that takes two arguments (the current tile slice and the
      predictions slice) and returns the updated tile slice.
    mask: Optional boolean validity mask matching the spatial dimensions of
      `preds`. When provided, only pixels where `mask` is True are updated.
  """
  # Calculate the slice within the tile.
  y_start_local = round((preds_y_slice.start - tile_y_slice.start) / cell_size)
  y_end_local = round((preds_y_slice.stop - tile_y_slice.start) / cell_size)
  x_start_local = round((preds_x_slice.start - tile_x_slice.start) / cell_size)
  x_end_local = round((preds_x_slice.stop - tile_x_slice.start) / cell_size)

  # Determine the corresponding slice of predictions.
  p_y_start, p_y_end = 0, preds.shape[0]
  p_x_start, p_x_end = 0, preds.shape[1]

  if y_start_local < 0:
    # The slice starts before the tile. Adjust predictions slice start.
    p_y_start = -y_start_local
    y_start_local = 0
  if y_end_local > tile.shape[0]:
    # The slice ends after the tile. Adjust predictions slice end.
    p_y_end -= y_end_local - tile.shape[0]
    y_end_local = tile.shape[0]

  if x_start_local < 0:
    # The slice starts before the tile. Adjust predictions slice start.
    p_x_start = -x_start_local
    x_start_local = 0
  if x_end_local > tile.shape[1]:
    # The slice ends after the tile. Adjust predictions slice end.
    p_x_end -= x_end_local - tile.shape[1]
    x_end_local = tile.shape[1]

  # If the calculated slices result in no overlap, return early.
  if (
      y_start_local >= y_end_local
      or x_start_local >= x_end_local
      or p_y_start >= p_y_end
      or p_x_start >= p_x_end
  ):
    return

  # Create the final local slices for the tile.
  y_slice_local = slice(y_start_local, y_end_local)
  x_slice_local = slice(x_start_local, x_end_local)

  # Update the tile with the predictions.
  preds_sliced = preds[p_y_start:p_y_end, p_x_start:p_x_end]
  updated = op(tile[y_slice_local, x_slice_local], preds_sliced)
  if mask is not None:
    mask_sliced = mask[p_y_start:p_y_end, p_x_start:p_x_end]
    updated = np.where(mask_sliced, updated, tile[y_slice_local, x_slice_local])
  tile[y_slice_local, x_slice_local] = updated


def create_tiff_options(numpy_array: np.ndarray, write_cogs: bool) -> list[str]:
  """Returns options for the maximum lossless compression.

  Also handles the special case required for signed int8 data.

  Note that COMPRESS=LZW sometimes exhibits worst-case behavior for data
  with >1 byte per pixel: resulting file sizes can exceed raw file size.

  See:
  https://gdal.org/drivers/raster/gtiff.html#creation-options
  https://numpy.org/doc/stable/reference/arrays.dtypes.html

  Args:
    numpy_array: Array whose type will be used to deturmine the best options.
    write_cogs: Whether to write the tiff in COG format.

  Returns:
    A list of strings to be passed as options to Create or CreateCopy.

  Raises:
    InputError if the numpy_array is not an integer or float type.
  """
  # All tifs should use these standard options.
  options = ["BIGTIFF=YES"]
  dtype = numpy_array.dtype

  # Force GDAL to create a signed int8 image.
  if dtype == np.dtype(np.int8):
    options.append("PIXELTYPE=SIGNEDBYTE")

  if np.issubdtype(dtype, np.integer):
    options.append("PREDICTOR=2")
  elif np.issubdtype(dtype, np.floating):
    options.append("PREDICTOR=3")

  if write_cogs:
    options += [
        "INTERLEAVE=BAND",
        "COMPRESS=ZSTD",
        "LEVEL=9",
    ]
  else:
    options += ["COMPRESS=DEFLATE", "ZLEVEL=9"]

  return options


_VSI_CHUNK_BYTES = 4 * 1024 * 1024


def _stream_vsimem_to_gfile(vsi_filename: str, target_filename: str) -> int:
  """Streams a GDAL /vsimem/ file to target_filename and unlinks it."""
  vsi_file = gdal.VSIFOpenL(vsi_filename, "rb")
  if vsi_file is None:
    gdal.Unlink(vsi_filename)
    raise RuntimeError(
        f"Failed to open {vsi_filename}: {gdal.GetLastErrorMsg()}"
    )
  total_bytes = 0
  try:
    with gfile.GFile(target_filename, "wb") as out_file:
      while chunk := gdal.VSIFReadL(1, _VSI_CHUNK_BYTES, vsi_file):
        out_file.write(chunk)
        total_bytes += len(chunk)
  finally:
    gdal.VSIFCloseL(vsi_file)
    gdal.Unlink(vsi_filename)

  logging.info(
      "Wrote %d bytes from /vsimem/ to %s", total_bytes, target_filename
  )
  return total_bytes


def write_tif(
    np_data: np.ndarray,
    np_data_mask: np.ndarray | None,
    geotransform: tuple[float, float, float, float, float, float],
    epsg: str,
    write_cogs: bool,
    resampling: str = "AVERAGE",
    output_filename: str | None = None,
) -> str:
  """Writes a raster array as a GeoTIFF or Cloud Optimized GeoTIFF (COG).

  Args:
    np_data: Raster array of shape `(H, W)` or `(H, W, C)`.
    np_data_mask: Optional 2D boolean mask of shape `(H, W)` where False marks
      nodata pixels.
    geotransform: Six-element GDAL affine geotransform tuple.
    epsg: Projection string (WKT or `EPSG:<code>`).
    write_cogs: Whether to write a Cloud Optimized GeoTIFF (COG) with overviews
      instead of a standard GeoTIFF.
    resampling: Overview resampling algorithm passed to `BuildOverviews`.
    output_filename: Optional destination path (local or cloud storage). When
      `None`, writes to a newly generated path in the system temp directory.

  Returns:
    The path to the written GeoTIFF/COG file (`output_filename` if provided, or
    the generated temporary file path).

  Raises:
    ValueError: If `np_data_mask` is provided and is not 2D.
    RuntimeError: If GDAL fails to serialize or read the `/vsimem/` dataset.
  """
  if np_data_mask is not None and np_data_mask.ndim != 2:
    raise ValueError(f"np_data_mask must be 2D. Got {np_data_mask.shape}")
  if output_filename is None:
    target_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.tif")
  else:
    target_filename = output_filename
  # Serialize in GDAL virtual memory (/vsimem/) rather than local /tmp so
  # high-parallelism workers do not exhaust container scratch space.
  vsi_filename = f"/vsimem/{uuid.uuid4()}.tif"

  np_data = np.atleast_3d(np_data)

  dtype = gdalconst.GDT_Byte
  if isinstance(np_data.flat[0], np.float32):
    dtype = gdalconst.GDT_Float32
  elif isinstance(np_data.flat[0], np.int16):
    dtype = gdalconst.GDT_Int16
  no_data_value = 255 if dtype == gdalconst.GDT_Byte else -1
  driver_name = "MEM" if write_cogs else "GTiff"
  driver_path = "" if write_cogs else vsi_filename
  driver_opts = [] if write_cogs else create_tiff_options(np_data, False)
  try:
    ds = gdal.GetDriverByName(driver_name).Create(
        driver_path,
        np_data.shape[1],
        np_data.shape[0],
        np_data.shape[2],
        dtype,
        options=driver_opts,
    )
    if ds is None:
      raise RuntimeError(
          f"GDAL {driver_name}.Create failed: {gdal.GetLastErrorMsg()}"
      )
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(epsg)
    for i in range(np_data.shape[-1]):
      rb = ds.GetRasterBand(i + 1)
      data = np_data[..., i]
      if np_data_mask is not None:
        data = np.where(np_data_mask, data, no_data_value)
      rb.SetNoDataValue(no_data_value)
      rb.WriteArray(data)
    if write_cogs:
      # Overviews must exist on the source dataset before COG.CreateCopy runs so
      # the COG driver copies them into the tiled layout.
      ds.BuildOverviews(resampling, [2, 4, 8, 16, 32, 64])
      ds.FlushCache()
      ds2 = gdal.GetDriverByName("COG").CreateCopy(
          vsi_filename, ds, options=create_tiff_options(np_data, True)
      )
      if ds2 is None:
        raise RuntimeError(
            f"GDAL CreateCopy failed for {vsi_filename}:"
            f" {gdal.GetLastErrorMsg()}"
        )
      ds2.FlushCache()
      ds2 = None  # pylint: disable=unused-variable
    else:
      ds.FlushCache()
    ds = None  # pylint: disable=unused-variable
  except Exception:
    gdal.Unlink(vsi_filename)
    raise

  _stream_vsimem_to_gfile(vsi_filename, target_filename)
  return target_filename


def vectorize_raster(
    raster: np.ndarray,
    geotransform: tuple[float, float, float, float, float, float],
    epsg_code: int,
) -> gpd.GeoDataFrame | None:
  """Converts a binary raster array to geolocalised polygons."""

  # Preliminary checks.
  if raster.ndim != 2:
    raise ValueError(f"Raster must be 2D, got {raster.shape}.")
  if not np.all(np.isin(raster, [0, 1])):
    raise ValueError("Raster must be binary.")

  # Create in-memory raster.
  mem_driver = gdal.GetDriverByName("MEM")
  mem_raster = mem_driver.Create(
      "", raster.shape[1], raster.shape[0], 2, gdal.GDT_Byte
  )
  if mem_raster is None:
    raise ValueError("Failed to create in-memory raster.")

  # Set spatial information.
  mem_raster.SetGeoTransform(geotransform)
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(epsg_code)
  mem_raster.SetProjection(srs.ExportToWkt())

  # Write data and mask to separate bands.
  data_band = mem_raster.GetRasterBand(1)
  data_band.WriteArray(raster)
  mask_array = (raster == 1).astype("uint8")
  mask_band = mem_raster.GetRasterBand(2)
  mask_band.WriteArray(mask_array)

  # Create in-memory shapefile.
  output_shp_path = f"/vsimem/{uuid.uuid4()}.shp"
  shp_driver = ogr.GetDriverByName("ESRI Shapefile")
  output_ds = shp_driver.CreateDataSource(output_shp_path)
  if output_ds is None:
    raise ValueError("Failed to create in-memory shapefile.")
  output_layer = output_ds.CreateLayer("polygons", srs=srs)
  field_def = ogr.FieldDefn("value", ogr.OFTInteger)
  output_layer.CreateField(field_def)

  # Polygonize gdal operation.
  gdal.Polygonize(data_band, mask_band, output_layer, 0, [], callback=None)

  # Explicitly destroy GDAL objects to flush data and release locks.
  del mem_raster, output_layer, output_ds

  gdf = gpd.read_file(output_shp_path)

  # In case of failure, clean up the temporary files and raise an error.
  if gdf.empty:
    for ext in [".shp", ".shx", ".dbf", ".prj"]:
      try:
        gdal.Unlink(output_shp_path.replace(".shp", ext))
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    raise ValueError("Empty GeoDataFrame. Polygonization failed.")

  # Drop mask column before returning.
  return gdf.drop(columns=["value"])


def generate_utm_metadata(grid_spacing_m: float) -> pd.DataFrame:
  """Generates metadata for UTM zones."""
  utm_zones = ee_export_utils.get_utm_zones()
  d = []
  for utm_zone in utm_zones:
    start_lat, start_lon, lat_step, lon_step = utm_zone
    end_lon = start_lon + lon_step
    end_lat = start_lat + lat_step
    x_num, y_num = utils.get_utm_grid_size(
        start_lat, start_lon, end_lat, end_lon, grid_spacing_m
    )
    roi = coords.UtmGridMapping.from_latlon_center(
        (start_lat + end_lat) / 2,
        (start_lon + end_lon) / 2,
        grid_spacing_m,
        x_num,
        y_num,
    )
    d.append((
        start_lat,
        start_lon,
        end_lat,
        end_lon,
        roi.utm_zone,
        roi.utm_x_min,
        roi.utm_y_min + roi.cell_size * y_num,
        roi.epsg,
    ))

  lat_lon_columns = ["lat", "lon", "lat_end", "lon_end"]
  utm_columns = ["utm_zone", "utm_x_min", "utm_y_max"]
  return pd.DataFrame(d, columns=[*lat_lon_columns, *utm_columns, "epsg"])


def read_cells_metadata(cells_metadata_path: str, utm_zone: str):
  """Reads cells metadata."""
  if UTM_ZONE_KEYWORD in cells_metadata_path:
    cells_metadata_path = cells_metadata_path.replace(
        UTM_ZONE_KEYWORD, utm_zone
    )
  if not gfile.exists(cells_metadata_path):
    logging.info("File %s does not exist.", cells_metadata_path)
    return None

  data = pipelines.read_file_to_df(cells_metadata_path)
  if "id" not in data.columns:
    data["id"] = np.arange(0, len(data))

  data = data[data["utm_zone"] == utm_zone]
  return data


def _apply_temp_scaling(
    probs: np.ndarray, temp: float, eps: float = 1e-10
) -> np.ndarray:
  probs = np.clip(probs, eps, 1 - eps)
  pseudo_logits = np.log(probs / (1 - probs))
  temp_probs = 1 / (1 + np.exp(-pseudo_logits / temp))
  return temp_probs


def _extract_data(npz, column: str) -> np.ndarray:
  """Extracts data from npz file for a given column."""
  channels = None
  time_index = None
  divide_by = 1
  temperature = 1
  if "@" in column:
    column, temperature = column.split("@")
    temperature = float(temperature)
  if "/" in column:
    column, divide_by = column.split("/")
    divide_by = float(divide_by)
  if "_" in column:
    column, time_index = column.split("_")
    time_index = int(time_index)
  if ":" in column:
    column, *channels = column.split(":")
    channels = [int(x) for x in channels]
  predictions = np.array(npz[column])
  if channels:
    predictions = predictions[..., channels]
  if time_index is not None:
    predictions = predictions[:, time_index]
  # Ensure that floats are converted to float32 to reduce memory usage.
  if predictions.dtype.kind == "f":
    predictions = predictions.astype(np.float32)
  if divide_by != 1:
    predictions = predictions / divide_by
  if temperature != 1:
    predictions = _apply_temp_scaling(predictions, temperature)
  # Add a fake C dimension for (B, H, W) case.
  if len(predictions.shape) == 3:
    predictions = np.expand_dims(predictions, axis=-1)
  # Add a fake (H, W) dimension for (B, C) case.
  if len(predictions.shape) == 2:
    predictions = np.expand_dims(predictions, axis=(1, 2))
  return predictions


def _read_file_data(
    file: str, cells_metadata: pd.DataFrame, columns_to_export: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
  """Read data from file."""

  with gfile.GFile(file, "rb") as f:
    npz = np.load(f)
    data = {"id": npz["id"].tolist()}
    if "dy" in npz.keys():
      data["dy"] = npz["dy"].tolist()
      data["dx"] = npz["dx"].tolist()
    else:
      data["dy"] = np.zeros_like(npz["id"])
      data["dx"] = np.zeros_like(npz["id"])
    ids = pd.DataFrame.from_dict(data)
    meta = ids.merge(cells_metadata, on="id")
    predictions = _extract_data(npz, columns_to_export[0])
    assert (
        meta.shape[0] == predictions.shape[0]
    ), f"meta: {meta.shape} preds: {predictions.shape}"
    for additional_column in columns_to_export[1:]:
      extra = _extract_data(npz, additional_column)
      assert (
          predictions.shape[:-1] == extra.shape[:-1]
      ), f"Predictions shape: {predictions.shape} Extra shape: {extra.shape}"
      predictions = np.concatenate((predictions, extra), axis=-1)
  return meta, predictions


def get_plot_data(
    utm_x: int,
    utm_y: int,
    x_splits: list[int],
    y_splits: list[int],
    num_splits: int,
    cell_size: float,
    plot_size_m: float,
    utm_zone: str,
    predictions: np.ndarray | dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    dx: int = 0,
    dy: int = 0,
) -> list[PlotData]:
  """Get the plot data for a given cell."""
  if isinstance(predictions, dict):
    shape = list(predictions.values())[0].shape
    if not all(shape == p.shape for p in predictions.values()):
      raise ValueError(
          "All predictions must have the same shape."
          f"Got: {[(k, p.shape) for k, p in predictions.items()]}"
      )
  else:
    shape = predictions.shape
  pred_y_shape = shape[-3]
  pred_x_shape = shape[-2]

  # This is currently inefficient when the number of splits is very large
  # (~1k) but it seems unlikely we are ever going to have that many.
  x_split = next(i - 1 for i, xx in enumerate(x_splits) if xx > utm_x)
  y_split = next(i - 1 for i, yy in enumerate(y_splits) if yy > utm_y)

  # TODO - The double for loop below is a lazy check to account for
  # the border size. We can do better.
  assignments = []
  for xs in range(max(0, x_split - 1), min(num_splits, x_split + 2)):
    for ys in range(max(0, y_split - 1), min(num_splits, y_split + 2)):

      x_split_slice = slice(x_splits[xs], x_splits[xs + 1])
      x_start = utm_x + dx * cell_size - plot_size_m // 2
      x_plot_slice = slice(x_start, x_start + pred_x_shape * cell_size)
      assert x_plot_slice.start is not None and x_plot_slice.stop is not None
      if (
          x_plot_slice.stop <= x_split_slice.start
          or x_plot_slice.start >= x_split_slice.stop
      ):
        continue

      y_split_slice = slice(y_splits[ys], y_splits[ys + 1])
      y_end = utm_y - dy * cell_size + plot_size_m // 2
      y_plot_slice = slice(y_end - pred_y_shape * cell_size, y_end)
      assert y_plot_slice.start is not None and y_plot_slice.stop is not None
      if (
          y_plot_slice.stop <= y_split_slice.start
          or y_plot_slice.start >= y_split_slice.stop
      ):
        continue

      assignments.append(
          PlotData(utm_zone, xs, ys, utm_x, utm_y, dx, dy, predictions, mask)
      )
  return assignments


def read_and_split_data(
    zone_and_file: tuple[str, str],
    bbox: tuple[float, float, float, float],
    # We need to use Optional because beam doesn't like None in type hints.
    cells_metadata: Optional[pd.DataFrame],
    cells_metadata_path: str,
    columns_to_export: list[str],
    grid_spacing_m: float,
    cell_size: float,
    plot_size_m: float,
    num_splits: int,
    rearrange: str,
) -> Iterable[PlotData]:
  """Read data from file and split it into splits."""
  utm_zone, file = zone_and_file
  if cells_metadata is None:
    cells_metadata = read_cells_metadata(cells_metadata_path, utm_zone)
  if cells_metadata is None:
    return

  meta, predictions = _read_file_data(file, cells_metadata, columns_to_export)
  logging.info(
      "Processing zone %s with %d plots from file %s",
      *(utm_zone, len(predictions), predictions.shape, file),
  )

  # Expectations on the predictions shape: (n, ..., y, x, c).
  if rearrange:
    predictions = einops.rearrange(predictions, rearrange)

  x_splits, y_splits = get_info(
      *bbox,
      grid_spacing_m,
      cell_size,
      plot_size_m,
      num_splits,
  )

  for i in range(meta.shape[0]):
    assignments = get_plot_data(
        utm_x=meta.iloc[i]["utm_x"],
        utm_y=meta.iloc[i]["utm_y"],
        x_splits=x_splits,
        y_splits=y_splits,
        num_splits=num_splits,
        cell_size=cell_size,
        plot_size_m=plot_size_m,
        predictions=predictions[i],
        utm_zone=utm_zone,
        dx=meta.iloc[i]["dx"],
        dy=meta.iloc[i]["dy"],
    )
    assert assignments, f"Plot: {i} has not been assigned. File: {file}"

    for plot in assignments:
      yield plot


def get_data_for_one_split(
    plots_data: Iterable[PlotData],
    bbox: tuple[float, float, float, float],
    epsg: str,
    border_mode: str,
    normalize: bool,
    clip_min: float,
    clip_max: float,
    discretization_factor: float,
    add_argmax: str,
    shift_argmax: int,
    output_type: str,
    grid_spacing_m: float,
    cell_size: float,
    plot_size_m: float,
    num_splits: int,
) -> Iterable[SplitData]:
  """Process one split of data."""
  lat, lon, lat_end, lon_end = bbox

  inference_data = collections.defaultdict(list)
  inference_masks = collections.defaultdict(list)
  plots_data = list(plots_data)
  if not plots_data:
    return

  has_masks = any(p.mask is not None for p in plots_data)
  utm_zone = plots_data[0].utm_zone
  x_split = plots_data[0].x_split
  y_split = plots_data[0].y_split
  while plots_data:
    plot_data = plots_data.pop()
    inference_data[plot_data.coords].append(plot_data.preds)
    if has_masks:
      inference_masks[plot_data.coords].append(plot_data.mask)

  # Plots are grouped by (utm zone, x_split, y_split), so we can use any.
  logging.info(
      "Processing utm_zone: %s x_split: %s, y_split: %s",
      *(utm_zone, x_split, y_split),
  )
  x_splits, y_splits = get_info(
      *bbox, grid_spacing_m, cell_size, plot_size_m, num_splits
  )
  x_slice = slice(x_splits[x_split], x_splits[x_split + 1])
  y_slice = slice(y_splits[y_split], y_splits[y_split + 1])
  geotransform = (
      x_slice.start,
      cell_size,
      0.0,
      y_slice.start,
      0.0,
      cell_size,
  )

  logging.info("Before get_numpy_data")
  np_data, np_data_mask = get_numpy_data(
      inference_data,
      lat,
      lon,
      lat_end,
      lon_end,
      x_split,
      y_split,
      num_splits,
      grid_spacing_m,
      cell_size,
      plot_size_m,
      border_mode,
      normalize,
      clip_min,
      clip_max,
      discretization_factor,
      add_argmax,
      shift_argmax,
      output_type,
      inference_masks=inference_masks if has_masks else None,
  )
  del inference_data, inference_masks

  # Avoid to create empty rasters.
  if np_data_mask.sum() > 0:
    yield SplitData(
        utm_zone, x_split, y_split, np_data, np_data_mask, geotransform, epsg
    )


def write_geotiff_for_split(
    split_data: SplitData,
    gcs_folder: str,
    ee_asset_id: str,
    raster_name: str,
    write_cogs: bool,
    resampling: str,
):
  """Write geotiff for a given split."""
  np_data = split_data.np_data
  np_data_mask = split_data.np_data_mask
  geotransform = split_data.geotransform
  epsg = split_data.epsg

  logging.info("Before write_tif, non masked pixels: %s", np_data_mask.sum())
  filename = os.path.join(gcs_folder, ee_asset_id, f"{raster_name}.tif")
  dirname = os.path.dirname(filename)
  if not dirname.startswith("gs://"):
    gfile.makedirs(dirname)
  write_tif(
      np_data,
      np_data_mask,
      geotransform,
      epsg,
      write_cogs,
      resampling,
      output_filename=filename,
  )
  logging.info("Wrote %s directly to %s", raster_name, filename)

  return filename


def get_geotiff_for_one_split(
    split_data: SplitData,
    gcs_folder: str,
    ee_asset_id: str,
    one_asset_per_utm_zone: bool,
    write_cogs: bool,
    ee_project: str,
    num_splits: int,
    resampling: str,
) -> Iterable[tuple[str, str]]:
  """Process one split of data."""
  utm_zone, x_split, y_split, *_ = dataclasses.astuple(split_data)
  raster_name = f"raster_{utm_zone}_{num_splits}_{y_split}_{x_split}"
  filename = write_geotiff_for_split(
      split_data, gcs_folder, ee_asset_id, raster_name, write_cogs, resampling
  )
  if not one_asset_per_utm_zone:
    create_upload_task(
        filenames=[filename],
        raster_name=raster_name,
        ee_asset_id=get_ee_asset_id(ee_project, ee_asset_id),
        force=write_cogs,  #  We overwrite when using COG format.
        write_cogs=write_cogs,
        ee_project=ee_project,
    )
  yield utm_zone, filename


def create_upload_task(
    filenames: list[str],
    raster_name: str,
    ee_asset_id: str,
    force: bool,
    write_cogs: bool = True,
    ee_project: str = "",
    properties: dict[str, Any] | None = None,
    start_time: str | None = None,
):
  """Creates upload task.

  Args:
    filenames: List of GCS file paths to upload.
    raster_name: Name of the raster asset.
    ee_asset_id: EE asset collection ID.
    force: Whether to overwrite existing assets.
    write_cogs: Whether to write Cloud Optimized GeoTIFFs.
    ee_project: EE project for authentication.
    properties: User-defined properties to set on the EE asset.
    start_time: Optional ISO 8601 timestamp (e.g. '2024-03-15T00:00:00Z') to set
      as the asset's temporal start_time for filterDate() support.

  Returns:
    The EE asset ID of the created asset.
  """
  logging.info("Create_upload_task for: %s", raster_name)

  if ee_project:
    utils.initialize_ee(ee_project)

  if write_cogs:
    for f in filenames:
      name = f[f.rindex("/") + 1 : -4]
      asset_body = {
          "type": "IMAGE",
          "cloud_storage_location": {"uris": [f]},
          "name": f"{ee_asset_id}/{name}",
      }
      if start_time:
        asset_body["start_time"] = start_time
      retry_count = 5
      for retry in range(retry_count):
        try:
          ee.data.createAsset(asset_body, force=force, properties=properties)
        except ee.EEException as e:
          if "Quota exceeded for quota metric" in str(e):
            logging.warning(
                "Quota exceeded for quota metric: %s, retrying in %s seconds",
                str(e),
                60 * (2**retry),
            )
            if retry == retry_count - 1:
              raise e
            time.sleep(60 * (2**retry))
            continue
          else:
            raise e
        break
  else:
    request_id = ee.data.newTaskId()[0]
    request = {
        "name": f"{ee_asset_id}/{raster_name}",
        "properties": properties,
        "tilesets": [
            {"id": "ts", "sources": [{"uris": [f]} for f in filenames]}
        ],
        "pyramidingPolicy": "MODE",
    }
    if start_time:
      request["start_time"] = start_time
    ee.data.startIngestion(request_id, request, force)


def export_single_item(
    data: np.ndarray,
    geotransform: tuple[float, float, float, float, float, float],
    epsg: str,
    gcs_folder: str,
    asset_name: str,
    asset_id: str,
    force: bool,
    mask: np.ndarray | None = None,
    write_cogs: bool = True,
):
  """Exports a raster item to EE.

  Note that, if write_cogs is True, it is not possible to reuse the same asset
  id/name for different data. For example, if you want to export Sentinel-2 and
  USDA/NAIP/DOQQ you cannot reused the same asset id/name even if force is
  True. This is not the case for non-COG formats.

  Args:
    data: The raster data [height, width, bands].
    geotransform: The raster geotransform.
    epsg: The raster epsg.
    gcs_folder: The GCS folder to upload the raster to.
    asset_name: The name of the raster asset.
    asset_id: The ID of the raster asset.
    force: Whether to force the upload.
    mask: The raster mask [height, width]. Default is no masking.
    write_cogs: Whether to write the raster in COG format.
  """

  # Write tmp tiff file before uploading to GCS.
  tmp_filename = write_tif(
      data, mask, geotransform=geotransform, epsg=epsg, write_cogs=write_cogs
  )

  # Upload the tiff file to GCS.
  filename = os.path.join(gcs_folder, f"{asset_name}.tif")
  gfile.Copy(tmp_filename, filename, overwrite=True)
  gfile.Remove(tmp_filename)

  create_upload_task(
      filenames=[filename],
      raster_name=asset_name,
      ee_asset_id=asset_id,
      force=force,
      write_cogs=write_cogs,
  )
