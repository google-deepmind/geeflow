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

r"""A pipeline to export inference result into EE.

Example command line for running locally:
python export_rasters.py \
--inference_data=/path_to_data/inference_results_*.npz \
--cells_metadata_path=/path_to_data/metadata.parquet \
--ee_asset_id=project/inference/forest_height \
--grid_spacing_m=150 --running_mode=direct \
-- --direct_num_workers=16 -- direct_running_mode=multi_threading
"""

import collections
from collections.abc import Iterable, Iterator
import math
import os
import re
import tempfile

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from geeflow import coords
from geeflow import ee_export_utils
from geeflow import pipelines
from geeflow import utils
import numpy as np
from osgeo import gdal
from osgeo import gdalconst
import pandas as pd

from tensorflow.io import gfile
import ee


flags.DEFINE_string(
    "ee_project", None,
    "A ':' contatenation of a GCP project and a EE service account.")
flags.DEFINE_string(
    "gcs_folder", None,
    "The folder for storing files before uploading them to EE.")
flags.DEFINE_enum(
    "running_mode", "direct", ["direct", "cloud"],
    "See https://beam.apache.org/releases/pydoc/2.33.0/_modules/apache_beam/options/pipeline_options.html"
    " for details on additional flags.")
# One could use UTM_ZONE_KEYWORD key word inside "output_file".
flags.DEFINE_list("utm_zones", None, "List of UTM zones to filter to.")
flags.DEFINE_float("grid_spacing_m",
                   960,
                   "Spacing in meteres between plots in the grid "
                   "(used via cells_metadata_path flag).")
flags.DEFINE_float("plot_size_m",
                   960,
                   "Size of a single plot in meters. Usually the same as "
                   "grid_spacing_m, but could be larger if running with "
                   "overlaps")
flags.DEFINE_bool("use_utm", True, "Use UTM projection.")
flags.DEFINE_float("img_width_deg", 0.01, "Image width in degrees.")
flags.DEFINE_float("cell_size", 10, "Image pixel size in meters.")
flags.DEFINE_enum("border_mode", "none",
                  ["none", "uniform_avg", "l2_dist_avg", "any",
                   "l1_border_dist_avg"],
                  "A mode of aggregation of overlapping pixels.")
flags.DEFINE_multi_string("inference_data", None,
                          "Input inference data path on CNS.")
flags.DEFINE_string("cells_metadata_path", "", "Cells metadata path.")
flags.DEFINE_string("ee_asset_id", "", "GEE asset id to export inference data.")
flags.DEFINE_bool("normalize", False,
                  "Normalizes the output across the last dimension.")
flags.DEFINE_bool("add_argmax", False, "Adds and stacks argmax to the output.")
flags.DEFINE_integer("shift_argmax", 0, "By how much to shift argmax.")
flags.DEFINE_float("clip_min", None, "Clip min value.")
flags.DEFINE_float("clip_max", None, "Clip max value.")
flags.DEFINE_integer("discretization_factor", None, "Discretization factor.")
flags.DEFINE_string("output_type", None, "Output type.")
# Can specify which channels to export with ":" separator (e.g. "preds:1:2:3")
# Can specify temperature with "@" separator (e.g. "preds@0.2")
# Can specify divisor with "/" separator (e.g. "preds/2")
# Order of separators matters. They have to be in :/@ order.
flags.DEFINE_list("columns_to_export", ["predictions"], "What data to export.")
flags.DEFINE_integer("num_splits", 1,
                     "Split each UTM zone into num_splits*num_splits parts. "
                     "Used purely as a performance optimization and to avoid "
                     "OOMs.")
flags.DEFINE_bool("one_asset_per_utm_zone", True,
                  "When True, one asset per UTM zone is created.")
flags.DEFINE_bool("read_cells_metadata_on_master", False,
                  "When True, read cells metadata on the master. Useful to "
                  "enable when running for a few UTM zones with many files "
                  "that contains inference results. This way every worker will"
                  " not need to read the same data over and over again.")
flags.DEFINE_bool("write_cogs", False,
                  "When True, stores rasters as COG files on GCP.")
FLAGS = flags.FLAGS


UTM_ZONE_KEYWORD = "{utm_zone}"


def _get_ee_asset_id() -> str:
  return f"projects/{FLAGS.ee_project.split(':')[0]}/assets/{FLAGS.ee_asset_id}"


def generate_utm_metadata():
  """Generates metadata for UTM zones."""
  utm_zones = ee_export_utils.get_utm_zones()
  d = []
  for utm_zone in utm_zones:
    start_lat, start_lon, lat_step, lon_step = utm_zone
    end_lon = start_lon + lon_step
    end_lat = start_lat + lat_step
    x_num, y_num = utils.get_utm_grid_size(
        start_lat, start_lon, end_lat, end_lon, FLAGS.grid_spacing_m)
    roi = coords.UtmGridMapping.from_latlon_center(
        (start_lat + end_lat) / 2,
        (start_lon + end_lon) / 2, FLAGS.grid_spacing_m, x_num, y_num)
    d.append((start_lat, start_lon, end_lat, end_lon, roi.utm_zone,
              roi.utm_x_min, roi.utm_y_min + roi.cell_size * y_num, roi.epsg))

  return pd.DataFrame(d, columns=["lat", "lon", "lat_end", "lon_end",
                                  "utm_zone", "utm_x_min", "utm_y_max", "epsg"])


def _apply_temp_scaling(probs: np.ndarray, temp: float, eps: float = 1e-10
                        ) -> np.ndarray:
  probs = np.clip(probs, eps, 1 - eps)
  pseudo_logits = np.log(probs / (1 - probs))
  temp_probs = 1 / (1 + np.exp(-pseudo_logits / temp))
  return temp_probs


def _extract_data(npz, column: str) -> np.ndarray:
  """Extracts data from npz file for a given column."""
  channels = None
  divide_by = 1
  temperature = 1
  if "@" in column:
    column, temperature = column.split("@")
    temperature = float(temperature)
  if "/" in column:
    column, divide_by = column.split("/")
    divide_by = float(divide_by)
  if ":" in column:
    column, *channels = column.split(":")
    channels = [int(x) for x in channels]
  predictions = np.array(npz[column])
  if channels:
    predictions = predictions[..., channels]
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
    file: str, cells_metadata: pd.DataFrame
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
    predictions = _extract_data(npz, FLAGS.columns_to_export[0])
    assert (
        meta.shape[0] == predictions.shape[0]
    ), f"meta: {meta.shape} preds: {predictions.shape}"
    for additional_column in FLAGS.columns_to_export[1:]:
      extra = _extract_data(npz, additional_column)
      assert predictions.shape[:-1] == extra.shape[:-1], (
          f"Predictions shape: {predictions.shape} Extra shape: {extra.shape}"
      )
      predictions = np.concatenate((predictions, extra), axis=-1)
  return meta, predictions


def read_and_split_data(
    data: tuple[str, str], bbox, cells_metadata: pd.DataFrame | None
) -> Iterable[tuple[tuple[str, int, int],
                    tuple[int, int, int, int, np.ndarray]]]:
  """Read data from file and split it into splits."""
  utm_zone, file = data
  if cells_metadata is None:
    cells_metadata = read_cells_metadata(FLAGS.cells_metadata_path, utm_zone)
  if cells_metadata is None: return

  meta, predictions = _read_file_data(file, cells_metadata)
  logging.info("Processing zone %s with %d plots from file %s", utm_zone,
               len(predictions), file)

  x_num, y_num, image_width, border_skip = _get_info(*bbox)
  x_splits = np.linspace(0, x_num, FLAGS.num_splits + 1, dtype=int)
  y_splits = np.linspace(0, y_num, FLAGS.num_splits + 1, dtype=int)

  for i in range(meta.shape[0]):
    x, y, dx, dy = meta.iloc[i][["x", "y", "dx", "dy"]]

    # This is currently inefficient when the number of splits is very large
    # (~1k) but it seems unlikely we are ever going to have that many.
    x_split = next(i - 1 for i, xx in enumerate(x_splits) if xx > x)
    y_split = next(
        i - 1 for i, yy in enumerate(y_splits) if yy > (y_num - 1 - y)
    )

    # TODO - The double for loop below is a lazy check to account for
    # the border size. We can do better.
    assignments = []
    for xs in range(max(0, x_split - 1), min(FLAGS.num_splits, x_split + 2)):
      for ys in range(max(0, y_split - 1), min(FLAGS.num_splits, y_split + 2)):

        x_split_slice = _get_split_slice(x_num, xs, image_width)
        x_plot_slice = _get_plot_slice(x * image_width + dx, image_width,
                                       border_skip)
        if (
            x_plot_slice[1] <= x_split_slice.start
            or x_plot_slice[0] >= x_split_slice.stop
        ):
          continue

        y_split_slice = _get_split_slice(y_num, ys, image_width)
        y_plot_slice = _get_plot_slice((y_num - 1 - y) * image_width + dy,
                                       image_width, border_skip)
        if (
            y_plot_slice[1] <= y_split_slice.start
            or y_plot_slice[0] >= y_split_slice.stop
        ):
          continue

        assignments.append(((utm_zone, xs, ys), (x, y, dx, dy, predictions[i])))

    assert assignments, f"Plot: {i} has not been assigned. File: {file}"
    for k, v in assignments:
      yield k, v


def read_cells_metadata(cells_metadata_path: str, utm_zone: str):
  """"Reads cells metadata."""
  if UTM_ZONE_KEYWORD in cells_metadata_path:
    cells_metadata_path = cells_metadata_path.replace(UTM_ZONE_KEYWORD,
                                                      utm_zone)
  if not gfile.exists(cells_metadata_path):
    logging.info("File %s does not exist.", cells_metadata_path)
    return None

  data = pipelines.read_file_to_df(cells_metadata_path)
  if "id" not in data.columns:
    data["id"] = np.arange(0, len(data))

  data = data[data["utm_zone"] == utm_zone]
  return data


def _get_weights(mode: str, size: int, dtype):
  """Returns weights for border averaging."""
  if mode == "l2_dist_avg":
    w = np.array([[[math.hypot(i - size / 2, j - size / 2)]
                   for j in range(size)]
                  for i in range(size)])
    w = np.maximum(1 - w / np.max(w), 0.001)
    return w
  if mode == "l1_border_dist_avg":
    assert size % 2 == 0, "size expected to be even"
    # TODO - Generalize border handling.
    border = size // 4
    assert border <= size//2, "not tested for border > size//2"
    m = np.tile(np.array(list(range(size//2-1, -1, -1)) +
                         list(range(size//2)))[None], (size, 1))
    m = np.maximum(m, m.T)  # L1 (Mahalanobis) distance from center.
    w = m.max() - m + 1  # Weights starting at 1.
    if border:
      w = np.minimum(w, border + 1)
    w = w / w.max()  # Normalize to max weight 1.
    w = np.expand_dims(w, -1)
    return w
  return np.ones((size, size, 1), dtype=dtype)


def _get_split_slice(num: int, split: int, image_width: int):
  total_splits = np.linspace(0, num, FLAGS.num_splits+1).astype(int)
  return slice(total_splits[split] * image_width,
               total_splits[split + 1] * image_width)


def _get_plot_slice(
    vmin: float, image_width: int, border_skip: int
) -> tuple[float, float]:
  if FLAGS.border_mode not in [
      "uniform_avg",
      "l2_dist_avg",
      "l1_border_dist_avg",
      "any",
  ]:
    return vmin, vmin + image_width
  return (vmin - border_skip, vmin + image_width + border_skip)


def _sliced_update(a: np.ndarray, y_slice: slice, x_slice: slice,
                   main_x_slice: slice, main_y_slice: slice, p: np.ndarray,
                   add: bool = True):
  """Updates a slice of a numpy array."""
  y_slice = slice(y_slice.start - main_y_slice.start,
                  y_slice.stop - main_y_slice.start)
  assert y_slice.start is not None and y_slice.stop is not None
  if y_slice.start < 0:
    p = p[-y_slice.start:]
    y_slice = slice(0, y_slice.stop)
  if y_slice.stop > main_y_slice.stop:
    p = p[:-(y_slice.stop - main_y_slice.stop)]
    y_slice = slice(y_slice.start, main_y_slice.stop)
  if y_slice.start >= y_slice.stop:
    return

  x_slice = slice(x_slice.start - main_x_slice.start,
                  x_slice.stop - main_x_slice.start)
  assert x_slice.start is not None and x_slice.stop is not None
  if x_slice.start < 0:
    p = p[:, -x_slice.start:]
    x_slice = slice(0, x_slice.stop)
  if x_slice.stop > main_x_slice.stop:
    p = p[:,:-(x_slice.stop - main_x_slice.stop)]
    x_slice = slice(x_slice.start, main_x_slice.stop)
  if x_slice.start >= x_slice.stop:
    return

  if add:
    a[y_slice, x_slice] += p
  else:
    a[y_slice, x_slice] = p


def _get_info(lat, lon, lat_end, lon_end):
  """Return information based on image size."""
  if FLAGS.use_utm:
    x_num, y_num = utils.get_utm_grid_size(
        lat, lon, lat_end, lon_end, FLAGS.grid_spacing_m)
    # Image width in pixels.
    image_width = FLAGS.grid_spacing_m / FLAGS.cell_size
    if abs(image_width - round(image_width)) < 1e-10:
      image_width = round(image_width)
    else:
      raise ValueError("grid_spacing_m / cell_size must be an exact integer.")
    # Skip overlapping border.
    border_skip = (FLAGS.plot_size_m - FLAGS.grid_spacing_m) / (
        2 * FLAGS.cell_size
    )
    if abs(border_skip - round(border_skip)) < 1e-10:
      border_skip = round(border_skip)
    else:
      raise ValueError(
          "(plot_size_m - grid_spacing_m) / (2 * cell_size) must be an exact"
          " integer."
      )
  else:
    x_num = math.ceil((lon_end - lon) / FLAGS.img_width_deg)
    y_num = math.ceil((lat_end - lat) / FLAGS.img_width_deg)
    # Parameters (image_width=1, border_skip=0) are for patch classification.
    # TODO: Add support for segmentation tasks here when needed.
    image_width = 1
    border_skip = 0
  return x_num, y_num, image_width, border_skip


def get_numpy_data(inference_data: dict[tuple[int, int, int, int],
                                        list[np.ndarray]],
                   lat: float, lon: float, lat_end: float, lon_end: float,
                   x_split: int, y_split: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns numpy array representing inference results."""
  x_num, y_num, image_width, border_skip = _get_info(lat, lon, lat_end,
                                                     lon_end)

  # Get one example to determine num_bands and dtype.
  p = next(iter(inference_data.values()))[0]
  num_bands = 1
  if len(p.shape) == 3 or len(p.shape) == 1:
    num_bands = p.shape[-1]
  npdtype = p.dtype
  if (p.dtype in [np.float32, np.float64] or
      FLAGS.border_mode in ["uniform_avg", "l2_dist_avg",
                            "l1_border_dist_avg"]):
    npdtype = np.float32
  main_x_slice = _get_split_slice(x_num, x_split, image_width)
  main_y_slice = _get_split_slice(y_num, y_split, image_width)
  # This is the global accumulator/aggregator.
  a = np.zeros((main_y_slice.stop - main_y_slice.start,
                main_x_slice.stop - main_x_slice.start, num_bands),
               dtype=npdtype)
  # This is global accumulated weights for normalization.
  w = np.zeros((main_y_slice.stop - main_y_slice.start,
                main_x_slice.stop - main_x_slice.start, 1), dtype=npdtype)
  w_mask = _get_weights(FLAGS.border_mode, p.shape[0], npdtype)
  for (x, y, dx, dy), predictions_list in inference_data.items():
    ymin = (y_num - 1 - y) * image_width - border_skip + dy
    xmin = x * image_width - border_skip + dx
    y_slice = slice(ymin, ymin + predictions_list[0].shape[0])
    x_slice = slice(xmin, xmin + predictions_list[0].shape[1])
    assert y_slice.start is not None
    assert x_slice.start is not None
    if y_slice.stop <= main_y_slice.start or y_slice.start >= main_y_slice.stop:
      continue
    if x_slice.stop <= main_x_slice.start or x_slice.start >= main_x_slice.stop:
      continue
    for predictions in predictions_list:
      p = predictions
      if FLAGS.border_mode in ["uniform_avg", "l2_dist_avg",
                               "l1_border_dist_avg", "any"]:
        assert w_mask is not None
        w_y_slice = slice(0, w_mask.shape[0])
        w_x_slice = slice(0, w_mask.shape[1])

        assert y_slice.start is not None and y_slice.stop is not None
        assert x_slice.start is not None and x_slice.stop is not None
        if y_slice.start < 0:
          p = p[-y_slice.start:, ...]
          w_y_slice = slice(-y_slice.start, w_y_slice.stop)
          y_slice = slice(0, y_slice.stop)
        if y_slice.stop > main_y_slice.stop:
          p = p[:-(y_slice.stop - main_y_slice.stop), ...]
          w_y_slice = slice(w_y_slice.start,
                            w_y_slice.stop - (y_slice.stop - main_y_slice.stop))
          y_slice = slice(y_slice.start, main_y_slice.stop)
        if x_slice.start < 0:
          p = p[:, -x_slice.start:, ...]
          w_x_slice = slice(-x_slice.start, w_x_slice.stop)
          x_slice = slice(0, x_slice.stop)
        if x_slice.stop > main_x_slice.stop:
          p = p[:,:-(x_slice.stop - main_x_slice.stop), ...]
          w_x_slice = slice(w_x_slice.start,
                            w_x_slice.stop - (x_slice.stop - main_x_slice.stop))
          x_slice = slice(x_slice.start, main_x_slice.stop)
        if FLAGS.border_mode == "any":
          _sliced_update(a, y_slice, x_slice, main_x_slice, main_y_slice,
                         p.astype(npdtype), add=False)
          _sliced_update(w, y_slice, x_slice, main_x_slice, main_y_slice,
                         w_mask.astype(npdtype), add=False)
        else:
          _sliced_update(a, y_slice, x_slice, main_x_slice, main_y_slice,
                         p * w_mask[w_y_slice, w_x_slice], add=True)
          _sliced_update(w, y_slice, x_slice, main_x_slice, main_y_slice,
                         w_mask[w_y_slice, w_x_slice], add=True)
      else:
        _sliced_update(a, y_slice, x_slice, main_x_slice, main_y_slice,
                       p.astype(npdtype), add=False)
        _sliced_update(w, y_slice, x_slice, main_x_slice, main_y_slice,
                       w_mask.astype(npdtype), add=False)

  if FLAGS.border_mode in ["uniform_avg", "l2_dist_avg", "l1_border_dist_avg"]:
    a = a / np.maximum(w, 1e-10)

  if FLAGS.normalize:
    a = a / np.maximum(np.sum(a, axis=-1, keepdims=True), 1e-10)

  if FLAGS.clip_min is not None and FLAGS.clip_max is not None:
    a = np.clip(a, FLAGS.clip_min, FLAGS.clip_max)
  if FLAGS.discretization_factor is not None:
    a = a * FLAGS.discretization_factor

  if FLAGS.add_argmax:
    a = np.concatenate([
        np.argmax(a, axis=-1, keepdims=True) + FLAGS.shift_argmax, a], axis=-1)

  if FLAGS.output_type == "int16":
    a = a.astype(np.int16)
  if FLAGS.output_type == "int8":
    a = a.astype(np.int8)
  if FLAGS.output_type == "uint8":
    a = a.astype(np.uint8)

  return a, w[..., 0] > 0


def create_tiff_options(numpy_array: np.ndarray) -> list[str]:
  """Returns options for the maximum lossless compression.

  Also handles the special case required for signed int8 data.

  Note that COMPRESS=LZW sometimes exhibits worst-case behavior for data
  with >1 byte per pixel: resulting file sizes can exceed raw file size.

  See:
  https://gdal.org/drivers/raster/gtiff.html#creation-options
  https://numpy.org/doc/stable/reference/arrays.dtypes.html

  Args:
    numpy_array: Array whose type will be used to deturmine the best options.

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

  if FLAGS.write_cogs:
    options += [
        "COPY_SRC_OVERVIEWS=YES",
        "TILED=YES",
        "INTERLEAVE=BAND",
        "OVERVIEWS=IGNORE_EXISTING",
        "COMPRESS=ZSTD",
        "LEVEL=22",
    ]
  else:
    options += ["COMPRESS=DEFLATE", "ZLEVEL=9"]

  return options


def write_tif(np_data: np.ndarray, np_data_mask: np.ndarray,
              geotransform: tuple[float, float, float, float, float, float],
              epsg: str) -> str:
  """Writes tiff."""
  _, temp_filename = tempfile.mkstemp(suffix=".tif")

  dtype = gdalconst.GDT_Byte
  if isinstance(np_data.flat[0], np.float32):
    dtype = gdalconst.GDT_Float32
  if isinstance(np_data.flat[0], np.int16):
    dtype = gdalconst.GDT_Int16
  no_data_value = -1
  if dtype == gdalconst.GDT_Byte:
    no_data_value = 255
  if FLAGS.write_cogs:
    driver = gdal.GetDriverByName("MEM")
    ds = driver.Create("", np_data.shape[1], np_data.shape[0],
                       np_data.shape[2], dtype)
  else:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(temp_filename, np_data.shape[1], np_data.shape[0],
                       np_data.shape[2], dtype, create_tiff_options(np_data))
  ds.SetGeoTransform(geotransform)
  ds.SetProjection(epsg)
  for i in range(np_data.shape[-1]):
    rb = ds.GetRasterBand(i + 1)
    data = np_data[..., i]
    data = np.where(np_data_mask, data, no_data_value)
    rb.SetNoDataValue(no_data_value)
    rb.WriteArray(data)
  if FLAGS.write_cogs:
    ds.BuildOverviews("MODE", [2, 4, 8, 16, 32, 64])
    ds.FlushCache()

    # Now that we have the image in memory with overviews, we can make the
    # cloud optimized geotiff, which is really just a geotiff that has tiling
    # turned on and has the overviews created before the tiling was done.
    ds2 = gdal.GetDriverByName("COG").CreateCopy(
        temp_filename, ds, options=create_tiff_options(np_data)
    )
    # Ensure it really really gets written to temp.
    ds2.FlushCache()
  else:
    ds.FlushCache()
  return temp_filename


def create_upload_task(filenames: list[str], raster_name: str) -> None:
  """Creates upload task."""
  logging.info("Before create_upload_task: %s", raster_name)
  # Initialize and authenticate EE on the flume workers.
  utils.initialize_ee(FLAGS.ee_project)

  if FLAGS.write_cogs:
    for f in filenames:
      name = f[f.rindex("/") + 1: -4]
      ee.data.createAsset({
          "type": "IMAGE",
          "cloud_storage_location": {"uris": [f]},
          "name": f"{_get_ee_asset_id()}/{name}",
      }, force=True)
  else:
    request_id = ee.data.newTaskId()[0]
    request = {
        "name": f"{_get_ee_asset_id()}/{raster_name}",
        "properties": {},
        "tilesets": [{"id": "ts", "sources": [{"uris": [f]} for f in filenames]}],
        "pyramidingPolicy": "MODE"}
    force = False
    ee.data.startIngestion(request_id, request, force)


def process_one_split(
    data: tuple[tuple[str, int, int],
                Iterable[tuple[int, int, int, int, np.ndarray]]],
    bbox, utm_zone_info
) -> Iterator[tuple[str, str]]:
  """Process one split of data."""
  (utm_zone, x_split, y_split), plots_data = data
  lat, lon, lat_end, lon_end = bbox
  utm_x_min, utm_y_max, epsg = utm_zone_info

  inference_data = collections.defaultdict(list)
  plots_data = list(plots_data)
  while plots_data:
    xp, yp, dx, dy, p = plots_data.pop()
    inference_data[xp, yp, dx, dy].append(p)

  logging.info("Processing utm_zone: %s x_split: %s, y_split: %s",
               utm_zone, x_split, y_split)
  if FLAGS.use_utm:
    x_num, y_num = utils.get_utm_grid_size(
        lat, lon, lat_end, lon_end, FLAGS.grid_spacing_m)
    # Grid plot width in pixels.
    grid_plot_width = FLAGS.grid_spacing_m / FLAGS.cell_size
    if abs(grid_plot_width - round(grid_plot_width)) < 1e-10:
      grid_plot_width = round(grid_plot_width)
    else:
      raise ValueError("grid_spacing_m / cell_size must be an exact integer.")
    x_slice = _get_split_slice(x_num, x_split, grid_plot_width)
    y_slice = _get_split_slice(y_num, y_split, grid_plot_width)
    geotransform = (
        utm_x_min + x_slice.start * FLAGS.cell_size,
        FLAGS.cell_size,
        0.0,
        utm_y_max - y_slice.start * FLAGS.cell_size,
        0.0,
        -FLAGS.cell_size,
    )
  else:
    geotransform = (lon, FLAGS.img_width_deg, 0.0,
                    lat_end, 0.0, -FLAGS.img_width_deg)
    epsg = "EPSG:4326"

  logging.info("Before get_numpy_data")
  np_data, np_data_mask = get_numpy_data(inference_data, lat, lon, lat_end,
                                         lon_end, x_split, y_split)

  # Avoid to create empty rasters.
  if np_data_mask.sum() == 0: return

  logging.info("Before write_tif, non masked pixels: %s", np_data_mask.sum())
  temp_filename = write_tif(np_data, np_data_mask, geotransform, epsg)

  raster_name = f"raster_{utm_zone}_{FLAGS.num_splits}_{y_split}_{x_split}"
  logging.info("Before copy(%s): size %s", raster_name,
               os.stat(temp_filename).st_size)
  filename = os.path.join(FLAGS.gcs_folder, FLAGS.ee_asset_id,
                          f"{raster_name}.tif")
  gfile.MakeDirs(os.path.dirname(filename),
                 mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE)
  gfile.Copy(temp_filename, filename, overwrite=True)
  gfile.Remove(temp_filename)

  if not FLAGS.one_asset_per_utm_zone:
    create_upload_task([filename], raster_name)

  yield utm_zone, filename


def create_utm_zone_tiles(data):
  """Aggregates stats."""
  utm_zone, files = data
  if FLAGS.one_asset_per_utm_zone:
    create_upload_task(files, f"raster_{utm_zone}")


def _get_utm_zones():
  """Returns UTM zones to process and the corresponding bbox."""
  utms_metadata = generate_utm_metadata()
  utm_zones = FLAGS.utm_zones or utms_metadata["utm_zone"].tolist()

  existing_assets = ee.ImageCollection(_get_ee_asset_id())
  existing_assets = existing_assets.toList(1000000)
  existing_assets = existing_assets.map(lambda y: ee.Image(y).id())
  existing_assets = existing_assets.getInfo()
  logging.info("UTM Zones already in EE: %s", existing_assets)

  result = []

  utm_zones_with_cells_metadata = None
  if (UTM_ZONE_KEYWORD in FLAGS.cells_metadata_path and
      "*" not in FLAGS.cells_metadata_path):
    utm_zones_with_cells_metadata = set()
    path = FLAGS.cells_metadata_path.replace(UTM_ZONE_KEYWORD, "*")
    pos = FLAGS.cells_metadata_path.index(UTM_ZONE_KEYWORD)
    files = gfile.Glob(path)
    for f in files:
      utm_zones_with_cells_metadata.add(f[pos: -(len(path) - pos - 1)])

  logging.info("UTM Zones with cells metadata: %s",
               utm_zones_with_cells_metadata)

  # Use glob only a single time for all utm zones per inference path
  # (significant speedup).
  all_files_no_utm_zone = []
  zone_to_files = collections.defaultdict(list)
  for path in FLAGS.inference_data:
    if UTM_ZONE_KEYWORD in path:
      regex = re.compile(
          path.replace("*", ".*").replace(UTM_ZONE_KEYWORD, "([0-9A-Z]*)"))
      logging.info(gfile.Glob(path.replace(UTM_ZONE_KEYWORD, "*")))
      for file in gfile.Glob(path.replace(UTM_ZONE_KEYWORD, "*")):
        match = regex.match(file)
        assert match is not None
        utm_zone = match.group(1)
        zone_to_files[utm_zone].append(file)
    else:
      all_files_no_utm_zone += gfile.Glob(path)

  for utm_zone in utm_zones:
    if f"raster_{utm_zone}" in existing_assets:
      logging.info("UTM Zone %s already in EE, skipping.", utm_zone)
      continue
    if (utm_zones_with_cells_metadata and
        utm_zone not in utm_zones_with_cells_metadata):
      logging.info("UTM Zone %s has no cells metadata, skipping.", utm_zone)
      continue
    files = all_files_no_utm_zone + zone_to_files[utm_zone]
    if not files:
      logging.info("UTM Zone %s has no inference data, skipping.", utm_zone)
      continue

    utm_metadata = utms_metadata[utms_metadata["utm_zone"] == utm_zone]
    assert utm_metadata.shape[0] == 1
    lat, lon = utm_metadata[["lat", "lon"]].iloc[0]
    lat_end, lon_end = utm_metadata[["lat_end", "lon_end"]].iloc[0]
    bbox = lat, lon, lat_end, lon_end

    utm_x_min, utm_y_max = utm_metadata[["utm_x_min", "utm_y_max"]].iloc[0]
    epsg = utm_metadata["epsg"].iloc[0]
    utm_zone_info = utm_x_min, utm_y_max, epsg

    result.append((utm_zone, bbox, utm_zone_info, files))

  logging.info("UTM Zones to process(%d): %s", len(result), result)
  return result


class IngestRastersPipeline(beam.PTransform):
  """Pipeline to ingest inference results into EE."""

  def expand(self, root):
    pipeline = {}
    for utm_zone, bbox, info, files in _get_utm_zones():
      cells_metadata = None
      if FLAGS.read_cells_metadata_on_master:
        cells_metadata = read_cells_metadata(FLAGS.cells_metadata_path,
                                             utm_zone)
        if cells_metadata is None: continue

      if not files:
        logging.info("No files found for utm_zone: %s", utm_zone)
        continue

      pipeline[utm_zone] = (
          root
          | f"Create_{utm_zone}" >> beam.Create([(utm_zone, f) for f in files])
          | f"Read {utm_zone}" >> beam.FlatMap(read_and_split_data, bbox,
                                               cells_metadata)
          | f"Group_files_{utm_zone}" >> beam.GroupByKey()
          | f"Process_{utm_zone}" >> beam.FlatMap(process_one_split, bbox, info)
          | f"Group_{utm_zone}" >> beam.GroupByKey()
          | f"CreateUTMZoneTiles_{utm_zone}"
          >> beam.FlatMap(create_utm_zone_tiles)
      )
    return pipeline


def main(argv):
  assert FLAGS.columns_to_export, "No columns to export specified."
  assert not (FLAGS.write_cogs and FLAGS.one_asset_per_utm_zone), (
      "If write_cogs is enabled the one_asset_per_utm_zone must not be set.")
  utils.initialize_ee(FLAGS.ee_project)
  ee.data.create_assets(
      [_get_ee_asset_id()], ee.data.ASSET_TYPE_IMAGE_COLL, True
  )

  if FLAGS.running_mode == "direct":
    options = beam.options.pipeline_options.DirectOptions(argv)
  else:
    options = beam.options.pipeline_options.GoogleCloudOptions(argv)
  with beam.Pipeline(options=options) as p:
    (p | IngestRastersPipeline())

if __name__ == "__main__":
  flags.mark_flags_as_required(["ee_project", "gcs_folder"])
  app.run(main)
