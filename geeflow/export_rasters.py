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
import os
import re

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import einops
from geeflow import coords
from geeflow import ee_export_utils
from geeflow import export_rasters_utils as export_utils
from geeflow import pipelines
from geeflow import utils
import numpy as np
import pandas as pd

from tensorflow.io import gfile
import ee


flags.DEFINE_string(
    "ee_project", None,
    "A ':' concatenation of a GCP project and a EE service account.")
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
flags.DEFINE_string("rearrange", None, "Rearrange the predictions shape. "
                    "Examples: 'n t y x -> n y x t' or 'n y x -> n y x 1'")
FLAGS = flags.FLAGS


UTM_ZONE_KEYWORD = "{utm_zone}"


def create_upload_task(filenames: list[str], raster_name: str):
  export_utils.create_upload_task(
      filenames=filenames,
      raster_name=raster_name,
      ee_asset_id=export_utils.get_ee_asset_id(
          FLAGS.ee_project, FLAGS.ee_asset_id
      ),
      force=FLAGS.write_cogs,  #  We overwrite when using COG format.
      write_cogs=FLAGS.write_cogs,
      ee_project=FLAGS.ee_project,
  )


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
    zone_and_file: tuple[str, str], bbox, cells_metadata: pd.DataFrame | None
) -> Iterable[tuple[tuple[str, int, int],
                    tuple[int, int, int, int, np.ndarray]]]:
  """Read data from file and split it into splits."""
  utm_zone, file = zone_and_file
  if cells_metadata is None:
    cells_metadata = read_cells_metadata(FLAGS.cells_metadata_path, utm_zone)
  if cells_metadata is None: return

  meta, predictions = _read_file_data(file, cells_metadata)
  logging.info("Processing zone %s with %d (%s) plots from file %s", utm_zone,
               len(predictions), predictions.shape, file)

  # Expectations on the predictions shape: (n, ..., y, x, c).
  if FLAGS.rearrange:
    predictions = einops.rearrange(predictions, FLAGS.rearrange)

  x_splits, y_splits = export_utils.get_info(
      *bbox,
      FLAGS.grid_spacing_m,
      FLAGS.cell_size,
      FLAGS.plot_size_m,
      FLAGS.num_splits,
  )

  pred_y_shape = predictions.shape[-3]
  pred_x_shape = predictions.shape[-2]

  for i in range(meta.shape[0]):
    utm_x, utm_y, dx, dy = meta.iloc[i][["utm_x", "utm_y", "dx", "dy"]]

    # This is currently inefficient when the number of splits is very large
    # (~1k) but it seems unlikely we are ever going to have that many.
    x_split = next(i - 1 for i, xx in enumerate(x_splits) if xx > utm_x)
    y_split = next(i - 1 for i, yy in enumerate(y_splits) if yy > utm_y)

    # TODO - The double for loop below is a lazy check to account for
    # the border size. We can do better.
    assignments = []
    for xs in range(max(0, x_split - 1), min(FLAGS.num_splits, x_split + 2)):
      for ys in range(max(0, y_split - 1), min(FLAGS.num_splits, y_split + 2)):

        x_split_slice = slice(x_splits[xs], x_splits[xs + 1])
        x_start = utm_x + dx * FLAGS.cell_size - FLAGS.plot_size_m // 2
        x_plot_slice = slice(x_start, x_start + pred_x_shape * FLAGS.cell_size)
        assert x_plot_slice.start is not None and x_plot_slice.stop is not None
        if (
            x_plot_slice.stop <= x_split_slice.start
            or x_plot_slice.start >= x_split_slice.stop
        ):
          continue

        y_split_slice = slice(y_splits[ys], y_splits[ys + 1])
        y_end = utm_y - dy * FLAGS.cell_size + FLAGS.plot_size_m // 2
        y_plot_slice = slice(y_end - pred_y_shape * FLAGS.cell_size, y_end)
        assert y_plot_slice.start is not None and y_plot_slice.stop is not None
        if (
            y_plot_slice.stop <= y_split_slice.start
            or y_plot_slice.start >= y_split_slice.stop
        ):
          continue

        assignments.append(((utm_zone, xs, ys),
                            (utm_x, utm_y, dx, dy, predictions[i])))

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


def process_one_split(
    data: tuple[tuple[str, int, int],
                Iterable[tuple[int, int, int, int, np.ndarray]]],
    bbox, epsg: str
) -> Iterator[tuple[str, str]]:
  """Process one split of data."""
  (utm_zone, x_split, y_split), plots_data = data
  lat, lon, lat_end, lon_end = bbox

  inference_data = collections.defaultdict(list)
  plots_data = list(plots_data)
  while plots_data:
    xp, yp, dx, dy, p = plots_data.pop()
    inference_data[xp, yp, dx, dy].append(p)

  logging.info("Processing utm_zone: %s x_split: %s, y_split: %s",
               utm_zone, x_split, y_split)
  x_splits, y_splits = export_utils.get_info(
      *bbox,
      FLAGS.grid_spacing_m,
      FLAGS.cell_size,
      FLAGS.plot_size_m,
      FLAGS.num_splits,
  )
  x_slice = slice(x_splits[x_split], x_splits[x_split + 1])
  y_slice = slice(y_splits[y_split], y_splits[y_split + 1])
  geotransform = (
      x_slice.start,
      FLAGS.cell_size,
      0.0,
      y_slice.start,
      0.0,
      FLAGS.cell_size,
  )

  logging.info("Before get_numpy_data")
  np_data, np_data_mask = export_utils.get_numpy_data(
      inference_data,
      lat,
      lon,
      lat_end,
      lon_end,
      x_split,
      y_split,
      FLAGS.num_splits,
      FLAGS.grid_spacing_m,
      FLAGS.cell_size,
      FLAGS.plot_size_m,
      FLAGS.border_mode,
      FLAGS.normalize,
      FLAGS.clip_min,
      FLAGS.clip_max,
      FLAGS.discretization_factor,
      FLAGS.add_argmax,
      FLAGS.shift_argmax,
      FLAGS.output_type,
  )

  # Avoid to create empty rasters.
  if np_data_mask.sum() == 0: return

  logging.info("Before write_tif, non masked pixels: %s", np_data_mask.sum())
  temp_filename = export_utils.write_tif(
      np_data, np_data_mask, geotransform, epsg, FLAGS.write_cogs
  )

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

  existing_assets = ee.ImageCollection(
      export_utils.get_ee_asset_id(FLAGS.ee_project, FLAGS.ee_asset_id)
  )
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
          path.replace("*", ".*")
          .replace(UTM_ZONE_KEYWORD, "([0-9]{1,2}[A-Z])"))
      logging.info(files := gfile.Glob(path.replace(UTM_ZONE_KEYWORD, "*")))
      for file in files:
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

    epsg = utm_metadata["epsg"].iloc[0]

    result.append((utm_zone, bbox, epsg, files))

  logging.info("UTM Zones to process(%d): %s", len(result), result)
  if not result:
    logging.warning("No UTM Zones to process (but an empty asset was created).")
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
      [export_utils.get_ee_asset_id(FLAGS.ee_project, FLAGS.ee_asset_id)],
      ee.data.ASSET_TYPE_IMAGE_COLL,
      True,
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
