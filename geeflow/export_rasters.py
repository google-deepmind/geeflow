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
import functools
import re

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from geeflow import export_rasters_utils as export_utils
from geeflow import utils

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
flags.DEFINE_float(
    "grid_spacing_m",
    960,
    "Spacing in meteres between plots in the grid "
    "(used via cells_metadata_path flag).",
)
flags.DEFINE_float(
    "plot_size_m",
    960,
    "Size of a single plot in meters. Usually the same as "
    "grid_spacing_m, but could be larger if running with "
    "overlaps",
)
flags.DEFINE_float("cell_size", 10, "Image pixel size in meters.")
flags.DEFINE_enum(
    "border_mode",
    "none",
    ["none", "uniform_avg", "l2_dist_avg", "any", "l1_border_dist_avg"],
    "A mode of aggregation of overlapping pixels.",
)
flags.DEFINE_multi_string(
    "inference_data", None, "Input inference data path on CNS."
)
flags.DEFINE_string("cells_metadata_path", "", "Cells metadata path.")
flags.DEFINE_string("ee_asset_id", "", "GEE asset id to export inference data.")
flags.DEFINE_bool(
    "normalize", False, "Normalizes the output across the last dimension."
)
flags.DEFINE_string("add_argmax", "",
                    "Adds argmax to the output. "
                    "Possible values: '', 'add', 'only'."
                    "'add' - stacks argmax as the first channel"
                    "'only' - only outputs argmax")
flags.DEFINE_integer("shift_argmax", 0, "By how much to shift argmax.")
flags.DEFINE_float("clip_min", None, "Clip min value.")
flags.DEFINE_float("clip_max", None, "Clip max value.")
flags.DEFINE_integer("discretization_factor", None, "Discretization factor.")
flags.DEFINE_string("output_type", None, "Output type.")
flags.DEFINE_enum("resampling", "AVERAGE", ["AVERAGE", "MODE"],
                  "Resampling method to use when creating overviews.")
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


def create_utm_zone_tiles(data):
  """Aggregates stats."""
  utm_zone, files = data
  if FLAGS.one_asset_per_utm_zone:
    create_upload_task(files, f"raster_{utm_zone}")


def _get_utm_zones():
  """Returns UTM zones to process and the corresponding bbox."""
  utms_metadata = export_utils.generate_utm_metadata(FLAGS.grid_spacing_m)
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
  if (
      UTM_ZONE_KEYWORD in FLAGS.cells_metadata_path
      and "*" not in FLAGS.cells_metadata_path
  ):
    utm_zones_with_cells_metadata = set()
    path = FLAGS.cells_metadata_path.replace(UTM_ZONE_KEYWORD, "*")
    pos = FLAGS.cells_metadata_path.index(UTM_ZONE_KEYWORD)
    files = gfile.Glob(path)
    for f in files:
      utm_zones_with_cells_metadata.add(f[pos : -(len(path) - pos - 1)])

  logging.info(
      "UTM Zones with cells metadata: %s", utm_zones_with_cells_metadata
  )

  # Use glob only a single time for all utm zones per inference path
  # (significant speedup).
  all_files_no_utm_zone = []
  zone_to_files = collections.defaultdict(list)
  for path in FLAGS.inference_data:
    if UTM_ZONE_KEYWORD in path:
      regex = re.compile(
          path.replace("*", ".*").replace(UTM_ZONE_KEYWORD, "([0-9]{1,2}[A-Z])")
      )
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
    if (
        utm_zones_with_cells_metadata
        and utm_zone not in utm_zones_with_cells_metadata
    ):
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
    for utm_zone, bbox, epsg, files in _get_utm_zones():
      cells_metadata = None
      if FLAGS.read_cells_metadata_on_master:
        cells_metadata = export_utils.read_cells_metadata(
            FLAGS.cells_metadata_path, utm_zone
        )
        if cells_metadata is None:
          continue

      if not files:
        logging.info("No files found for utm_zone: %s", utm_zone)
        continue

      read_and_split_data = functools.partial(
          export_utils.read_and_split_data,
          bbox=bbox,
          cells_metadata=cells_metadata,
          cells_metadata_path=FLAGS.cells_metadata_path,
          columns_to_export=FLAGS.columns_to_export,
          grid_spacing_m=FLAGS.grid_spacing_m,
          cell_size=FLAGS.cell_size,
          plot_size_m=FLAGS.plot_size_m,
          num_splits=FLAGS.num_splits,
          rearrange=FLAGS.rearrange,
      )
      get_data_for_one_split = functools.partial(
          export_utils.get_data_for_one_split,
          bbox=bbox,
          epsg=epsg,
          border_mode=FLAGS.border_mode,
          normalize=FLAGS.normalize,
          clip_min=FLAGS.clip_min,
          clip_max=FLAGS.clip_max,
          discretization_factor=FLAGS.discretization_factor,
          add_argmax=FLAGS.add_argmax,
          shift_argmax=FLAGS.shift_argmax,
          output_type=FLAGS.output_type,
          grid_spacing_m=FLAGS.grid_spacing_m,
          cell_size=FLAGS.cell_size,
          plot_size_m=FLAGS.plot_size_m,
          num_splits=FLAGS.num_splits,
      )
      get_geotiff_for_one_split = functools.partial(
          export_utils.get_geotiff_for_one_split,
          gcs_folder=FLAGS.gcs_folder,
          one_asset_per_utm_zone=FLAGS.one_asset_per_utm_zone,
          write_cogs=FLAGS.write_cogs,
          ee_project=FLAGS.ee_project,
          ee_asset_id=FLAGS.ee_asset_id,
          num_splits=FLAGS.num_splits,
          resampling=FLAGS.resampling,
      )

      pipeline[utm_zone] = (
          root
          | f"Create_{utm_zone}" >> beam.Create([(utm_zone, f) for f in files])
          | f"Read {utm_zone}" >> beam.FlatMap(read_and_split_data)
          | f"Group_files_{utm_zone}" >> beam.GroupBy(lambda x: x.key)
          | f"Get_data_{utm_zone}" >> beam.Values()
          | f"Process_{utm_zone}" >> beam.FlatMap(get_data_for_one_split)
          | f"Get_geotiff_{utm_zone}" >> beam.FlatMap(get_geotiff_for_one_split)
          | f"Group_{utm_zone}" >> beam.GroupByKey()
          | f"CreateUTMZoneTiles_{utm_zone}"
          >> beam.FlatMap(create_utm_zone_tiles)
      )
    return pipeline


  def main(argv):
  assert FLAGS.columns_to_export, "No columns to export specified."
  assert not (
      FLAGS.write_cogs and FLAGS.one_asset_per_utm_zone
  ), "If write_cogs is enabled the one_asset_per_utm_zone must not be set."
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
