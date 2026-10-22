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

"""Export EE data beam pipeline.

Example command line for running locally:
python -m geeflow.export_beam_tfds \
--config_path=/path_to_config/config.py \
--output_dir=/path_to_dir --ee_project=<my_project> \
--running_mode=direct \
-- --direct_num_workers=16 -- direct_running_mode=multi_threading
"""

from absl import app
from absl import flags
from absl import logging
from geeflow import export_beam_tfds_utils
from geeflow import times
from ml_collections import config_flags
import tensorflow_datasets.public_api as tfds

flags.DEFINE_string(
    "config_path", None, "Export config file path.", required=True)
flags.DEFINE_string(
    "ee_project", None,
    "A ':' concatenation of a GCP project and a EE service account.")
flags.DEFINE_string("output_dir", "", "Output directory")
flags.DEFINE_string("tfds_name", "", "TFDS name, builder config name and "
                    "version (`name[/builder_config_name][:version]`).")
flags.DEFINE_list("splits", "train,validation,test", "Splits to generate.")
flags.DEFINE_string("split_column_name", "split", "Column to use for a split.")
flags.DEFINE_integer("splits_s2_cell_level", 9, "S2 level for random splits.")
flags.DEFINE_string(
    "splits_ratios",
    "8:1:1",
    "Ratios of splits (train:val:test).",
)
flags.DEFINE_string("file_format", "bagz", "The format for the dataset files")
flags.DEFINE_bool("nondeterministic_order", False,
                  "DownloadConfig.nondeterministic_order feature.")
flags.DEFINE_enum(
    "running_mode",
    "direct",
    ["direct", "cloud"],
    "https://beam.apache.org/releases/pydoc/2.33.0/_modules/apache_beam/options/pipeline_options.html"
    " for details of additional flags",
)
FLAGS = flags.FLAGS


def main(argv):
  """Launch the pipeline."""
  config_flags.DEFINE_config_file(
      "config", FLAGS.config_path, "Export config.", lock_config=True
  )

  now = times.to_timestr(times.now(times.ZRH))
  logging.info("Reference time now: %s", now)

  if FLAGS.running_mode == "direct":
    options = beam.options.pipeline_options.DirectOptions(argv)
  else:
    options = beam.options.pipeline_options.GoogleCloudOptions(argv)
  download_config = tfds.download.DownloadConfig(
      beam_options=beam.options.pipeline_options.PipelineOptions(
          options=options
      )
  )

  builder = export_beam_tfds_utils.TFDSBuilder(
      FLAGS.config,
      FLAGS.tfds_name,
      FLAGS.output_dir,
      splits=FLAGS.splits,
      ee_project=FLAGS.ee_project,
      split_column_name=FLAGS.split_column_name,
      splits_s2_cell_level=FLAGS.splits_s2_cell_level,
      splits_ratios=FLAGS.splits_ratios,
  )
  builder.download_and_prepare(download_dir=FLAGS.output_dir,
                               file_format=FLAGS.file_format,
                               download_config=download_config)


if __name__ == "__main__":
  app.run(main)
