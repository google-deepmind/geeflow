# Copyright 2024 DeepMind Technologies Limited.
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

"""Compute and save data stats (per band for multi-band imagery)."""

import os

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from geeflow import utils
from geeflow.stats import compute_stats_beam_utils
import jax
import tensorflow_datasets as tfds

flags.DEFINE_enum(
    "running_mode",
    "direct",
    ["direct", "cloud"],
    "https://beam.apache.org/releases/pydoc/2.33.0/_modules/apache_beam/options/pipeline_options.html"
    " for details of additional flags",
)


flags.DEFINE_string("tfds_name", "", "TFDS name.")
flags.DEFINE_string("data_dir", "", "TFDS input data dir.")
flags.DEFINE_string("output_dir", "", "Output dir for stats.")
flags.DEFINE_list("split", "train", "Split.")
FLAGS = flags.FLAGS


def convert_format(example):
  yield jax.tree.map(utils.to_np, example)


def create_pipeline_for_split(split: str, root: beam.Pipeline) -> beam.Pipeline:
  """Creates the beam pipeline for stats computation."""
  ds = tfds.builder(FLAGS.tfds_name, data_dir=FLAGS.data_dir)
  data = (
      root
      | f"ReadInput_{split}" >> tfds.beam.ReadFromTFDS(ds, split=split)
      | f"ConvertFormat_{split}" >> beam.FlatMap(convert_format)
  )
  if FLAGS.output_dir:
    output_dir = os.path.join(FLAGS.output_dir, ds.info.full_name)
  else:  # ds.data_dir includes ds.info.full_name.
    output_dir = os.path.join(ds.data_dir, "stats")
  return compute_stats_beam_utils.create_pipeline(data, split=split,
                                                  output_dir=output_dir)


def create_pipeline(root: beam.Pipeline) -> beam.Pipeline:
  """Creates the beam pipeline for stats computation."""
  pipeline = {}
  for split in FLAGS.split:
    pipeline[split] = create_pipeline_for_split(split, root)
  return pipeline


def main(argv):
  if FLAGS.running_mode == "direct":
    options = beam.options.pipeline_options.DirectOptions(argv)
  else:
    options = beam.options.pipeline_options.GoogleCloudOptions(argv)
  with beam.Pipeline(options=options) as p:
    create_pipeline(p)


if __name__ == "__main__":
  app.run(main)
