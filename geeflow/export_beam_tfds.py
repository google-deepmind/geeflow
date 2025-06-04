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

"""Export EE data beam pipeline.

Example command line for running locally:
python -m geeflow.export_beam_tfds \
--config_path=/path_to_config/config.py \
--output_dir=/path_to_dir --ee_project=<my_project> \
--running_mode=direct \
-- --direct_num_workers=16 -- direct_running_mode=multi_threading
"""

import copy
import hashlib
import math

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from geeflow import coords
from geeflow import ee_algo
from geeflow import ee_export_utils
from geeflow import pipelines
from geeflow import times
from geeflow import utils
import ml_collections
from ml_collections import config_flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

flags.DEFINE_string(
    "config_path", None, "Export config file path.", required=True)
flags.DEFINE_string(
    "ee_project",
    "computing-engine-190414:earth-engine@computing-engine-190414.iam.gserviceaccount.com",
    "A ':' contatenation of a GCP project and a EE service account.")
flags.DEFINE_string("output_dir", "", "Output directory")
flags.DEFINE_string("tfds_name", "", "TFDS name, builder config name and "
                    "version (`name[/builder_config_name][:version]`).")
flags.DEFINE_list("splits", "train,validation,test", "Splits to generate.")
flags.DEFINE_string("split_column_name", "split", "Column to use for a split.")
flags.DEFINE_integer("splits_s2_cell_level", 9, "S2 level for random splits.")
flags.DEFINE_string("file_format", "array_record",
                    "The format for the dataset files")
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


SPLIT_PLACEHOLDER = "{split}"


class CreateData(beam.DoFn):
  """Beam DoFn to create an initalial collection of features grouped."""

  def __init__(self, config: ml_collections.ConfigDict, split):
    self.config = config
    self.split = split

  def start_bundle(self):
    # Initialize and authenticate EE.
    utils.initialize_ee(FLAGS.ee_project)

  def process(self, path):
    config = copy.deepcopy(self.config)
    config.labels.path = path
    # Disable caching since we are running multithreading experiments and
    # caching sometimes fails since multiple threads try to write the same
    # file.
    df = pipelines.get_labels_df(config, cache=False)
    label_items = pipelines.pipeline_labels(config, df)
    if self.split != "full":
      def _filter_example(ex):
        if FLAGS.split_column_name in ex:
          # Allow validation examples to be in either "val" or "validation".
          if self.split in ["val", "validation"]:
            return ex[FLAGS.split_column_name] in ["val", "validation"]
          return ex[FLAGS.split_column_name] == self.split
        else:
          # Compute s2-cell patch id (int64)
          # Levels: http://s2geometry.io/resources/s2cell_statistics.html
          s2c = coords.latlon_to_s2(ex["lat"], ex["lon"], level=
                                    FLAGS.splits_s2_cell_level)  # L9: 14-20km.
          s2hash = int(hashlib.md5(str(s2c).encode("utf-8")).hexdigest(), 16)
          if self.split == "test" and s2hash % 10 == 9:
            return True
          elif self.split in ["val", "validation"] and s2hash % 10 == 8:
            return True
          elif self.split == "train" and s2hash % 10 not in {8, 9}:
            return True
        return False
      label_items = list(filter(_filter_example, label_items))
    if not label_items and path == self.config.labels.path:
      # Only crash if we are reading a single file.
      raise ValueError(f"No examples identified for split `{self.split}` "
                       f"(full labels pd.DataFrame shape: {df.shape}). "
                       "Possibly due to random geographic sampling.")
    return label_items


def _has_forest_loss(feature):
  return np.max(np.array(feature["hansen"])[..., 1]) > 0


def apply_filters(feature, config):
  export_config = config.get("export", {})
  if export_config.get("filter_empty_forest_loss", False):
    if not _has_forest_loss(feature):
      return False
  if keys := export_config.get("filter_empty_sequences", []):
    keys = [keys] if isinstance(keys, str) else keys
    for key in keys:
      if not feature[key]:
        return False
  return True


def generate_ccdc(data, config, key: str):
  """Generates CCDC data in [T,H,W,C] format."""
  assert f"{key}_tStart" in data
  start_dates = np.array(data[f"{key}_tStart"])
  h = start_dates.shape[0]
  w = start_dates.shape[1]
  year_from = config["from"]
  year_to = config["to"]
  ignore_tstart = config.get("ignore_tstart", False)
  num_bands = 0
  ccdc_features = {}
  for k, v in data.items():
    if not k.startswith(f"{key}_"): continue
    if ignore_tstart and k == f"{key}_tStart": continue
    nv = np.array(v)
    ccdc_features[k] = nv
    if len(nv.shape) == 3:
      num_bands += 1
    else:
      num_bands += nv.shape[-1]
  ccdc = np.zeros((year_to - year_from + 1, h, w, num_bands), dtype=np.float32)
  ccdc_mask = np.ones(ccdc.shape, dtype=np.bool_)

  for x in range(h):
    for y in range(w):
      dates = start_dates[x, y]
      num_segments = dates.shape[0]
      segment_pos = 0
      previous_segment_pos = -1
      for year in range(year_from, year_to + 1):
        while (segment_pos + 1 < num_segments and
               year + 0.5 >= dates[segment_pos + 1] and
               dates[segment_pos + 1] > 0):
          segment_pos += 1
        if dates[segment_pos] == 0:
          # There's no data (value should be fractional year) -> mask out.
          ccdc_mask[year - year_from, x, y] = np.zeros_like(
              ccdc_mask[year - year_from, x, y])
        else:
          if previous_segment_pos == segment_pos:
            # No need to recompute, the segment didn't change.
            # Copy previous value.
            ccdc[year - year_from, x, y] = ccdc[year - year_from - 1, x, y]
          else:
            a = []
            for v in ccdc_features.values():
              if len(v.shape) == 3:
                a.append(v[x, y, segment_pos])
              else:
                a.extend(v[x, y, segment_pos])

            ccdc[year - year_from, x, y] = a
            previous_segment_pos = segment_pos
  if "year_selection" in config:
    ccdc = ccdc[config["year_selection"]]
    ccdc_mask = ccdc_mask[config["year_selection"]]
  return ccdc, ccdc_mask


def apply_transforms(feature, config, cropped_features_counter=None):
  """Apply transforms to feature."""
  for key, value in config.sources.items():
    if key.startswith("ccdc"):
      keys = [x for x in feature.keys() if x.startswith(f"{key}_")]
      feature[key], feature[f"{key}_mask"] = generate_ccdc(
          feature, value.format_config, key)
      for k in keys:
        del feature[k]

  default_image_width = config.labels.img_width_m
  for key, value in config.sources.items():
    if value.get("scalar", False): continue
    image_width = value.get("img_width_m", default_image_width)
    scale = value.get("scale", None)
    if not scale: continue
    assert image_width >= scale, f"{key}: {image_width} < {scale}"
    if isinstance(scale, float):
      s = math.ceil(image_width / scale)
      assert abs(s  * scale - image_width) < 1e-6
    else:
      if config.labels.get("use_utm", True):
        assert image_width % scale == 0, f"{key}: {image_width} % {scale} != 0"
        s = image_width // scale
      else:
        s = math.ceil(image_width / scale)

    for sub_key in [key, f"{key}_mask"]:
      data = np.array(feature[sub_key])
      # The expected data format is [H, W, C] or [T, H, W, C]. Even if C==1.
      if len(data.shape) != 4 and len(data.shape) != 3: continue
      if data.shape[-3] != s or data.shape[-2] != s:
        if cropped_features_counter:
          cropped_features_counter.inc()
        logging.info("Cropping %s from %s to %s", sub_key, data.shape, s)
        assert data.shape[-3] == s or data.shape[-3] == s + 1, sub_key
        assert data.shape[-2] == s or data.shape[-2] == s + 1, sub_key
        feature[sub_key] = data[
            ...,
            data.shape[-3] // 2 - s // 2 : data.shape[-3] // 2 + s - s // 2,
            data.shape[-2] // 2 - s // 2 : data.shape[-2] // 2 + s - s // 2,
            :]
  return feature


def _process_example(x, config):
  """Process a single example."""
  for k in config.get("skip_keys", []) + ["split"]:
    x.pop(k, None)
  out = {}
  for k, v in x.items():
    dtype = None
    if k.endswith("_mask") or k == "hr":
      dtype = np.uint8
    elif config.sources.get(k):
      dtype = config.sources[k].get("dtype")
    t = np.array(v, dtype=dtype)
    # Automatically convert data to float (even if array is empty).
    # Exceptions are:
    #  - !"dtype is None" - predetermined types
    #  - !"isinstance(t.flat[0], np.integer)" - don't touch non int types
    #  - !"in ignore_for_float_conversion" - keys in the exception list
    if (dtype is None and
        (not t.size or (isinstance(t.flat[0], np.integer) or
                        isinstance(t.flat[0], np.float64))) and
        k not in config.sources.get("ignore_for_float_conversion", []) and
        k not in config.labels.get("ignore_for_float_conversion", [])):
      t = t.astype(np.float32)
    if t.shape:
      out[k] = t
    else:
      # Keep scalars as they are.
      out[k] = v
  tfds_id_keys = config.labels.get("tfds_id_keys", ("id",))
  key = "-".join(map(str, (x[k] for k in tfds_id_keys)))
  return (key, out)


def _get_metadata_keys(config: ml_collections.ConfigDict) -> list[str]:
  """Returns a list of metadata keys to be exported."""
  ee_algos_with_ts = [
      ee_algo.ic_sample,
      ee_algo.rgb_ic_sample,
      ee_algo.ic_sample_date_ranges,
  ]

  metadata_keys = []
  for name, cfg in config.sources.items():
    source_ee_algo = cfg.get("algo") or pipelines.ALGO_MAP.get(
        cfg.get("module")
    )
    # Algorithms that always returns timestamps.
    if source_ee_algo in ee_algos_with_ts:
      metadata_keys.append(f"{name}_timestamps")
    # Additional properties requested.
    if additional_properties := cfg.get("sampling_kw", {}).get(
        "additional_properties"
    ):
      metadata_keys += [f"{name}_{p}" for p in additional_properties]
  return metadata_keys


def _is_time_varying_algo(
    config: ml_collections.ConfigDict, name: str
) -> bool:
  """Returns True if the given source is time varying."""
  cfg = config.sources.get(name, {})
  source_ee_algo = cfg.get("algo")
  default_ee_algo = pipelines.ALGO_MAP.get(cfg.get("module"))
  return (source_ee_algo or default_ee_algo) == ee_algo.ic_sample


def get_split_tuples(splits):
  """Constructs split tuples of TFDS split names and split names in data."""
  split_tuples = []
  for split in splits:
    if "=" in split:
      split_tuples.append(split.split("="))
    elif split == "validation":
      split_tuples.append(("validation", "val"))
    else:
      split_tuples.append((split, split))
  return split_tuples


class TFDSBuilder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder."""

  def __init__(self, config, name, data_dir, splits):
    for k in config.sources:
      if k.endswith("_mask"):
        raise ValueError(
            f"Sources must not end with the reserved postfix `_mask`: {k}"
        )

    self.config = config
    self.num_query_calls = beam.metrics.Metrics.counter(
        self.__class__, "num_query_calls")
    self.num_items_processed = beam.metrics.Metrics.counter(
        self.__class__, "num_items_processed")
    self.num_cropped_features = beam.metrics.Metrics.counter(
        self.__class__, "num_cropped_features")
    self.num_query_failures = beam.metrics.Metrics.counter(
        self.__class__, "num_query_failures")
    self.num_query_retries = beam.metrics.Metrics.counter(
        self.__class__, "num_query_retries")
    self.query_duration_distribution = beam.metrics.Metrics.distribution(
        self.__class__, "query_duration_distribution")
    self.query_bytes_distribution = beam.metrics.Metrics.distribution(
        self.__class__, "query_bytes_distribution")
    self.query_sleep_distribution = beam.metrics.Metrics.distribution(
        self.__class__, "query_sleep_distribution")
    self.splits = splits
    # Set name and VERSION and BUILDER_CONFIG before parent initialization.
    version = "0.0.1"  # Setting a version is required.
    if ":" in name:
      name, version = name.split(":")
    self.VERSION = tfds.core.Version(version)  # pylint: disable=invalid-name
    if "/" in name:  # Builder config is optional.
      name, builder_config_name = name.split("/")
      self.BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=builder_config_name)]  # pylint: disable=invalid-name
    self.name = name
    super().__init__(data_dir=data_dir)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(make_tfds_features(self.config)),
        supervised_keys=None,  # Set to `None` to disable.
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    if self.splits and self.splits[0]:
      return {split: self._generate_examples(split_name)
              for (split, split_name) in get_split_tuples(self.splits)}
    else:
      return {"full": self._generate_examples("full")}

  def _generate_examples(self, split):
    """Generate examples as dicts."""
    paths = tf.io.gfile.glob(
        self.config.labels.path.replace(SPLIT_PLACEHOLDER, split))
    tfds_pipe = (
        "Dummy create" >> beam.Create(paths)
        | "Create EE data" >> beam.ParDo(CreateData(self.config, split))
        | "Reshuffle after create data" >> beam.Reshuffle()
        | "Convert & get info" >> beam.ParDo(ee_export_utils.GetInfo(
            FLAGS.ee_project,
            self.config,
            self.num_query_calls,
            self.num_query_failures,
            self.num_items_processed,
            self.num_query_retries,
            self.query_duration_distribution,
            self.query_bytes_distribution,
            self.query_sleep_distribution,
            add_retries_counters=True))
        | "Filter" >> beam.Filter(apply_filters, self.config)
        | "Transform" >> beam.Map(apply_transforms, self.config,
                                  self.num_cropped_features))
    if post_process_map := self.config.get("post_process_map"):
      tfds_pipe |= f"Posterprocess {split}" >> beam.ParDo(post_process_map)
    return tfds_pipe | f"Process {split}" >> beam.Map(
        _process_example, self.config
    )


def make_tfds_features(config):
  """Converts features description to TFDS form."""
  config = copy.deepcopy(config)
  # It is possible to provide wildcards in label paths.
  if not (paths := tf.io.gfile.glob(
      config.labels.path.replace(SPLIT_PLACEHOLDER, FLAGS.splits[0]))):
    raise ValueError(f"Invalid labels path: {config.labels.path}")
  config.labels.path = paths[0]
  labels = pipelines.pipeline_labels(config)
  assert labels
  get_info = ee_export_utils.GetInfo(FLAGS.ee_project, config)
  get_info.start_bundle()
  feature_data = next(get_info.process(labels[0]))
  feature_data = apply_transforms(feature_data, config)
  if post_process_map := config.get("post_process_map"):
    if isinstance(post_process_map, beam.DoFn):
      post_process_map.setup()
      post_process_map.start_bundle()
      # Assuming that the post_process_map.process function takes `flush` makes
      # things significantly simpler for the inference pipe, which constructs
      # batches of data and the finish bundle returns
      # `beam.transforms.window.WindowedValue` objects instead of dict.
      feature_data = next(post_process_map.process(feature_data, flush=True))
    elif callable(post_process_map):
      feature_data = post_process_map(feature_data)
    else:
      raise ValueError(f"Unsupported post_process_map: {post_process_map}")
  _, data = _process_example(feature_data, config)
  out = {}

  metadata_keys = _get_metadata_keys(config)
  for k, v in data.items():
    v = np.array(v)
    dtype = v.dtype
    if k in metadata_keys:
      out[k] = tfds.features.Sequence(
          tfds.features.Tensor(shape=(), dtype=dtype)
      )
    elif _is_time_varying_algo(config, k.replace("_mask", "")):
      # Assume that width, height and channels are the same for all plots.
      out[k] = tfds.features.Sequence(
          tfds.features.Tensor(shape=v.shape[-3:], dtype=dtype)
      )
    else:
      if v.shape:
        out[k] = tfds.features.Tensor(shape=v.shape, dtype=dtype)
      else:
        out[k] = tfds.features.Scalar(dtype)
  return out


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

  builder = TFDSBuilder(FLAGS.config, FLAGS.tfds_name, FLAGS.output_dir,
                        splits=FLAGS.splits)
  builder.download_and_prepare(download_dir=FLAGS.output_dir,
                               file_format=FLAGS.file_format,
                               download_config=download_config)


if __name__ == "__main__":
  app.run(main)
