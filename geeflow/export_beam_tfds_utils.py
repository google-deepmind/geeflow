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

"""Shared utilities for GEE export Beam pipelines.

Provides the common GEE fetch → filter → transform → post-process pipeline
stages used by both export_beam_tfds.py and e2e_inference_pipeline.py.
"""

import copy
import hashlib
from typing import Any

from absl import logging
import apache_beam as beam
from geeflow import coords
from geeflow import ee_algo
from geeflow import ee_export_utils
from geeflow import pipelines
from geeflow import scalar_fetch
from geeflow import utils
import ml_collections
import numpy as np
import tensorflow_datasets.public_api as tfds

from tensorflow.io import gfile

Metrics = beam.metrics.Metrics


def build_gee_fetch_pipeline(
    items_pcoll: beam.PCollection,
    config: ml_collections.ConfigDict,
    ee_project: str,
    prefix: str = '',
) -> beam.PCollection:
  """Builds the shared GEE fetch → filter → transform → post-process pipeline.

  This function encapsulates the common Beam pipeline stages that fetch data
  from Google Earth Engine, apply filters and transforms, and optionally
  post-process the results. It is used by both export_beam_tfds.py and
  e2e_inference_pipeline.py.

  Args:
    items_pcoll: Input PCollection of label items (dicts with lat/lon/etc).
    config: Geeflow configuration dict.
    ee_project: EE project:service_account string for authentication.
    prefix: Optional prefix for Beam stage names to allow multiple parallel
      branches in a single pipeline.

  Returns:
    A PCollection of processed examples (dicts with numpy arrays).
  """

  def _name(stage):
    return f'{prefix}/{stage}' if prefix else stage

  # Create metrics for GEE fetch monitoring.
  num_query_calls = Metrics.counter('GEEFetch', 'num_query_calls')
  num_query_failures = Metrics.counter('GEEFetch', 'num_query_failures')
  num_items_processed = Metrics.counter('GEEFetch', 'num_items_processed')
  num_query_retries = Metrics.counter('GEEFetch', 'num_query_retries')
  query_duration_distribution = Metrics.distribution(
      'GEEFetch', 'query_duration_distribution'
  )
  query_bytes_distribution = Metrics.distribution(
      'GEEFetch', 'query_bytes_distribution'
  )
  query_sleep_distribution = Metrics.distribution(
      'GEEFetch', 'query_sleep_distribution'
  )
  cropped_features_counter = Metrics.counter('GEEFetch', 'num_cropped_features')

  scalar_config, non_scalar_config = scalar_fetch.split_scalar_config(config)
  examples = items_pcoll
  if scalar_config.sources:
    examples = (
        examples
        | _name('BatchBySpatialCell')
        >> scalar_fetch.BatchBySpatialCell.from_config(scalar_config)
        | _name('SampleScalarSources')
        >> beam.ParDo(
            scalar_fetch.SampleScalarSources(ee_project, scalar_config)
        )
    )
  if non_scalar_config.sources:
    examples = (
        examples
        | _name('Reshuffle') >> beam.Reshuffle()
        | _name('FetchFromGEE')
        >> beam.ParDo(
            ee_export_utils.GetInfo(
                ee_project,
                non_scalar_config,
                num_calls_counter=num_query_calls,
                num_failures_counter=num_query_failures,
                num_items_processed=num_items_processed,
                num_query_retries=num_query_retries,
                query_duration_distribution=query_duration_distribution,
                query_bytes_distribution=query_bytes_distribution,
                query_sleep_distribution=query_sleep_distribution,
                add_retries_counters=True,
            )
        )
    )
  examples = (
      examples
      | _name('ApplyFilters')
      >> beam.Filter(ee_export_utils.apply_filters, config)
      | _name('ApplyTransforms')
      >> beam.Map(
          ee_export_utils.apply_transforms, config, cropped_features_counter
      )
  )

  # Apply PostProcess (e.g., chunking into 512×512 tiles).
  if post_process := config.get('post_process_map'):
    if isinstance(post_process, beam.DoFn):
      examples |= _name('PostProcess') >> beam.ParDo(post_process)
    elif isinstance(post_process, beam.ParDo):
      examples |= _name('PostProcess') >> post_process
    elif callable(post_process):
      examples |= _name('PostProcess') >> beam.Map(post_process)
    else:
      raise ValueError(f'Unsupported post_process_map type: {post_process}')

  # Format examples to the final schema.
  examples |= _name('ProcessExample') >> beam.Map(
      ee_export_utils.process_example, config
  )

  return examples


def _get_metadata_keys(config: ml_collections.ConfigDict) -> list[str]:
  """Returns a list of metadata keys to be exported."""
  ee_algos_with_ts = [ee_algo.ic_sample, ee_algo.ic_sample_date_ranges]
  ee_algos_with_bands = [
      ee_algo.ic_sample, ee_algo.ic_sample_date_ranges,
      ee_algo.ic_sample_reduced, ee_algo.sample_roi,
      ee_algo.fc_to_image, ee_algo.get_image_from_item,
  ]

  metadata_keys = []
  for name, cfg in config.sources.items():
    source_ee_algo = pipelines.get_algo_from_config(config, name)
    # Algorithms that always returns timestamps.
    if source_ee_algo in ee_algos_with_ts:
      metadata_keys.append(f'{name}_timestamps')
    # Algorithms that produce image data that have band names.
    if source_ee_algo in ee_algos_with_bands:
      metadata_keys.append(f'{name}_bands')
    # Additional properties requested.
    if additional_properties := cfg.get('sampling_kw', {}).get(
        'additional_properties'
    ):
      metadata_keys += [f'{name}_{p}' for p in additional_properties]
  return metadata_keys


def _is_time_varying_algo(config: ml_collections.ConfigDict, name: str) -> bool:
  """Returns True if the given source is time varying."""
  if name not in config.sources:
    return False
  source_ee_algo = pipelines.get_algo_from_config(config, name)
  return source_ee_algo == ee_algo.ic_sample


def _get_sample_data(labels, config, ee_project):
  """Returns a single sample with data for the given labels and config."""
  if config.get('post_process_map'):
    # A postprocessing map allows to skip labels (e.g. due to missing data) so
    # we allow for some retries.
    num_retries = config.labels.get('num_max_samples_for_tfds_features', 10)
    for i in range(num_retries):
      try:
        data = ee_export_utils.process_single_item(
            labels[i], config, ee_project
        )
        return data
      except Exception as e:  # pylint: disable=broad-except
        logging.exception('Failed to process sample %s: %s', i, e)
        continue
    raise ValueError(f'Failed to process the first {num_retries} samples.')
  else:
    return ee_export_utils.process_single_item(labels[0], config, ee_project)


def make_tfds_features(
    config: ml_collections.ConfigDict, ee_project: str, split: str = 'full'
) -> dict[str, Any]:
  """Converts features description to TFDS form."""
  config = copy.deepcopy(config)
  # It is possible to provide wildcards in label paths.
  if not (paths := gfile.Glob(config.labels.path.replace('{split}', split))):
    raise ValueError(f'Invalid labels path: {config.labels.path}')
  config.labels.path = paths[0]
  labels = pipelines.pipeline_labels(config)
  assert labels
  data = _get_sample_data(labels, config, ee_project)

  out = {}

  metadata_keys = _get_metadata_keys(config)
  for k, v in data.items():
    if k in list(config.get('skip_keys', [])) + ['split', '__path']:
      continue
    source = k.replace('_mask', '')

    tfds_type = tfds.features.Tensor
    if source in config.sources and 'tfds_type' in config.sources[source]:
      tfds_type = getattr(tfds.features, config.sources[source].tfds_type)

    v = np.array(v)
    if source in config.sources and 'shape' in config.sources[source]:
      shape = config.sources[source].shape
    elif _is_time_varying_algo(config, source):
      shape = v.shape[-3:]
    else:
      shape = v.shape

    dtype = v.dtype
    labels_config = getattr(config, 'labels', {})
    var_len_features = labels_config.get('variable_length_features', [])

    if k in metadata_keys:
      if k.endswith('_bands'):
        out[k] = tfds.features.Sequence(tfds.features.Text())
      else:
        out[k] = tfds.features.Sequence(tfds_type(shape=(), dtype=dtype))
    elif k in var_len_features:
      out[k] = tfds.features.Sequence(tfds_type(shape=v.shape[1:], dtype=dtype))
    elif _is_time_varying_algo(config, source):
      # Assume that width, height and channels are the same for all plots.
      out[k] = tfds.features.Sequence(tfds_type(shape=shape, dtype=dtype))
    else:
      if v.shape:
        out[k] = tfds_type(shape=shape, dtype=dtype)
      else:
        out[k] = tfds.features.Scalar(dtype)
  return out


def _get_split_tuples(splits):
  """Constructs split tuples of TFDS split names and split names in data."""
  split_tuples = []
  for split in splits:
    if '=' in split:
      split_tuples.append(split.split('='))
    elif split == 'validation':
      split_tuples.append(('validation', 'val'))
    else:
      split_tuples.append((split, split))
  return split_tuples


class CreateData(beam.DoFn):
  """Beam DoFn to create an initalial collection of features grouped."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      split: str,
      ee_project: str,
      split_column_name: str = 'split',
      splits_s2_cell_level: int = 9,
      splits_ratios: str = '8:1:1',
      drop_polar: bool = False,
  ):
    self.config = config
    self.split = split
    self.ee_project = ee_project
    self.split_column_name = split_column_name
    self.splits_s2_cell_level = splits_s2_cell_level
    self.splits_ratios = splits_ratios
    self.drop_polar = drop_polar
    self.num_labels = beam.metrics.Metrics.counter(self.__class__, 'num_labels')
    self.num_skipped_polar = beam.metrics.Metrics.counter(
        self.__class__, 'num_skipped_polar'
    )

  def start_bundle(self):
    # Initialize and authenticate EE.
    utils.initialize_ee(self.ee_project)

  def process(self, path):
    config = copy.deepcopy(self.config)
    config.labels.path = path
    # Disable caching since we are running multithreading experiments and
    # caching sometimes fails since multiple threads try to write the same
    # file.
    df = pipelines.get_labels_df(config, cache=False)
    assert '__path' not in df.columns
    df['__path'] = path
    label_items = pipelines.pipeline_labels(config, df)

    # Filter out polar labels outside the UTM grid limits [-80, 84].
    # UTM projection is not defined for these regions, and attempting to
    # convert them raises utm.error.OutOfRangeError.
    polar_items = [
        ex for ex in label_items
        if 'lat' in ex and not (-80.0 <= ex['lat'] <= 84.0)
    ]
    if polar_items:
      if self.drop_polar:
        self.num_skipped_polar.inc(len(polar_items))
        label_items = [ex for ex in label_items if ex not in polar_items]
      else:
        raise ValueError(
            f'Found {len(polar_items)} labels with latitude outside UTM '
            f'range [-80, 84]. Set drop_polar=True to silently drop them, '
            f'or filter your input labels. First offending label: '
            f'lat={polar_items[0].get("lat")}, lon={polar_items[0].get("lon")}'
        )

    if self.split != 'full':

      def _filter_example(ex):
        if self.split_column_name in ex:
          # Allow validation examples to be in either 'val' or 'validation'.
          if self.split in ['val', 'validation']:
            return ex[self.split_column_name] in ['val', 'validation']
          return ex[self.split_column_name] == self.split
        else:
          # Compute s2-cell patch id (int64)
          # Levels: http://s2geometry.io/resources/s2cell_statistics.html
          s2c = coords.latlon_to_s2(
              ex['lat'], ex['lon'], level=self.splits_s2_cell_level
          )  # L9: 14-20km.
          s2hash = int(hashlib.md5(str(s2c).encode('utf-8')).hexdigest(), 16)

          train_ratio, val_ratio, test_ratio = [
              int(x) for x in self.splits_ratios.split(':')
          ]

          total = test_ratio + val_ratio + train_ratio

          s2hash_mod = s2hash % total
          if self.split == 'test' and s2hash_mod < test_ratio:
            return True
          elif (
              self.split in ['val', 'validation']
              and s2hash_mod < test_ratio + val_ratio
          ):
            return True
          elif self.split == 'train' and s2hash_mod >= test_ratio + val_ratio:
            return True
          return False

      label_items = list(filter(_filter_example, label_items))

    self.num_labels.inc(len(label_items))
    if not label_items and path == self.config.labels.path:
      raise ValueError(
          f'No examples identified for split `{self.split}` '
          f'(full labels pd.DataFrame shape: {df.shape}). '
          'Possibly due to random geographic sampling.'
      )
    return label_items


class TFDSBuilder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      name: str,
      data_dir: str,
      splits: list[str],
      ee_project: str,
      split_column_name: str = 'split',
      splits_s2_cell_level: int = 9,
      splits_ratios: str = '8:1:1',
  ):
    for k in config.sources:
      if k.endswith('_mask'):
        raise ValueError(
            f'Sources must not end with the reserved postfix `_mask`: {k}'
        )

    self.config = config
    self.splits = splits
    self.ee_project = ee_project
    self.split_column_name = split_column_name
    self.splits_s2_cell_level = splits_s2_cell_level
    self.splits_ratios = splits_ratios

    # Set name and VERSION and BUILDER_CONFIG before parent initialization.
    version = '0.0.1'  # Setting a version is required.
    if ':' in name:
      name, version = name.split(':')
    self.VERSION = tfds.core.Version(version)  # pylint: disable=invalid-name
    if '/' in name:  # Builder config is optional.
      name, builder_config_name = name.split('/')
      self.BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=builder_config_name)]  # pylint: disable=invalid-name
    self.name = name  # pyrefly: ignore[read-only]
    super().__init__(data_dir=data_dir)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(
            make_tfds_features(
                self.config,
                self.ee_project,
                split=self.splits[0] if self.splits else 'full',
            )
        ),
        supervised_keys=None,  # Set to `None` to disable.
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    if self.splits and self.splits[0]:
      return {
          split: self._generate_examples(split_name)
          for (split, split_name) in _get_split_tuples(self.splits)
      }
    else:
      return {'full': self._generate_examples('full')}

  def _generate_examples(self, split):
    """Generate examples as dicts."""
    paths = gfile.Glob(self.config.labels.path.replace('{split}', split))
    items_pcoll = 'Create plots file paths' >> beam.Create(
        paths
    ) | 'Create EE data' >> beam.ParDo(
        CreateData(
            self.config,
            split,
            self.ee_project,
            self.split_column_name,
            self.splits_s2_cell_level,
            self.splits_ratios,
        )
    )
    return build_gee_fetch_pipeline(
        items_pcoll=items_pcoll,
        config=self.config,
        ee_project=self.ee_project,
        prefix=split,
    )
