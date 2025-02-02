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

"""Stats computation utils for Beam pipelines."""

from collections.abc import Iterable, Iterator
import functools
import hashlib
from typing import Any

import apache_beam as beam
from geeflow.stats import stats_util
import numpy as np


# Number of reduction/grouping steps to compute stats.
_NUM_AGGREGATION_STEPS = 5
# How many items do we group per reduction step.
_AGGREGATION_DIVISOR = 10
# Modulo for the original group assignment.
_AGGREGATION_MODULO = _AGGREGATION_DIVISOR ** _NUM_AGGREGATION_STEPS


def _is_multi_band(name: str, feature: Any) -> bool:
  del name
  return len(np.array(feature).shape) > 1


def _is_numeric(name: str, feature: Any) -> bool:
  del name
  if isinstance(feature, str):
    return False
  if (isinstance(feature, np.ndarray) and
      feature.flat and
      isinstance(feature.flat[0], str)):
    return False
  return True


def extract_features(
    example: dict[str, Any],
) -> Iterator[tuple[tuple[str, int], tuple[Any, Any]]]:
  """Extracts features from an example."""
  d = {}
  group_id = int(
      int(hashlib.md5(str(example["id"]).encode("utf-8")).hexdigest(), 16)
      % _AGGREGATION_MODULO)
  for name, feature in example.items():
    d[name] = feature
  for name, feature in example.items():
    if _is_multi_band(name, feature):
      feature = np.array(feature)
      feature_mask = d.get(f"{name}_mask", None)
      if feature_mask is not None:
        feature_mask = np.array(feature_mask, dtype=bool)
      for i in range(feature.shape[-1]):
        yield (f"{name}_band_{i}", group_id), (
            feature[..., i],
            feature_mask[..., i] if feature_mask is not None else None)
    else:
      yield (name, group_id), (feature, d.get(f"{name}_mask", None))


def convert_to_accumulator(
    example: tuple[tuple[str, int], tuple[Any, Any]],
) -> Iterator[tuple[tuple[str, int], stats_util.CounterAccumulator]]:
  (name, group_id), (feature, mask) = example
  acc = stats_util.CounterAccumulator(numeric=_is_numeric(name, feature))
  acc.add(feature, mask)
  yield (name, group_id), acc


def reduce_key(
    example: tuple[tuple[str, int], stats_util.CounterAccumulator],
) -> Iterator[tuple[tuple[str, int], stats_util.CounterAccumulator]]:
  (name, group_id), acc = example
  yield (name, group_id // _AGGREGATION_DIVISOR), acc


def aggregate(
    example: tuple[tuple[str, int], Iterable[stats_util.CounterAccumulator]],
) -> Iterator[tuple[tuple[str, int], stats_util.CounterAccumulator]]:
  """Aggregates stats."""
  key, accs = example

  a = None
  for acc in accs:
    if a is None:
      a = acc
    else:
      a.merge(acc)

  assert a is not None
  yield key, a


def write(
    example: tuple[tuple[str, int], stats_util.StatsAccumulator],
    output_dir: str,
    # NOTE: For some unknown reason, doing "split: str | None = None" makes
    # beam fail. So be careful when changing this.
    split: str | None = None
) -> None:
  """Write stats."""
  (name, group_id), acc = example
  assert group_id == 0  # Make sure enough reduction steps were run.
  if acc is not None:
    acc.save_json(output_dir, split, name)


def create_pipeline(root: beam.Pipeline, output_dir: str,
                    split: str) -> beam.Pipeline:
  """Creates the beam pipeline for stats computation."""

  write_fn = functools.partial(
      write, output_dir=output_dir, split=split)

  p = (
      root
      | f"ExtractFeatures_{split}" >> beam.FlatMap(extract_features)
      | f"ConvertToAccumulator_{split}" >> beam.FlatMap(convert_to_accumulator)
  )
  for it in range(_NUM_AGGREGATION_STEPS):
    p = (
        p
        | f"ReduceKey_{it}_{split}" >> beam.FlatMap(reduce_key)
        | f"GroupByKey_{it}_{split}" >> beam.GroupByKey()
        | f"Aggregate_{it}_{split}" >> beam.FlatMap(aggregate)
    )
  return (
      p
      | f"Write_{split}" >> beam.FlatMap(write_fn)
  )
