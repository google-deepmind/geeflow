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

"""Config validation."""

import importlib

from absl import logging
from geeflow import ee_data
import ml_collections as mlc


# The canonical set of keys that pipeline_sources() and get_requests_fn()
# actually read from source config entries.
SOURCE_CONFIG_KEYS = frozenset({
    # Core identity.
    "module",
    "kw",
    "out",
    "out_kw",
    # Band selection.
    "select",
    "select_final",
    # Spatial.
    "scale",
    "img_width_m",
    "img_width_deg",
    "crs",
    "scalar",
    # Temporal.
    "date_ranges",
    "date_range",
    "start_date",
    "end_date",
    "date_ranges_fn",
    "date_range_fn",
    "start_date_fn",
    "end_date_fn",
    "filter_date",
    # Sampling.
    "algo",
    "sampling_kw",
    "limit",
    # Output processing.
    "cast",
    "mask_value",
    "dtype",
    "dummy_image_id",
    "tfds_type",
    # Request splitting.
    "split_dates_into_separate_requests",
    # CCDC-specific.
    "format_config",
    # get_image_from_item algo.
    "asset_id_key",
})

# The canonical set of keys that the pipeline reads from config.labels.
LABEL_CONFIG_KEYS = frozenset({
    # File path to the labels CSV / Parquet.
    "path",
    # Subsetting.
    "num_max_samples",
    "cache",
    "meta_keys",
    # Spatial.
    "use_utm",
    "img_width_m",
    "img_width_deg",
    "max_cell_size_m",
    "default_scale",
    # Output processing.
    "ignore_for_float_conversion",
    "tfds_id_keys",
})

# The canonical set of top-level config keys.
TOP_LEVEL_CONFIG_KEYS = frozenset({
    "sources",
    "labels",
    # Controls whether unknown source/label keys raise an error.
    "fail_if_unknown_keys",
    # Export filters.
    "export",
    # Keys to drop before writing examples.
    "skip_keys",
    # Optional post-processing Beam transform.
    "post_process_map",
})


def resolve_source_class(module_str: str):
  """Resolves a module string to a data source class.

  Supports:
    - "ClassName" -> looks up ClassName in ee_data (backward compatible).
    - "submodule.ClassName" -> imports from geeflow.data_sources.submodule.
    - "path.submodule.ClassName" -> imports from the given (global imprort) path
      (for data in custom directories).

  Args:
    module_str: The module string, e.g. "Sentinel2" or "sar.Sentinel1".

  Returns:
    The data source class.
  """
  match len(parts := module_str.rsplit(".")):
    case 1:
      submodule, class_name = ee_data, module_str
    case 2:
      submodule, class_name = parts
      submodule = importlib.import_module(f"geeflow.data_sources.{submodule}")
    case _:
      *submodule, class_name = parts
      submodule = importlib.import_module(".".join(submodule))
  return getattr(submodule, class_name)


def validate_sources_config(config: mlc.ConfigDict) -> None:
  """Validates source config entries for required fields and known keys.

  Args:
    config: The full config with a 'sources' sub-config.

  Raises:
    ValueError: If a source is missing required fields or has invalid values.
  """
  errors = []
  for name, src in config.sources.items():
    # Check required 'module' field.
    if "module" not in src:
      errors.append(f"Source '{name}' is missing required field 'module'.")
    elif isinstance(src.module, str):
      try:
        resolve_source_class(src.module)
      except (AttributeError, ModuleNotFoundError, ImportError):
        errors.append(
            f"Source '{name}' has module='{src.module}' which cannot be "
            "resolved. Check for typos."
        )

    # Check required 'algo' field.
    if "algo" not in src or src.get("algo") is None:
      # Allow EE asset passthrough (module is an EE object, not a string).
      if "module" in src and not isinstance(src.module, str):
        pass  # Direct EE asset — algo is applied differently.
      else:
        errors.append(
            f"Source '{name}' has no 'algo' set. It must be set to a valid "
            "sampling algorithm (e.g. 'ic_sample_date_ranges', 'sample_roi')."
        )

    # Warn about unknown keys (typo catcher) — don't error for flexibility.
    unknown_keys = set(src.keys()) - SOURCE_CONFIG_KEYS
    if unknown_keys:
      logging.warning(
          "Source '%s' has unknown config keys: %s.", name, unknown_keys)
      if config.get("fail_if_unknown_keys", True):
        errors.append(
            f"Source '{name}' has unknown config keys: {unknown_keys}.")

  if errors:
    raise ValueError("Source config validation failed:\n " + "\n ".join(errors))


def validate_labels_config(config: mlc.ConfigDict) -> None:
  """Validates label config entries for known keys.

  Args:
    config: The full config with a 'labels' sub-config.

  Raises:
    ValueError: If config.labels has unknown keys.
  """
  if "labels" not in config or not config.labels:
    return
  errors = []
  unknown_keys = set(config.labels.keys()) - LABEL_CONFIG_KEYS
  if unknown_keys:
    logging.warning("Labels config has unknown keys: %s.", unknown_keys)
    if config.get("fail_if_unknown_keys", True):
      errors.append(f"Labels config has unknown keys: {unknown_keys}.")
  if errors:
    raise ValueError(
        "Labels config validation failed:\n " + "\n ".join(errors)
    )


def validate_top_level_config(config: mlc.ConfigDict) -> None:
  """Validates that the top-level config only contains known keys.

  Args:
    config: The full config.

  Raises:
    ValueError: If the config has unknown top-level keys.
  """
  errors = []
  unknown_keys = set(config.keys()) - TOP_LEVEL_CONFIG_KEYS
  if unknown_keys:
    logging.warning("Config has unknown top-level keys: %s.", unknown_keys)
    if config.get("fail_if_unknown_keys", True):
      errors.append(f"Config has unknown top-level keys: {unknown_keys}.")
  if errors:
    raise ValueError(
        "Top-level config validation failed:\n " + "\n ".join(errors)
    )


def validate_config(config: mlc.ConfigDict) -> None:
  """Validates the entire config, including sources, labels, etc."""
  validate_top_level_config(config)
  validate_sources_config(config)
  validate_labels_config(config)
