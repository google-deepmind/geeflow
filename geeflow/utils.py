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

"""Utils."""

from collections.abc import Callable
import math
import os
from typing import Any

from absl import logging
import jax
import ml_collections as mlc
import numpy as np

from tensorflow.io import gfile
import ee


LAT_TO_METERS = 111694  # From https://en.wikipedia.org/wiki/Latitude
LON_TO_METERS = 112000


def get_utm_grid_size(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    img_width_m: float,
    cell_expansion_offset_meters: int = 5000,
) -> tuple[int, int]:
  """Returns the grid dimensions for the given UTM zone."""
  x_num = math.ceil(LON_TO_METERS * max(
      math.cos(start_lat / 180 * math.pi), math.cos(end_lat / 180 * math.pi),
  ) * (end_lon - start_lon) / img_width_m)
  # Since UTM and Lat/Lon grids are not aligned, we add a small offset to ensure
  # that we'll be able to get all the cells within the UTM zone.
  # Here's an example of the problem:
  #  - https://code.earthengine.google.com/71fd4250184e7c041f13516959fe9d11
  y_num = math.ceil((LAT_TO_METERS * (end_lat - start_lat) +
                     cell_expansion_offset_meters) / img_width_m)
  return x_num, y_num


def initialize_ee(ee_project: str) -> None:
  """Initialize and authenticate earth engine."""
  if ee_project == "skip":  # Skip initializing Earth Engine.
    return
  ee.Initialize(project=ee_project)


def to_np(x: Any) -> np.ndarray:
  """Converts `x` to numpy (including bytes to strings and nested dicts)."""
  if isinstance(x, dict):
    return jax.tree.map(to_np, x)
  x = np.array(x)
  if x.dtype == "O":
    x = x.tolist()
    if isinstance(x, list):
      x = [y.decode() for y in x]
    else:
      x = x.decode()
  return np.asarray(x)


def cache_data(
    src: str, dst_dir: str | None = None, sub_dir: str = "", **kwargs
) -> str:
  """Caches file or dir locally and returns path to new local location."""
  dst_dir = dst_dir or "/tmp/geeflow_cache"
  if sub_dir:
    dst_dir = os.path.join(dst_dir, sub_dir)
  return _copy(src, dst_dir, **kwargs)


def _copy(src: str, dst_dir: str) -> str:
  """Copies file or directory to destination dir and returns dst path."""
  if src.endswith("/"):
    src = src[:-1]
  if dst_dir.endswith("/"):
    dst_dir = dst_dir[:-1]
  os.umask(0o022); gfile.makedirs(dst_dir)
  if not gfile.IsDirectory(src):
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not gfile.exists(dst):
      logging.info("Copy %s to %s", src, dst)
      gfile.Copy(src, dst)
    else:
      logging.info("Skip copy (file exists): %s", dst)
  else:
    dst = os.path.join(dst_dir, os.path.basename(src))
    for src_file in gfile.Glob(os.path.join(src, "*")):
      _copy(src_file, dst)
  return dst


def parse_arg(arg: str | None, lazy: bool = False, **spec) -> mlc.ConfigDict:
  """Makes ConfigDict's get_config single-string argument more usable.

  Example use in the config file:

    def get_config(arg):
      arg = utils.parse_arg(arg,
          res=(224, int),
          runlocal=False,
          schedule="short",
      )

      # ...

      config.shuffle_buffer = 250_000 if not arg.runlocal else 50

  Ways that values can be passed when launching:

    --config amazing.py:runlocal,schedule=long,res=128
    --config amazing.py:res=128
    --config amazing.py:runlocal  # A boolean needs no value for "true".
    --config amazing.py:runlocal=False  # Explicit false boolean.
    --config amazing.py:128  # The first spec entry may be passed unnamed alone.

  Uses strict bool conversion (converting "True", "true" to True, and "False",
    "false", "" to False).

  Args:
    arg: the string argument that"s passed to get_config.
    lazy: allow lazy parsing of arguments, which are not in spec. For these,
      the type is auto-extracted in dependence of most complex possible type.
    **spec: the name and default values of the expected options.
      If the value is a tuple, the value"s first element is the default value,
      and the second element is a function called to convert the string.
      Otherwise the type is automatically extracted from the default value.

  Returns:
    ConfigDict object with extracted type-converted values.
  """
  # Normalize arg and spec layout.
  arg = arg or ""  # Normalize None to empty string.
  spec = {k: get_type_with_default(v) for k, v in spec.items()}

  result = mlc.ConfigDict(type_safe=False)  # For convenient dot-access only.

  # Expand convenience-cases for a single parameter without = sign.
  if arg and "," not in arg and "=" not in arg:
    # (think :runlocal) If it"s the name of sth in the spec (or there is no
    # spec), it"s that in bool.
    if arg in spec or not spec:
      arg = f"{arg}=True"
    # Otherwise, it is the value for the first entry in the spec.
    else:
      arg = f"{list(spec.keys())[0]}={arg}"
      # Yes, we rely on Py3.7 insertion order!

  # Now, expand the `arg` string into a dict of keys and values:
  raw_kv = {raw_arg.split("=")[0]:
                raw_arg.split("=", 1)[-1] if "=" in raw_arg else "True"
            for raw_arg in arg.split(",") if raw_arg}

  # And go through the spec, using provided or default value for each:
  for name, (default, type_fn) in spec.items():
    val = raw_kv.pop(name, None)
    result[name] = type_fn(val) if val is not None else default

  if raw_kv:
    if lazy:  # Process args which are not in spec.
      for k, v in raw_kv.items():
        result[k] = autotype(v)
    else:
      raise ValueError(f"Unhandled config args remain: {raw_kv}")

  return result


def get_type_with_default(v: Any) -> tuple[Any, Callable[[Any], Any]]:
  """Returns (v, string_to_v_type) with lenient bool parsing."""
  # For bool, do safe string conversion.
  if isinstance(v, bool):
    def strict_bool(x):
      assert x.lower() in {"true", "false", ""}
      return x.lower() == "true"
    return (v, strict_bool)
  # If already a (default, type) tuple, use that.
  if isinstance(v, (tuple, list)):
    assert len(v) == 2 and isinstance(v[1], type), (
        "List or tuple types are currently not supported because we use `,` as"
        " dumb delimiter. Contributions (probably using ast) welcome. You can"
        " unblock by using a string with eval(s.replace(';', ',')) or similar")
    return (v[0], v[1])
  # Otherwise, derive the type from the default value.
  return (v, type(v))


def autotype(x: str) -> Any:
  """Auto-converts string to bool/int/float if possible."""
  if x.lower() in {"true", "false"}:
    return x.lower() == "true"  # Returns as bool.
  try:
    return int(x)  # Returns as int.
  except ValueError:
    try:
      return float(x)  # Returns as float.
    except ValueError:
      return x  # Returns as str.


def get_source_config(module: str, out: str) -> mlc.ConfigDict:
  """Return a dummy ConfigDict for a GEE source."""
  return mlc.config_dict.create(
      module=module,
      out=out,
      kw={},
      out_kw={},
      select=None,
      select_final=None,
      sampling_kw={},
      scale=None,
  )


def standardized_path(
    path,
    split_name=None,
    postfix=None,
    default_dir=None,
    file_extension=".json",
):
  """Constructs/adjusts full path for files."""
  # To use a standardized way with TFDS version:
  #   standardized_path(tfds_info.full_name, split_name)
  # Optionally add a postfix, eg. var name or number of sampled elements.
  if not path.startswith("/"):
    path = path.replace(":", "/")  # In case it's a tfds name string.
    if default_dir is not None:
      path = os.path.join(default_dir, path)
  if split_name:
    path = os.path.join(path, split_name)
  if postfix:
    if path.endswith("/"):
      path = os.path.join(path, postfix)
    else:
      path = f"{path}_{postfix}"
  if not path.endswith(file_extension):
    path += file_extension
  return path
