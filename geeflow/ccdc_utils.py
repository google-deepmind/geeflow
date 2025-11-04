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

"""CCDC processing utils."""
import functools

from ml_collections import config_dict
import numpy as np

import ee

_COEFS_DIM = 8


def im_add_ccdc_bands_2d(ic, num_segments, im=None) -> ee.Image:
  """Pads and adds CCDC bands with shape (num_segments, 8)."""
  zeros = ee.Array([0]).repeat(0, num_segments).repeat(1, 8)
  for j in range(num_segments):
    for i in range(8):
      im_ji = (
          ic.unmask(zeros, False)
          .arrayCat(zeros, 0)
          .float()
          .arraySlice(0, 0, num_segments)
          .arrayGet([j, i])
      )
      rename_fn = functools.partial(
          lambda n, idx: ee.String(n).cat(ee.String(idx)), idx=f"#{j}#{i}"
      )
      im_ji = im_ji.rename(im_ji.bandNames().map(rename_fn))
      im = im_ji if im is None else im.addBands(im_ji)
  return im


def im_add_ccdc_bands_1d(ic, num_segments, im=None) -> ee.Image:
  """Pads and adds CCDC bands with shape (num_segments,)."""
  zeros = ee.Array([0]).repeat(0, num_segments)
  for j in range(num_segments):
    im_i = ic.unmask(zeros, False).arrayCat(zeros, 0).float().arrayGet(j)
    rename_fn = functools.partial(
        lambda n, idx: ee.String(n).cat(ee.String(idx)), idx=f"#{j}#0"
    )
    im_i = im_i.rename(im_i.bandNames().map(rename_fn))
    im = im_i if im is None else im.addBands(im_i)
  return im


def get_ccdc_pixels(
    pixels: dict[str, np.ndarray], cfg: config_dict.ConfigDict, key: str
) -> dict[str, np.ndarray]:
  """Reconstructs CCDC bands with shape (H, W, T, C)."""
  ccdc_pixels = {}
  for b in cfg.select:
    if b.endswith("_coefs"):
      tmp = [[pixels.pop(f"{key}_{b}#{j}#{i}")
              for j in range(cfg.sampling_kw.num_segments)]
             for i in range(_COEFS_DIM)]
    else:
      tmp = [[pixels.pop(f"{key}_{b}#{j}#0")
              for j in range(cfg.sampling_kw.num_segments)]]
    ccdc_pixels[f"{key}_{b}"] = np.transpose(tmp, (2, 3, 1, 0))  # (H, W, T, C)
  return ccdc_pixels


def generate_ccdc(
    data: dict[str, np.ndarray], cfg: config_dict.ConfigDict, key: str
) -> tuple[np.ndarray, np.ndarray]:
  """Generates CCDC data in (T,H,W,C) format."""
  ccdc = [data[f"{key}_{b}"] for b in cfg.select]
  ccdc = np.concatenate(ccdc, axis=-1)  # (H, W, T, C)
  ccdc = np.transpose(ccdc, (2, 0, 1, 3))  # (T, H, W, C)
  if "format_config" not in cfg:
    return ccdc, ccdc[..., cfg.select.index("tStart")] > 0

  # Annual CCDC segments extraction.
  num_segments, h, w, num_bands = ccdc.shape
  start_dates = ccdc[..., cfg.select.index("tStart")]  # (T, H, W)
  years = np.arange(cfg.format_config["from"], cfg.format_config["to"] + 1)

  if cfg.format_config.get("selection", "longest") == "longest":
    start_dates_within_year = np.maximum(
        years[:, None, None, None],
        start_dates[None, ...])
    end_dates = ccdc[..., cfg.select.index("tEnd")]  # (T, H, W)
    end_dates_within_year = np.minimum(
        (years + 1)[:, None, None, None],
        end_dates[None, ...])
    durations = np.maximum(0, end_dates_within_year - start_dates_within_year)
    final_indices = np.argmax(durations, axis=1)  # (Y, H, W)
  else:  # Last segment that started before the middle of each year.
    # For this, identify the first segment that started *after* the target date,
    # and then take the index just before it.
    target_dates = years[:, None, None, None] + 0.5  # Middle of each year.
    is_after = start_dates[None, ...] > target_dates  # (Y, T, H, W)
    # Pad segments with one more segment with True values.
    padding = np.ones((len(years), 1, h, w), dtype=bool)
    is_after = np.concatenate([is_after, padding], axis=1)
    first_after = np.argmax(is_after, axis=1)  # (Y, H, W)
    # Get the last segment *before* the first after the target date.
    final_indices = np.maximum(0, first_after - 1)  # (Y, H, W)
  final_indices = final_indices.reshape(len(years), h * w, 1)  # (Y, H*W, 1)

  ccdc = ccdc.reshape(num_segments, h * w, num_bands)  # (T, H*W, C)
  ccdc = np.take_along_axis(ccdc, final_indices, axis=0)  # (Y, H*W, C)
  ccdc = ccdc.reshape(-1, h, w, num_bands)  # (Y, H, W, C)
  ccdc_mask = ccdc[..., cfg.select.index("tStart")] > 0

  if "year_selection" in cfg.format_config:
    ccdc = ccdc[cfg.format_config["year_selection"]]
    ccdc_mask = ccdc_mask[cfg.format_config["year_selection"]]
  return ccdc, ccdc_mask
