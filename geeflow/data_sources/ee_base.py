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

"""Abstract classes for EE datasets."""

import abc

import numpy as np

import ee


class EeData(abc.ABC):
  """Abstract class for EE ImageCollection and Image datasets."""

  BANDS = []  # Ordered list of bands.
  VIS_BANDS = []  # Bands for RGB visualization.

  @property
  @abc.abstractmethod
  def asset_name(self) -> str:
    """Returns asset name."""

  @classmethod
  def stack(cls, data, vis=False) -> np.ndarray:
    """Returns HWC numpy array with C=(R,G,B,[N])."""
    bands = cls.VIS_BANDS if vis else cls.BANDS
    if isinstance(data, dict):
      d = [data[band] for band in bands if band in data]
      return np.stack(d, axis=2)  # (H,W,C)
    raise ValueError(f"Unsupported data type: {type(data)}")

  @classmethod
  def vis(cls, data, **kwargs) -> np.ndarray:
    """Returns RGB or single-channel image ready for visualization."""
    img = cls.stack(data, vis=True)
    return cls.vis_norm(img, **kwargs)

  @classmethod
  def vis_norm(cls, img: np.ndarray) -> np.ndarray:
    """Visualization normalization. Can be overwriting by inheriting classes."""
    return img

  @property
  def ic(self):
    return ee.ImageCollection(self.asset_name)

  @property
  def im(self):
    return ee.Image(self.asset_name)

  @property
  def image(self):
    return ee.Image(self.asset_name)


class EeDataFC(EeData, abc.ABC):
  """Abstract class for EE FeatureCollection datasets."""

  @property
  def fc(self):
    return ee.FeatureCollection(self.asset_name)

  @property
  def ic(self):
    raise ValueError("This is a FeatureCollection.")

  @property
  def im(self):
    raise ValueError("This is a FeatureCollection.")
