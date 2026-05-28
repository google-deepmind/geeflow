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

"""Custom configuration sources."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

from geeflow.data_sources import ee_base

ee = ee_base.ee


@dataclasses.dataclass
class CustomFC(ee_base.EeDataFC):
  """A custom source that loads a FeatureCollection.

  An example:
    c.gedi = copy.deepcopy(all_sources.CustomFC)
    c.gedi.kw.asset_name =
        "projects/computing-engine-190414/assets/gedi_nico_all_predictions"
    c.gedi.select = ["canopy_height"]
    c.gedi.scale = 10
    c.gedi.algo = ee_algo.fc_to_image

  Attributes:
    asset_name: A name or a list of names of the assets to load.
    filters: A list of filters to apply to the asset.
    buffer_points: How many meters to buffer the point features.
    buffer: How many meters to buffer all features (on top of buffer_points).
    use_bounds: Whether to use bounds instead of actual geometries.
    set_property: A tuple of (property_name, property_value) to set on the
      features.
  """
  asset_name: Sequence[str] | str = ""  # Needs to be specified.
  filters: Sequence[tuple[str, Any]] | None = None
  buffer_points: int = 0
  buffer: int = 0
  # NOTE: Currently ".bounds" methiod is very slow and could incurr very
  # significant slowdown. For more context, see:
  # (internal link)
  # Only use on small collections.
  use_bounds: bool = False
  set_property: tuple[str, Any] | None = None

  @property
  def fc(self):
    if isinstance(self.asset_name, (tuple, list)):
      fc = ee.FeatureCollection(
          [ee.FeatureCollection(x) if isinstance(x, str) else CustomFC(**x).fc
           for x in self.asset_name])
      fc = fc.flatten()
    else:
      fc = ee.FeatureCollection(self.asset_name)
    if self.filters:
      for k, v in self.filters:
        if isinstance(v, (tuple, list)):
          if k.startswith("!"):
            fc = fc.filter(ee.Filter.inList(k[1:], v).Not())
          else:
            fc = fc.filter(ee.Filter.inList(k, v))
        else:
          if k.startswith("<="):
            fc = fc.filter(ee.Filter.lte(k[2:], v))
          elif k.startswith("<"):
            fc = fc.filter(ee.Filter.lt(k[1:], v))
          elif k.startswith(">="):
            fc = fc.filter(ee.Filter.gte(k[2:], v))
          elif k.startswith(">"):
            fc = fc.filter(ee.Filter.gt(k[1:], v))
          elif k.startswith("!~"):
            fc = fc.filter(ee.Filter.stringContains(k[2:], v).Not())
          elif k.startswith("~"):
            fc = fc.filter(ee.Filter.stringContains(k[1:], v))
          elif k.startswith("!"):
            fc = fc.filter(ee.Filter.neq(k[1:], v))
          else:
            fc = fc.filter(ee.Filter.eq(k, v))
    if self.buffer_points > 0:
      fc_points = fc.filter(ee.Filter.hasType(".geo", "Point"))
      fc_not_points = fc.filter(ee.Filter.hasType(".geo", "Point").Not())
      fc_points = fc_points.map(lambda x: x.buffer(self.buffer_points))
      if self.use_bounds:
        fc_points = fc_points.map(lambda x: x.bounds())
      fc = ee.FeatureCollection([fc_points, fc_not_points]).flatten()
    # NOTE: We allow for negative values too.
    if self.buffer:
      fc = fc.map(lambda x: x.buffer(self.buffer))
    if self.set_property:
      fc = fc.map(lambda x: x.set(self.set_property[0], self.set_property[1]))
    return fc


@dataclasses.dataclass
class CustomImage(ee_base.EeData):
  """A custom source that loads an image.

  An example:
    c.primary_forest = copy.deepcopy(all_sources.CustomImage)
    c.primary_forest.kw.asset_name =
        "projects/computing-engine-190414/assets/arbaro/suso/primary_forests"
  """
  asset_name: str = ""  # Needs to be specified.
  im_fn: Callable[[str], ee.Image] | None = None

  @property
  def im(self):
    if self.im_fn:
      return self.im_fn(self.asset_name)
    return ee.Image(self.asset_name)

  @property
  def ic(self):
    raise ValueError("This is considered as an Image and not a Collection.")


@dataclasses.dataclass
class CustomIC(ee_base.EeData):
  """A custom source that loads an IC.

  An example:
    c.google = copy.deepcopy(all_sources.CustomImage)
    c.google.kw.asset_name =
        "projects/satellite-segmentation/assets/labels"
  """
  # asset_name could refer to:
  #  - str: a single ImageCollection (merge should be False)
  #  - list/tuple: list of Image assets (merge should be False)
  #  - list/tuple: list of ImageCollection assets (merge should be True)
  asset_name: str | list[str] = ""  # Needs to be specified.
  merge: bool = False
  ic_fn: Callable[[str | list[str]], ee.ImageCollection] | None = None

  @property
  def im(self):
    raise ValueError("This is considered as an IC and not an Image.")

  @property
  def ic(self):
    if self.ic_fn:
      assert not self.merge, "merge should be handled by ic_fn if ic_fn given"
      return self.ic_fn(self.asset_name)
    if self.merge and not isinstance(self.asset_name, str):
      ic = ee.ImageCollection(self.asset_name[0])
      for asset_name in self.asset_name[1:]:
        ic = ic.merge(ee.ImageCollection(asset_name))
      return ic
    return ee.ImageCollection(self.asset_name)
