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

"""Mock classes for EarthEngine objects."""
import numpy as np

import ee


class MockEEArray:
  """Mock EE Array."""

  def __init__(self, data):
    self.data = np.array(data)

  def repeat(self, axis, count):
    return MockEEArray(np.repeat(self.data[np.newaxis], count, axis=axis))

  def __array__(self):
    return self.data

  def reshape(self, shape):
    return MockEEArray(self.data.reshape(shape))

  def shape(self):
    return self.data.shape


class MockEEList:
  """Mock EE List."""

  def __init__(self, data):
    self.data = list(data)

  def map(self, func):
    return MockEEList([func(x) for x in self.data])

  def __iter__(self):
    return iter(self.data)

  def __toList__(self):    # pylint: disable=invalid-name
    return self.data


class MockEEString:
  """Mock EE String."""

  def __init__(self, data):
    self.data = str(data)

  def cat(self, other):
    return MockEEString(self.data + str(other))

  def __str__(self):
    return self.data

  def __eq__(self, other):
    return self.data == str(other)

  def __hash__(self):
    return hash(self.data)


class MockEEImage:
  """Mock EE Image."""

  def __init__(self, bands=None):
    if isinstance(bands, MockEEImage):
      self.bands = bands.bands.copy()
    elif isinstance(bands, dict):
      self.bands = bands
    else:
      self.bands = {}

  def bandNames(self):  # pylint: disable=invalid-name
    return MockEEList(self.bands.keys())

  def rename(self, new_names):
    if isinstance(new_names, list):
      self.bands = dict(zip(new_names, self.bands.values()))
    return self

  def addBands(self, other):  # pylint: disable=invalid-name
    self.bands.update(other.bands)
    return self

  def unmask(self, value, newMask):  # pylint: disable=invalid-name, unused-argument
    return self

  def arrayCat(self, array, axis):  # pylint: disable=invalid-name, unused-argument
    return self

  def float(self):
    return self

  def arraySlice(self, axis, start, end):  # pylint: disable=invalid-name, unused-argument
    return self

  def arrayGet(self, position):  # pylint: disable=invalid-name, unused-argument
    return self

  def select(self, bands):
    return MockEEImage({k: v for k, v in self.bands.items() if k in bands})


# Mock EE classes automatically (to be used in tests only).
ee.Image = MockEEImage
ee.Array = MockEEArray
ee.String = MockEEString
ee.List = MockEEList
