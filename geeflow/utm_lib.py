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

"""A wrapper around UTM library to allow working with vectorized coordinates."""

import copy
import utm as utm_lib


def to_latlon(easting, northing, zone_number,
              zone_letter=None, northern=None, strict=True):
  return utm_lib.to_latlon(copy.deepcopy(easting), copy.deepcopy(northing),
                           zone_number, zone_letter, northern, strict)


def from_latlon(latitude, longitude, force_zone_number=None,
                force_zone_letter=None):
  return utm_lib.from_latlon(latitude, longitude,
                             force_zone_number, force_zone_letter)
