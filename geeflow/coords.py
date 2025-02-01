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

"""Working with coordinates."""

import dataclasses
from typing import Sequence

import numpy as np
import shapely.geometry
import utm as utm_lib

import ee
DEG_AT_EQUATOR_IN_M = 111_111.111


def get_lat_lon_roi(lat, lon, width_m=None, width_deg=None):
  width_deg = width_deg or width_m / DEG_AT_EQUATOR_IN_M
  delta = width_deg / 2
  roi = ee.Geometry.Rectangle([lon - delta, lat - delta,
                               lon + delta, lat + delta])
  return roi


@dataclasses.dataclass
class UtmGridMapping:
  """Universal Transverse Mercator (UTM) Grid Mapping.

  To convert world coordinates (lat-lon) to projected pixel coordinates in UTM.
  Based on (internal link)/cityscan/utm_grid/proto/utm_grid.proto?l=20
  """
  utm_zone: str  # UTM Zone (A-M: Southern, N-Z: Northern hemisphere).
  cell_size: float  # Pixel size, in meters.
  width: int  # East-west dimension, in grid cells.
  height: int  # North-south dimension, in grid cells.
  # Left bottom corner of discrete pixels.
  utm_x_min: float = 0.0  # Minimum easting.
  utm_y_min: float = 0.0  # Minimum northing.

  def __post_init__(self):
    self.utm_x_min = round(self.utm_x_min / self.cell_size) * self.cell_size
    self.utm_y_min = round(self.utm_y_min / self.cell_size) * self.cell_size

  @classmethod
  def from_bbox(cls, utm_zone: str, cell_size: float, bbox: Sequence[float],
                check_not_latlon: bool = True):
    if check_not_latlon:  # A weak test that bbox is not in lat-lon.
      assert max(np.abs(bbox)) > 180.0
    x0, y0, x1, y1 = bbox  # (west, south, east, north).
    width = int((x1 - x0) / cell_size)
    height = int((y1 - y0) / cell_size)
    return cls(utm_zone, cell_size, width, height, x0, y0)

  @classmethod
  def from_latlon_center(cls, lat: float, lon: float, cell_size: float,
                         width: int, height: int | None = None):
    height = height or width
    easting, northing, zone_number, zone_letter = utm_lib.from_latlon(lat, lon)
    utm_zone = f"{zone_number}{zone_letter}"
    x0 = easting - cell_size * width / 2.
    y0 = northing - cell_size * height / 2.
    return cls(utm_zone, cell_size, width, height, x0, y0)

  @property
  def grid_cell_size(self) -> float:
    return self.cell_size

  @property
  def epsg(self) -> str:
    northern_hemisphere = self.utm_zone[-1].upper() >= "N"
    longitude_band = int(self.utm_zone[:-1])
    return f"EPSG:32{6 if northern_hemisphere else 7}{longitude_band:02}"

  @property
  def crs(self) -> tuple[float, float, float, float, float, float]:
    return (
        self.cell_size,  # x scale.
        0.0,  # x shear.
        self.utm_x_min,  # x translation.
        0.0,  # y shear.
        -self.cell_size,  # y scale (y is going down from the top).
        self.utm_y_min + self.cell_size * self.height  # y translation.
        )

  @property
  def centroid(self) -> tuple[float, float]:
    return (
        self.utm_x_min + (self.width * self.cell_size) / 2.0,
        self.utm_y_min + (self.height * self.cell_size) / 2.0,
    )  # (x, y)

  @property
  def centroid_latlon(self) -> tuple[float, float]:
    y0, x0, y1, x1 = self.bbox_latlon
    return ((y1 + y0) / 2.0, (x1 + x0) / 2.0)  # (y, x) = (lat, lon)

  @property
  def bbox(self) -> tuple[float, float, float, float]:
    return (
        self.utm_x_min,
        self.utm_y_min,
        self.utm_x_min + (self.width * self.cell_size),
        self.utm_y_min + (self.height * self.cell_size),
    )  # (x0, y0, x1, y1)

  @property
  def bbox_latlon(self) -> tuple[float, float, float, float]:
    south, west = utm_lib.to_latlon(self.utm_x_min, self.utm_y_min,
                                    int(self.utm_zone[:-1]), self.utm_zone[-1],
                                    strict=False)
    north, east = utm_lib.to_latlon(
        self.utm_x_min + self.width * self.cell_size,
        self.utm_y_min + self.height * self.cell_size,
        int(self.utm_zone[:-1]), self.utm_zone[-1],
        strict=False)
    return south, west, north, east  # (y0, x0, y1, x1)

  def rescale(self, cell_size: float) -> "UtmGridMapping":
    return UtmGridMapping(
        self.utm_zone,
        cell_size,
        int(self.width * self.cell_size / cell_size),
        int(self.height * self.cell_size / cell_size),
        # Round x/y min to closest new cell size (to keep pixels fixed for given
        # grid size).
        utm_x_min=round(self.utm_x_min / cell_size) * cell_size,
        utm_y_min=round(self.utm_y_min / cell_size) * cell_size)

  def to_ee_point(self, utm: bool = False) -> ee.geometry.Geometry:
    if utm:
      return ee.Geometry.Point(self.centroid, proj=self.epsg)
    return ee.Geometry.Point(self.centroid_latlon)

  def to_ee(self, utm: bool = False) -> ee.Geometry:
    """Returns ee.Geometry.Rectangle in LatLon or in UTM."""
    if utm:
      x0, y0 = self.utm_x_min, self.utm_y_min
      x1 = self.utm_x_min + self.cell_size * self.width
      y1 = self.utm_y_min + self.cell_size * self.height
      return ee.Geometry.Rectangle(
          [x0, y0, x1, y1], proj=self.epsg, geodesic=False)
    y0, x0, y1, x1 = self.bbox_latlon
    return ee.Geometry.Rectangle([x0, y0, x1, y1])

  def to_ee_bbox(self) -> ee.Geometry:
    # EE BBox is only possible in LatLon.
    south, west, north, east = self.bbox_latlon
    return ee.Geometry.BBox(west, south, east, north)

  def to_shapely_polygon(self) -> shapely.geometry.Polygon:
    # Only LatLon for now.
    y0, x0, y1, x1 = self.bbox_latlon
    return shapely.geometry.Polygon([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
