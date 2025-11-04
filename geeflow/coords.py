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

"""Working with coordinates."""

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any

from absl import logging
from geeflow import utm_lib
import numpy as np
import pyproj
import shapely.geometry
import shapely.ops

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

  Attributes:
    utm_zone: UTM Zone (A-M: Southern, N-Z: Northern hemisphere).
    cell_size: Pixel size, in meters.
    width: East-west dimension, in grid cells.
    height: North-south dimension, in grid cells.
    utm_x_min: Minimum easting.
    utm_y_min: Minimum northing.
    use_floor: Round or take the floor for clipping to the grid. Use it to
      ensure the given center coordinates is to the top right of the origin.
    grid_cell_size: Pixel size, in meters.
    epsg: EPSG code for the UTM projection.
    crs: Coordinate reference system (CRS) for the UTM projection.
    centroid: Centroid of the grid, in UTM coordinates.
    centroid_latlon: Centroid of the grid, in LatLon coordinates.
    bbox: Bounding box of the grid, in UTM coordinates.
    bbox_latlon: Bounding box of the grid, in LatLon coordinates.
  """
  utm_zone: str  # UTM Zone (A-M: Southern, N-Z: Northern hemisphere).
  cell_size: float  # Pixel size, in meters.
  width: int | np.ndarray  # East-west dimension, in grid cells.
  height: int | np.ndarray  # North-south dimension, in grid cells.
  # Left bottom corner of discrete pixels.
  utm_x_min: float | np.ndarray = 0.0  # Minimum easting.
  utm_y_min: float | np.ndarray = 0.0  # Minimum northing.
  use_floor: bool = False  # Round or take the floor for clipping to the grid.

  def __post_init__(self):
    fn = np.floor if self.use_floor else np.round
    self.utm_x_min = fn(self.utm_x_min / self.cell_size) * self.cell_size
    self.utm_y_min = fn(self.utm_y_min / self.cell_size) * self.cell_size

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
                         width: int, height: int | None = None,
                         use_floor: bool = False):
    height = width if height is None else height
    easting, northing, zone_number, zone_letter = utm_lib.from_latlon(lat, lon)
    utm_zone = f"{zone_number}{zone_letter}"
    x0 = easting - cell_size * width / 2.
    y0 = northing - cell_size * height / 2.
    return cls(utm_zone, cell_size, width, height, x0, y0, use_floor)

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
  def centroid_latlon(self) -> tuple[float | np.ndarray, float | np.ndarray]:
    y0, x0, y1, x1 = self.bbox_latlon
    x1 += 360 * (x1 < x0)
    x_mid = (x0 + x1) / 2.0
    x_mid = x_mid - 360 * (x_mid > 180)
    return ((y1 + y0) / 2.0, x_mid)  # (y, x) = (lat, lon)

  @property
  def bbox(self) -> tuple[float, float, float, float]:
    return (
        self.utm_x_min,
        self.utm_y_min,
        self.utm_x_min + (self.width * self.cell_size),
        self.utm_y_min + (self.height * self.cell_size),
    )  # (x0, y0, x1, y1)

  @functools.cached_property
  def bbox_latlon(
      self,
  ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray,
             float | np.ndarray]:
    """Returns the bounding box in lat/lon coordinates."""
    south, west = utm_lib.to_latlon(self.utm_x_min, self.utm_y_min,
                                    int(self.utm_zone[:-1]), self.utm_zone[-1],
                                    strict=False)
    north, east = utm_lib.to_latlon(
        self.utm_x_min + self.width * self.cell_size,
        self.utm_y_min + self.height * self.cell_size,
        int(self.utm_zone[:-1]), self.utm_zone[-1],
        strict=False)
    return south, west, north, east  # (y0, x0, y1, x1)

  @functools.cached_property
  def corners_latlon(self) -> np.ndarray:
    """Computes the lat/lon for the four corners of the UTM grid."""
    zone_number = int(self.utm_zone[:-1])
    zone_letter = self.utm_zone[-1]
    x_max = self.utm_x_min + self.width * self.cell_size
    y_max = self.utm_y_min + self.height * self.cell_size
    bottom_left = utm_lib.to_latlon(
        self.utm_x_min, self.utm_y_min, zone_number, zone_letter, strict=False)
    bottom_right = utm_lib.to_latlon(
        x_max, self.utm_y_min, zone_number, zone_letter, strict=False)
    top_right = utm_lib.to_latlon(
        x_max, y_max, zone_number, zone_letter, strict=False)
    top_left = utm_lib.to_latlon(
        self.utm_x_min, y_max, zone_number, zone_letter, strict=False)

    return np.array([bottom_left, bottom_right, top_right, top_left])

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


def get_geotransform_info(
    lat: float, lon: float, img_width_m: float, resolution: float
) -> dict[str, Any]:
  """Returns geotransform info for a given location and image width in meters.

  Note that the geotransform is not necessarily aligned with the UTM resolution
  grid.

  Args:
    lat: Latitude of the center of the region to fetch.
    lon: Longitude of the center of the region to fetch.
    img_width_m: Width of the image in meters.
    resolution: Resolution of the image in meters per pixel.

  Returns:
    A dictionary with the geotransform and EPSG code for the UTM projection.
  """

  # Convert from lat/lon projection to UTM projection.
  lat_lon_crs = pyproj.CRS("EPSG:4326")
  epsg_code = (32600 if lat >= 0 else 32700) + (math.floor((lon + 180) / 6) + 1)
  utm_crs = pyproj.CRS(f"EPSG:{epsg_code}")
  utm_x, utm_y = pyproj.Transformer.from_crs(
      lat_lon_crs, utm_crs, always_xy=True
  ).transform(lon, lat)

  # Calculate geotransform and EPSG code for the UTM projection.
  top_left_x = utm_x - (img_width_m / 2)
  top_left_y = utm_y + (img_width_m / 2)
  geotransform = (top_left_x, resolution, 0, top_left_y, 0, -resolution)
  epsg = f"EPSG:{epsg_code}"

  return dict(geotransform=geotransform, epsg=epsg)
