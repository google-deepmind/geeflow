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

"""EarthEngine sampling and processing algorithms for GeeFlow pipelines.

This module contains the core algorithms that define how EE data sources are
sampled, reduced, and transformed into images for export. Each algorithm
follows the naming convention ``{source_type}_{algorithm}``, where
``source_type`` indicates the expected EE input type:

- ``ic_*``: Expects an ``ee.ImageCollection`` input.
- ``im_*``: Expects an ``ee.Image`` input.
- ``fc_*``: Expects an ``ee.FeatureCollection`` input.
- ``ccdc_*``: Expects a CCDC customized ImageCollection input.

Canonical algorithm names (new):

- ``ic_stack``: Stack individual IC images across time (multi-temporal).
- ``ic_reduced``: Reduce an IC to a single composite image.
- ``ic_temporal_composites``: Create one composite per date range window.
- ``im_static``: Use a static image (e.g. DEM) clipped to the ROI.
- ``im_from_item``: Load a specific image by asset ID from metadata.
- ``fc_properties``: Extract FeatureCollection properties as metadata.
- ``fc_rasterize``: Rasterize a FeatureCollection to an image.
- ``ccdc_coefficients``: Extract CCDC change detection coefficients.

The old names (``ic_sample``, ``ic_sample_reduced``, ``ic_sample_date_ragnges``,
``sample_roi``, ``get_image_from_item``, ``fc_get``, ``fc_to_image``,
``get_ccdc``) are kept as aliases for backwards compatibility but are
deprecated.
"""

import collections
from collections.abc import Callable, Sequence
import datetime
from typing import Any

from geeflow import ccdc_utils
from geeflow import times
import ml_collections
import numpy as np

import ee

_SYSTEM_PREFIX: str = "system:"
_SYSTEM_PREFIX_SUBSTITUTE: str = "notsystem:"
_SYSTEM_PROPERTIES = "system:time_start", "system:index"
_TIMESTAMP_PROP = "system:time_start"
_ASSET_ID_PROP = "system:index"

FEATURE_EXISTS_INTEGER_KEY = "GEEFLOW_INTERNAL_EXISTS"


def sample_roi(*_, **__):  # pylint: disable=unused-argument
  """Placeholder for backwards compatibility."""


def get_image_from_item(
    item: dict[str, Any], asset_id_key: str, select: Sequence[str]
) -> ee.Image:
  """Returns an ee.Image based on the specified asset ID in the config file.

  This sampling algorithm gets an ee.Image using the key specified to retrieve
  the Earth Engine asset ID from the config/item metadata.

  Args:
    item: The config item dictionary.
    asset_id_key: Key under which the asset ID is stored in item.
    select: Optional sequence of bands to select.
  """
  im = ee.Image(item[asset_id_key])
  if select:
    return im.select(select)
  return im


def preprocess_ic(
    roi: ee.Geometry,
    ic: ee.ImageCollection,
    cloud_mask_fn: Callable[[ee.Image], ee.Image] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    bands: list[str] | None = None,
    filter_bounds: bool = True,
    roi_filter_fn: Callable[..., ee.ImageCollection] | None = None,
    dummy_im: ee.Image | None = None,
    date_range: tuple[str, str] | None = None,
    limit: int | None = None,
    match_season_date: str | int | float | None = None,
    match_season_window_days: int = 45,
    temporal_anchor_ts: int | None = None,
    time_window_days: int | None = None,
) -> ee.ImageCollection:
  """Preprocesses the image collection.

  Args:
    roi: The region of interest.
    ic: The input ImageCollection.
    cloud_mask_fn: An optional function to apply a cloud mask to each image.
    sort_by: The property to sort the ImageCollection by.
    ascending: Whether to sort in ascending order.
    bands: A list of band names to select.
    filter_bounds: Whether to filter the ImageCollection by the bounds of the
      ROI.
    roi_filter_fn: An optional function to filter the ImageCollection based on
      the ROI.
    dummy_im: An optional dummy image to use if the ImageCollection is empty
      after filtering.
    date_range: A tuple of (start_date, end_date) in 'YYYY-MM-DD' format to
      filter the ImageCollection.
    limit: The maximum number of images to keep in the ImageCollection.
    match_season_date: A date string, timestamp, or ee.Date to filter images
      based on their day of year (DOY). Images are kept if their DOY is within
      `match_season_window_days` of the `match_season_date`'s DOY.
    match_season_window_days: The number of days around `match_season_date`'s
      DOY to include in the filter. Defaults to 45.
    temporal_anchor_ts: A timestamp in milliseconds. Used with
      `time_window_days` to filter images within a temporal window.
    time_window_days: The number of days before and after `temporal_anchor_ts`
      to include in the filter. Images are kept if their timestamp is within
      [`temporal_anchor_ts` - `time_window_days`, `temporal_anchor_ts` +
      `time_window_days`].

  Returns:
    The preprocessed ImageCollection.
  """
  if filter_bounds:
    ic = ic.filterBounds(roi)  # Can lead to error if all images filtered out.
    # For NICFI (& other fixed-sized ICs), consider to set filter_bounds=False.
  if roi_filter_fn:
    ic = roi_filter_fn(ic, roi=roi)
  if sort_by:
    ic = ic.sort(sort_by, ascending)
  if cloud_mask_fn:
    ic = ic.map(lambda x: x.updateMask(cloud_mask_fn(x)))
  if bands:
    ic = ic.select(bands)
  if date_range:
    ic = ic.filterDate(*date_range)
  if temporal_anchor_ts is not None:
    if time_window_days is None:
      raise ValueError(
          "time_window_days must be set if temporal_anchor_ts is set."
      )
    time_window = datetime.timedelta(days=time_window_days)
    time_window_ms = time_window.total_seconds() * 1000
    ic = ic.filter(
        ee.Filter.gte(
            "system:time_start",
            ee.Number(temporal_anchor_ts).subtract(time_window_ms),
        )
    )
    ic = ic.filter(
        ee.Filter.lte(
            "system:time_start",
            ee.Number(temporal_anchor_ts).add(time_window_ms),
        )
    )
  if match_season_date:
    anchor_date = ee.Date(match_season_date)
    # ee.Date.getRelative returns 0-indexed day of year (0-364).
    # We add 1 to convert it to the standard 1-indexed DOY (1-365).
    anchor_doy = anchor_date.getRelative("day", "year").add(1)

    def compute_season_distance(image):
      """Computes a season distance property and adds it to the image."""
      # Use standard 1-indexed DOY.
      image_doy = (
          ee.Date(image.get("system:time_start"))
          .getRelative("day", "year")
          .add(1)
      )
      diff = anchor_doy.subtract(image_doy).abs()
      doy_distance = ee.Number(
          ee.Algorithms.If(diff.gt(182), ee.Number(365).subtract(diff), diff)
      )
      return image.set("season_distance", doy_distance)

    ic = ic.map(compute_season_distance).filter(
        ee.Filter.lte("season_distance", match_season_window_days)
    )
  if limit:
    ic = ic.limit(limit)
  if dummy_im:
    ic = ee.ImageCollection(
        ee.Algorithms.If(ic.size().neq(0), ic, ee.ImageCollection(dummy_im))
    )
  return ic


def fetch_image_collection_properties(
    collection: ee.ImageCollection | ee.FeatureCollection,
    additional_properties: Sequence[str] = (),
    page_size: int = 1000,
    drop_unknown_timestamps: bool = True,
):
  """Fetches properties of all the images, sorted by timestamp if available.

  Args:
    collection: The input image collection.
    additional_properties: Additional property names to fetch.
    page_size: The maximum number of results per page. The server may return
      fewer images than requested. If unspecified, the page size default is 1000
      results per page.
    drop_unknown_timestamps: Whether to drop images with unknown timestamps.

  Returns:
    List of dicts containing the retrieved properties for each image in the
    collection sorted by timestamp if available.
  """
  properties = (*_SYSTEM_PROPERTIES, *additional_properties)

  def remap_features(f: ee.Feature):
    # Drop the geometry and any features we don't want.
    f = ee.Feature(None, {}).copyProperties(f, properties)  # pyrefly: ignore[bad-assignment]
    for property_name in properties:
      if property_name.startswith(_SYSTEM_PREFIX):
        remapped_name = _SYSTEM_PREFIX_SUBSTITUTE + property_name.removeprefix(
            _SYSTEM_PREFIX
        )
        f = f.set(remapped_name, f.get(property_name))  # pyrefly: ignore[bad-assignment]
    return f

  feature_col = ee.FeatureCollection(collection).map(remap_features)
  params = {"expression": feature_col, "pageSize": page_size}
  features = []
  while True:
    response = ee.data.computeFeatures(params)
    features.extend(response.get("features", []))
    if "nextPageToken" in response:
      params["pageToken"] = response["nextPageToken"]
    else:
      break

  for f in features:
    for property_name in properties:
      if property_name.startswith(_SYSTEM_PREFIX):
        remapped_name = _SYSTEM_PREFIX_SUBSTITUTE + property_name.removeprefix(
            _SYSTEM_PREFIX
        )
        if remapped_name in f["properties"]:
          f["properties"][property_name] = f["properties"].pop(remapped_name)
      f["properties"].setdefault(property_name, "unknown")
  features = [f["properties"] for f in features]
  if drop_unknown_timestamps:
    features = [f for f in features if f[_TIMESTAMP_PROP] != "unknown"]
  # Sort by timestamp if available.
  return sorted(features, key=lambda x: x.get(_TIMESTAMP_PROP, 0))


def fc_get(
    fc: ee.FeatureCollection, roi: ee.Geometry, properties: Sequence[str]
) -> dict[str, Any | set[Any]]:
  fc = fc.filterBounds(roi)
  return fetch_image_collection_properties(fc, properties)[0]
  # What if there are multiple properties? For example, a plot that belong to
  # multiple countries. Could we group them as a set?
  # return {k: set(dic[k] for dic in fc_properties) for k in fc_properties[0]}


def fc_to_image(
    fc: ee.FeatureCollection,
    roi: ee.Geometry,
    props: list[str],
    class_names: dict[str, list[str]] | None = None,
    reducer: str = "max",
    reduce_resolution_reducer: str | None = None,
    drop_missing_classes: bool = True,
    missing_class_value: int = -1,
    unmask_value: int | None = None,
    dtype: str | None = None,
    temporal_split: tuple[str, list[Any]] | None = None,
) -> ee.Image:
  """Rasterizes a FeatureCollection to an image.

  Args:
    fc: FeatureCollection.
    roi: ROI UTM bounds geometry with an HxW grid.
    props: Selected properties of fc.
    class_names: A dict of property names mapped to list of classes, which are
      to be mapped to ints. If only a single list is given, it is matched to
      props[0] property.
    reducer: Name of the ee.reducer.
    reduce_resolution_reducer: Name of a reducer to use for resuceResolution op.
    drop_missing_classes: Features are dropped if it's value for given property
      is not in class_names (be careful with multiple props).
    missing_class_value: Value to use for missing classes if
      `drop_missing_classes` is False.
    unmask_value: If provided, will unmask the image with this value.
    dtype: If provided, will cast the image to this dtype.
    temporal_split: If provided (a tuple of (property_name, list_of_values)),
      will split the FC based on the property values and create a time series
      folded into channels.

  Returns:
    An np.ndarray with dimensions (N, H, W, C) or (H, W, C) and its mask.
  """
  # If dtype is not provided, but FEATURE_EXISTS_INTEGER_KEY is in props, cast
  # the image to byte as it's a binary output.
  if (
      dtype is None
      and FEATURE_EXISTS_INTEGER_KEY in props
      and len(props) == 1
  ):
    dtype = "uint8"

  def to_scalar(feature, props, dic):
    return feature.set(
        props, ee.Dictionary(dic).get(feature.get(props), missing_class_value)
    )

  fc = fc.filterBounds(roi)
  fc = fc.map(lambda f: f.setGeometry(f.geometry().intersection(roi, 1e-3)))

  if FEATURE_EXISTS_INTEGER_KEY in props:
    fc = fc.map(lambda f: f.set(FEATURE_EXISTS_INTEGER_KEY, 1))
  if class_names is not None:
    if not isinstance(class_names, (dict, ml_collections.ConfigDict)):
      class_names = {props[0]: class_names}
    for key, classes in class_names.items():
      # Convert string class into integer.
      dic = {class_name: i for i, class_name in enumerate(classes)}
      if drop_missing_classes:
        fc = fc.filter(ee.Filter.inList(key, ee.List(list(classes))))
      fc = fc.map(lambda f: to_scalar(feature=f, props=key, dic=dic))  # pylint: disable=cell-var-from-loop
  if temporal_split is not None:
    new_props = []
    im = None
    for val in temporal_split[1]:
      tmp_im = fc.filter(ee.Filter.eq(temporal_split[0], val)).reduceToImage(
          properties=props, reducer=get_fc_reduce_fn(reducer).forEach(props)
      )
      tmp_props = [f"{p}_{val}" for p in props]
      new_props.extend(tmp_props)
      tmp_im = tmp_im.rename(tmp_props)
      if im is None:
        im = tmp_im
      else:
        im = im.addBands(tmp_im)
    assert im is not None
    props = new_props
  else:
    im = fc.reduceToImage(
        properties=props, reducer=get_fc_reduce_fn(reducer).forEach(props)
    )
  if unmask_value is not None:
    im = im.unmask(unmask_value, False)

  if reduce_resolution_reducer:
    im = im.reproject(roi.projection())
    im = im.reduceResolution(
        reducer=get_fc_reduce_fn(reduce_resolution_reducer), maxPixels=1024)

  if dtype == "uint8":
    im = im.toByte()
  elif dtype is not None:
    # All properties are converted to the same dtype.
    im = im.cast(dict.fromkeys(props, dtype))

  return im


def get_ccdc(
    ic: ee.ImageCollection,
    roi: ee.Geometry,
    *,
    bands: list[str],
    num_segments: int,
) -> ee.Image:
  """Extracts CCDC change detection coefficients.

  Args:
    ic: CCDC's ImageCollection.
    roi: ROI UTM bounds geometry with an HxW grid. Alternatively, it can be an
      ee.Point which will results in (H, W) == (1, 1).
    bands: What bands to include in the output.
    num_segments: How many detected CCDC segments to return. If not enough, will
      be padded with zeros.

  Returns:
    An image with CCDC bands.
  """
  ccdc = ic.filterBounds(roi)
  ccdc = ee.Image(
      ee.Algorithms.If(ccdc.size().eq(0), ic.first(), ccdc.mosaic())
  )

  bands_1d = [band for band in bands if not band.endswith("_coefs")]
  im = ccdc_utils.im_add_ccdc_bands_1d(ccdc.select(bands_1d), num_segments)
  bands_2d = [band for band in bands if band.endswith("_coefs")]
  if bands_2d:
    im = ccdc_utils.im_add_ccdc_bands_2d(
        ccdc.select(bands_2d), num_segments, im
    )
  assert im is not None
  return im


def get_dummy_image(im: ee.Image) -> ee.Image:
  """Returns an unrestricted image with all bands masked out."""
  # .updateMask(0) makes the whole image invalid
  # .unmask(0, False) will fill the whole image with 0, but will make it valid
  # .updateMask(0) makes the whole image invalid again
  return im.updateMask(0).unmask(0, False).updateMask(0)


def get_fc_reduce_fn(name: str = "first") -> ee.Reducer:
  """Returns FeatureCollection reduce function."""
  if name == "first":
    return ee.Reducer.first()
  if name == "firstNonNull":
    return ee.Reducer.firstNonNull()
  if name == "mode":
    return ee.Reducer.mode()
  elif name == "max":
    return ee.Reducer.max()
  elif name == "mean":
    return ee.Reducer.mean()
  elif name == "sum":
    return ee.Reducer.sum()
  else:
    raise ValueError(f"Reducer `{name}` not supported yet.")


def get_ic_sample_reducer(name: str = "mosaic", **kwargs) -> ee.Reducer:
  """Returns an `ee.Reducer` for reducing sampled ImageCollection features."""
  if name == "mosaic":
    name = "lastNonNull"
  elif name == "percentile":
    kwargs.setdefault("percentiles", [10, 25, 50, 75, 90])
  return getattr(ee.Reducer, name)(**kwargs)


def get_ic_reduce_fn(
    name: str,
    scale: int | None = None,
    roi: ee.Geometry | None = None,
    **kwargs,
) -> Callable[[ee.ImageCollection], ee.Image]:
  """Returns ImageCollection reduce function."""
  if name == "mosaic":
    return lambda x: x.mosaic()
  elif name == "qualityMosaic":
    return lambda x: x.qualityMosaic(**kwargs)
  elif name == "mean":
    # Convert explicitly to float since mean will not always preserve the type.
    return lambda x: x.map(lambda y: y.toFloat()).mean()
  elif name == "median":
    return lambda x: x.median()
  elif name == "max":
    return lambda x: x.max()
  elif name == "min":
    return lambda x: x.min()
  elif name == "mode":
    return lambda x: x.mode()
  elif name == "first":
    return lambda x: x.first()
  elif name.startswith("reduceResolutionTo"):
    if name == "reduceResolutionToMean":
      reducer = ee.Reducer.mean()
    elif name == "reduceResolutionToMeanAndStd":
      reducer = ee.Reducer.mean().combine(
          ee.Reducer.stdDev(), sharedInputs=True
      )
    elif name == "reduceResolutionToMeanAndStdAndMax":
      reducer = (
          ee.Reducer.mean()
          .combine(ee.Reducer.stdDev(), sharedInputs=True)
          .combine(ee.Reducer.max(), sharedInputs=True)
      )
    elif name == "reduceResolutionToMax":
      reducer = ee.Reducer.max()
    else:
      raise ValueError(f"Reducer `{name}` not supported yet.")
    assert roi is not None

    def rr(im):
      if "clip_max_value_before_reduce" in kwargs:
        im = im.min(kwargs["clip_max_value_before_reduce"])
      # When we fetch a huge plot (say 6400m x 6400m) and a source is at very
      # high resolution (say 0.5m) and we use reduceResolutionTo* reducer,
      # EE would fail with "Reprojection output too large (12800x12800 pixels)".
      # In order to avoid this, we first preaggreate at a lower resolution with
      # the same projection (which apparently does not trigger the check) and
      # then reproject to the final scale.
      if "pre_reduce_resolution" in kwargs:
        im = im.reproject(
            im.projection().atScale(kwargs["pre_reduce_resolution"])
        ).reduceResolution(reducer=reducer, maxPixels=4096)
      return im.reproject(
          roi.projection().atScale(im.projection().nominalScale())
      ).reduceResolution(reducer=reducer, maxPixels=4096)

    def reduce_resolution(im):
      if isinstance(im, ee.ImageCollection):
        im = im.map(rr)
        im = get_ic_reduce_fn(kwargs.get("ic_reduce_fn", "mosaic"))(im)
      else:
        im = rr(im)
      return im

    return reduce_resolution
  elif name == "percentile":
    kwargs.setdefault("percentiles", [10, 25, 50, 75, 90])
    return lambda x: x.reduce(ee.Reducer.percentile(**kwargs))
  elif name.startswith("with_most_valid_pixels_in_band_0"):

    def set_valid_pixel_count(image):
      num = (
          image.select(0)
          .reduceRegion(
              ee.Reducer.count(),
              maxPixels=1e9,
              scale=scale,
              geometry=roi,
              **kwargs,
          )
          .get(image.bandNames().get(0))
      )
      return image.set("numValidPixels", num)

    def mask(ic):
      ic = ic.map(set_valid_pixel_count)
      ic = ic.sort("numValidPixels", name.endswith("_mosaic"))
      if name.endswith("_mosaic"):
        return ic.mosaic()
      return ic.first()

    # Long story short: when using dummy_im we can get an image with a footprint
    # that is very far away from the ROI. And if we are using UTM projection
    # reduceRegion could cause very weird behaviors. As a "solution" we handle
    # a case with an IC having 1 image separately to avoid this case.
    # A proper solution would be to avoid using dummy_im completely, but this
    # would require much more substantial changes, since we'll have to be
    # computing the shapes manually when writing TFDS dataset.
    # TODO: drop usage of dummy_im
    def _fix_dummy_im(ic):
      return ee.Image(ee.Algorithms.If(ic.size().eq(1), ic.first(), mask(ic)))

    return _fix_dummy_im
  raise ValueError(f"Unrecognized reducer name `{name}`")


def ic_sample(
    roi: ee.Geometry,
    ic: ee.ImageCollection,
    *,
    cloud_mask_fn: Callable[[ee.Image], ee.Image] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    bands: list[str] | None = None,
    filter_bounds: bool = True,
    roi_filter_fn: Callable[..., ee.ImageCollection] | None = None,
    limit: int | None = None,
    additional_properties: Sequence[str] = (),
    date_range: tuple[str, str] | None = None,
    match_season_date: str | int | float | None = None,
    match_season_window_days: int = 45,
    temporal_anchor_ts: Any = None,
    time_window_days: int | None = None,
) -> tuple[Sequence[ee.Image], dict[str, np.ndarray]]:
  """Stacks individual IC images across time (multi-temporal).

  Returns one image per item in the ImageCollection (after filtering), producing
  a multi-temporal stack. Each image is individually accessible with its own
  timestamp metadata.

  Args:
    roi: Region of interest geometry.
    ic: Input ImageCollection to sample from.
    cloud_mask_fn: Optional cloud masking function applied to each image.
    sort_by: Optional property name to sort the collection by.
    ascending: Sort order (default: ascending).
    bands: Optional band selection.
    filter_bounds: Whether to filter the IC by ROI bounds.
    roi_filter_fn: Optional custom ROI filtering function.
    limit: Maximum number of images to return (most recent).
    additional_properties: Extra properties to include in metadata.
    date_range: Optional (start, end) date filter.
    match_season_date: A date string, timestamp, or ee.Date to filter images
      based on their day of year (DOY). Images are kept if their DOY is within
      `match_season_window_days` of the `match_season_date`'s DOY.
    match_season_window_days: The number of days around `match_season_date`'s
      DOY to include in the filter. Defaults to 45.
    temporal_anchor_ts: A timestamp in milliseconds. Used with
      `time_window_days` to filter images within a temporal window.
    time_window_days: The number of days before and after `temporal_anchor_ts`
      to include in the filter. Images are kept if their timestamp is within
      [`temporal_anchor_ts` - `time_window_days`, `temporal_anchor_ts` +
      `time_window_days`].

  Returns:
    A sequence of ee.Image objects for the filtered and sorted images.
    A dictionary of metadata, including 'timestamps' and any
      `additional_properties`, as numpy arrays.
  """
  ic = preprocess_ic(
      roi,
      ic,
      cloud_mask_fn,
      sort_by,
      ascending,
      bands,
      filter_bounds,
      roi_filter_fn,
      date_range=date_range,
      match_season_date=match_season_date,
      match_season_window_days=match_season_window_days,
      temporal_anchor_ts=temporal_anchor_ts,
      time_window_days=time_window_days,
  )
  ims, metadata = [], collections.defaultdict(list)
  sorted_image_properties = fetch_image_collection_properties(
      ic, additional_properties
  )
  if limit:
    # We apply the limit by only keeping the `limit` most recent images.
    sorted_image_properties = sorted_image_properties[-limit:]
  for prop in sorted_image_properties:
    im_t = ic.filter(ee.Filter.eq(_ASSET_ID_PROP, prop[_ASSET_ID_PROP])).first()
    ims.append(im_t)
    metadata["timestamps"].append(prop[_TIMESTAMP_PROP])
    for p in additional_properties:
      metadata[p].append(prop[p])
  if ims:
    return ims, {k: np.asarray(v) for k, v in metadata.items()}
  return [], {k: np.array([]) for k in additional_properties}


def ic_sample_reduced(
    roi: ee.Geometry,
    ic: ee.ImageCollection,
    *,
    scale: int,
    cloud_mask_fn: Callable[[ee.Image], ee.Image] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    bands: list[str] | None = None,
    filter_bounds: bool = True,
    roi_filter_fn: Callable[..., ee.ImageCollection] | None = None,
    dummy_im: ee.Image | None = None,
    reduce_fn: str = "mosaic",
    date_range: tuple[str, str] | None = None,
    **reduce_fn_kwargs,
) -> ee.Image:
  """Reduces an ImageCollection to a single composite image.

  Applies a reducer (e.g. mosaic, mean, median) over all images in the
  collection to produce a single composite.

  Args:
    roi: Region of interest geometry.
    ic: Input ImageCollection.
    scale: Spatial resolution in meters.
    cloud_mask_fn: Optional cloud masking function.
    sort_by: Optional sort property.
    ascending: Sort order.
    bands: Optional band selection.
    filter_bounds: Whether to filter by ROI bounds.
    roi_filter_fn: Optional custom ROI filter.
    dummy_im: Fallback image if IC is empty.
    reduce_fn: Reducer name (e.g. 'mosaic', 'mean', 'median').
    date_range: Optional (start, end) date filter.
    **reduce_fn_kwargs: Additional kwargs for the reducer.

  Returns:
    A single composite ee.Image.
  """
  ic = preprocess_ic(
      roi,
      ic,
      cloud_mask_fn,
      sort_by,
      ascending,
      bands,
      filter_bounds,
      roi_filter_fn,
      dummy_im,
      date_range=date_range,
  )
  reduce_fn = get_ic_reduce_fn(  # pyrefly: ignore[bad-assignment]
      reduce_fn, scale=scale, roi=roi, **reduce_fn_kwargs
  )
  return reduce_fn(ic)  # pyrefly: ignore[not-callable]


def ic_sample_date_ranges(
    roi: ee.Geometry,
    ic: ee.ImageCollection,
    *,
    scale: int,
    cloud_mask_fn: Callable[[ee.Image], ee.Image] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    bands: list[str] | None = None,
    filter_bounds: bool = True,
    roi_filter_fn: Callable[..., ee.ImageCollection] | None = None,
    date_ranges: Sequence[tuple[str, int, int]],
    dummy_im: ee.Image | None = None,
    reduce_fn: str = "mosaic",
    limit: int | None = None,
    **reduce_fn_kwargs,
) -> tuple[Sequence[ee.Image], dict[str, np.ndarray]]:
  """Creates one composite image per date range window.

  For each date range in ``date_ranges``, filters the IC to that time window
  and applies the reducer to produce a composite. Returns a list of composite
  images (one per date range) with their midpoint timestamps.

  Args:
    roi: Region of interest geometry.
    ic: Input ImageCollection.
    scale: Spatial resolution in meters.
    cloud_mask_fn: Optional cloud masking function.
    sort_by: Optional sort property.
    ascending: Sort order.
    bands: Optional band selection.
    filter_bounds: Whether to filter by ROI bounds.
    roi_filter_fn: Optional custom ROI filter.
    date_ranges: Sequence of (start_date, months, days) tuples defining the
      temporal windows.
    dummy_im: Fallback image if IC is empty for a given window.
    reduce_fn: Reducer name (e.g. 'mosaic', 'mean', 'median').
    limit: Maximum images per window.
    **reduce_fn_kwargs: Additional kwargs for the reducer.

  Returns:
    Tuple of (list of composite ee.Images, metadata dict with timestamps).
  """
  reduce_fn = get_ic_reduce_fn(  # pyrefly: ignore[bad-assignment]
      reduce_fn, scale=scale, roi=roi, **reduce_fn_kwargs
  )
  windows, timestamps = times.resolve_date_ranges(date_ranges)
  ims = [
      reduce_fn(  # pyrefly: ignore[not-callable]
          preprocess_ic(
              roi,
              ic,
              cloud_mask_fn,
              sort_by,
              ascending,
              bands,
              filter_bounds,
              roi_filter_fn,
              dummy_im,
              dates_range,
              limit=limit,
          )
      )
      for dates_range in windows
  ]
  return ims, dict(timestamps=timestamps)


def add_roi_validity(
    im: ee.Image, roi: ee.Geometry, band: str = "R", scale: int | None = None
) -> ee.Image:
  """Returns im with `validity` property set to percentage of non-0 values."""
  im_orig = im
  if scale:
    im = im_orig.reproject(crs=roi.projection().crs(), scale=scale)
  sample = im.select(band).mask().sampleRectangle(roi).get(band)
  arr = ee.Array(sample).toFloat()
  mean = arr.reduce(ee.Reducer.mean(), [0, 1])
  mean = mean.get([0, 0])  # Returns scalar instead of (1, 1) array.
  return im_orig.set({"validity": mean})  # pyrefly: ignore[bad-return]


def add_abs_time_difference(im: ee.Image, ref_date: ee.Date) -> ee.Image:
  """Returns im with `date_difference` property set to ms from ref_date."""
  abs_time_difference = (
      ee.Number(im.get("system:time_start")).subtract(ref_date.millis()).abs()
  )
  return im.set("abs_time_difference", abs_time_difference)  # pyrefly: ignore[bad-return]


# For backwards compatibility.
ic_stack = ic_sample
ic_reduced = ic_sample_reduced
ic_temporal_composites = ic_sample_date_ranges
im_from_item = get_image_from_item
im_static = sample_roi
fc_properties = fc_get
fc_rasterize = fc_to_image
ccdc_coefficients = get_ccdc
