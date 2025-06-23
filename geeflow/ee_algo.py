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

"""EarthEngine algorithms for data pipelines."""

import collections
from collections.abc import Callable, Sequence
import datetime
import functools
from typing import Any

from dateutil import relativedelta
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


def _preprocess_ic(
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
) -> ee.ImageCollection:
  """Preprocesses the image collection."""
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
  if limit:
    ic = ic.limit(limit)
  if dummy_im:
    ic = ee.ImageCollection(ee.Algorithms.If(ic.size().neq(0), ic,
                                             ee.ImageCollection(dummy_im)))
  return ic


def fetch_image_collection_properties(
    collection: ee.ImageCollection, additional_properties: Sequence[str] = ()
):
  """Fetches the specified properties of all the images sorted by timestamp.

  Args:
    collection: The input image collection.
    additional_properties: Additional property names to fetch.

  Returns:
    List of dicts containing the retrieved properties for each image in the
    collection.
  """
  properties = (*_SYSTEM_PROPERTIES, *additional_properties)

  def remap_features(f: ee.Feature):
    # Drop the geometry and any features we don't want.
    f = ee.Feature(None, {}).copyProperties(f, properties)
    for property_name in properties:
      if property_name.startswith(_SYSTEM_PREFIX):
        remapped_name = _SYSTEM_PREFIX_SUBSTITUTE + property_name.removeprefix(
            _SYSTEM_PREFIX
        )
        f = f.set(remapped_name, f.get(property_name))
    return f

  feature_col = ee.FeatureCollection(collection).map(remap_features)
  params = {"expression": feature_col}
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
  # Sort by timestamp if available.
  return sorted(features, key=lambda x: x.get(_TIMESTAMP_PROP, 0))


def fc_get(
    fc: ee.FeatureCollection, roi: ee.Geometry, properties: Sequence[str]
) -> dict[str, Any | set[Any]]:
  fc = fc.filterBounds(roi)
  fc_properties = fetch_image_collection_properties(fc, properties)
  return fc_properties[0]
  # What if there are multiple properties? For example, a plot that belong to
  # multiple countries. Could we group them as a set?
  # return {k: set(dic[k] for dic in fc_properties) for k in fc_properties[0]}


def fc_to_image(
    fc: ee.FeatureCollection,
    roi: ee.Geometry,
    props: list[str],
    class_names: dict[str, list[str]] | None = None,
    reducer: str = "max",
    drop_missing_classes: bool = True,
    missing_class_value: int = -1,
) -> ee.Image:
  """Returns FC data.

  Args:
    fc: FeatureCollection.
    roi: ROI UTM bounds geometry with an HxW grid.
    props: Selected properties of fc.
    class_names: A dict of property names mapped to list of classes, which are
      to be mapped to ints. If only a single list is given, it is matched to
      props[0] property.
    reducer: Name of the ee.reducer.
    drop_missing_classes: Features are dropped if it's value for given property
      is not in class_names (be careful with multiple props).
    missing_class_value: Value to use for missing classes if
      `drop_missing_classes` is False.

  Returns:
    An np.ndarray with dimensions (N, H, W, C) or (H, W, C) and its mask.
  """

  def to_scalar(feature, props, dic):
    return feature.set(props, ee.Dictionary(dic).get(
        feature.get(props), missing_class_value))

  fc = fc.filterBounds(roi)
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
  im = fc.reduceToImage(properties=props,
                        reducer=get_fc_reduce_fn(reducer).forEach(props))
  return im


def _add_ccdc_bands_2d(ic, num_segments, im=None):
  """Adds CCDC bands with shape (num_segments, 8)."""
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


def _add_ccdc_bands_1d(ic, num_segments, im=None):
  zeros = ee.Array([0]).repeat(0, num_segments)
  for j in range(num_segments):
    im_i = ic.unmask(zeros, False).arrayCat(zeros, 0).float().arrayGet(j)
    rename_fn = functools.partial(
        lambda n, idx: ee.String(n).cat(ee.String(idx)), idx=f"#{j}#0"
    )
    im_i = im_i.rename(im_i.bandNames().map(rename_fn))
    im = im_i if im is None else im.addBands(im_i)
  return im


def get_ccdc(
    ic: ee.ImageCollection,
    roi: ee.Geometry,
    *,
    bands: list[str],
    num_segments: int,
) -> ee.Image:
  """Returns CCDC data.

  Args:
    ic: CCDC's ImageCollection.
    roi: ROI UTM bounds geometry with an HxW grid. Alternatively, it can be an
      ee.Point which will results in (H, W) == (1, 1).
    bands: What bands to include in the output.
    num_segments: How many detected CCDC segments to return. If not enough, will
      be padded with zeros.

  Returns:
    A dict of band names to ee.Array().
  """
  ccdc = ic.filterBounds(roi)
  ccdc = ee.Image(
      ee.Algorithms.If(ccdc.size().eq(0), ic.first(), ccdc.mosaic())
  )

  bands_1d = [band for band in bands if not band.endswith("_coefs")]
  im = _add_ccdc_bands_1d(ccdc.select(bands_1d), num_segments)

  bands_2d = [band for band in bands if band.endswith("_coefs")]
  im = _add_ccdc_bands_2d(ccdc.select(bands_2d), num_segments, im)

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
  else:
    raise ValueError(f"Reducer `{name}` not supported yet.")


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
    if name == "reduceResolutionToMeanAndStd":
      reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(),
                                          sharedInputs=True)
    elif name == "reduceResolutionToMeanAndStdAndMax":
      reducer = ee.Reducer.mean().combine(
          ee.Reducer.stdDev(), sharedInputs=True).combine(
              ee.Reducer.max(), sharedInputs=True)
    elif name == "reduceResolutionToMax":
      reducer = ee.Reducer.max()
    else:
      raise ValueError(f"Reducer `{name}` not supported yet.")
    assert roi is not None
    def rr(im):
      return im.reproject(roi.projection().atScale(
          im.projection().nominalScale())).reduceResolution(
              reducer=reducer, maxPixels=4096)
    def reduce_resolution(im):
      if isinstance(im, ee.ImageCollection):
        im = im.map(rr)
        im = im.mosaic()
      else:
        im = rr(im)
      return im
    return reduce_resolution
  elif name == "percentile":
    kwargs.setdefault("percentiles", [10, 25, 50, 75, 90])
    return lambda x: x.reduce(ee.Reducer.percentile(**kwargs))
  elif name.startswith("with_most_valid_pixels_in_band_0"):
    def set_valid_pixel_count(image):
      num = image.select(0).reduceRegion(
          ee.Reducer.count(), maxPixels=1e9, scale=scale, geometry=roi,
          **kwargs).get(image.bandNames().get(0))
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
) -> tuple[Sequence[ee.Image], dict[str, np.ndarray]]:
  """Returns ROI samples for all filtered IC images."""
  ic = _preprocess_ic(roi, ic, cloud_mask_fn, sort_by, ascending, bands,
                      filter_bounds, roi_filter_fn, limit=limit,
                      date_range=date_range)
  ims, metadata = [], collections.defaultdict(list)
  for prop in fetch_image_collection_properties(ic, additional_properties):
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
    **reduce_fn_kwargs,
) -> ee.Image:
  """Returns ROI samples reduced over the whole IC."""
  ic = _preprocess_ic(roi, ic, cloud_mask_fn, sort_by, ascending, bands,
                      filter_bounds, roi_filter_fn, dummy_im)
  reduce_fn = get_ic_reduce_fn(reduce_fn, scale=scale, roi=roi,
                               **reduce_fn_kwargs)
  return reduce_fn(ic)


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
    date_ranges: tuple[str, int, int],
    dummy_im: ee.Image | None = None,
    reduce_fn: str = "mosaic",
    limit: int | None = None,
    **reduce_fn_kwargs,
) -> tuple[Sequence[ee.Image], dict[str, np.ndarray]]:
  """Returns ROI samples reduced over given date ranges."""
  reduce_fn = get_ic_reduce_fn(
      reduce_fn, scale=scale, roi=roi, **reduce_fn_kwargs
  )
  ims, ts = [], []
  for start, months, days in date_ranges:
    start = datetime.datetime.strptime(start, r"%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc  # Default EE timezone.
    )
    end = start + relativedelta.relativedelta(days=days, months=months)
    ts.append(int(start.timestamp() + end.timestamp()) // 2 * 1000)

    dates_range = start.strftime(r"%Y-%m-%d"), end.strftime(r"%Y-%m-%d")
    ic_t = _preprocess_ic(roi, ic, cloud_mask_fn, sort_by, ascending, bands,
                          filter_bounds, roi_filter_fn, dummy_im, dates_range,
                          limit=limit)
    ims.append(reduce_fn(ic_t))
  return ims, dict(timestamps=np.array(ts))


def add_roi_validity(im: ee.Image, roi: ee.Geometry, band: str = "R",
                     scale: int | None = None) -> ee.Image:
  """Returns im with `validity` property set to percentage of non-0 values."""
  im_orig = im
  if scale:
    im = im_orig.reproject(crs=roi.projection().crs(), scale=scale)
  sample = im.select(band).mask().sampleRectangle(roi).get(band)
  arr = ee.Array(sample).toFloat()
  mean = arr.reduce(ee.Reducer.mean(), [0, 1])
  mean = mean.get([0, 0])  # Returns scalar instead of (1, 1) array.
  return im_orig.set({"validity": mean})


def add_abs_time_difference(im: ee.Image, ref_date: ee.Date) -> ee.Image:
  """Returns im with `date_difference` property set to ms from ref_date."""
  abs_time_difference = (
      ee.Number(im.get("system:time_start")).subtract(ref_date.millis()).abs())
  return im.set("abs_time_difference", abs_time_difference)
