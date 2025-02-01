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

"""EarthEngine algorithms for data pipelines."""

from typing import Callable

from geeflow import times
import ml_collections

import ee


FEATURE_EXISTS_INTEGER_KEY = "GEEFLOW_INTERNAL_EXISTS"


def fc_get(fc: ee.FeatureCollection, roi: ee.Geometry, prop: str) -> ee.Array:
  fc = fc.filterBounds(roi)
  return ee.Algorithms.If(fc.size().neq(0), fc.first().get(prop), "unknown")


def fc_to_image(
    fc: ee.FeatureCollection,
    roi: ee.Geometry,
    props: list[str],
    scale: int,
    class_names: dict[str, list[str]] | None = None,
    reducer: str = "max",
    drop_missing_classes: bool = True,
    missing_class_value: int = -1,
) -> tuple[ee.Array, ee.Array]:
  """Returns FC data.

  Args:
    fc: FeatureCollection.
    roi: ROI UTM bounds geometry with an HxW grid.
    props: Selected properties of fc.
    scale: Output pixel spacing in meters.
    class_names: A dict of property names mapped to list of classes, which are
      to be mapped to ints. If only a single list is given, it is matched to
      props[0] property.
    reducer: Name of the ee.reducer.
    drop_missing_classes: Features are dropped if it's value for given property
      is not in class_names (be careful with multiple props).
    missing_class_value: Value to use for missing classes if
      `drop_missing_classes` is False.

  Returns:
    An ee.Array() with dimensions (N, H, W, C) or (H, W, C) and its mask.
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
  im = im.reproject(roi.projection().crs(), scale=scale)
  return sample_roi(im, roi, scale=scale)


def get_ccdc(
    ic: ee.ImageCollection,
    roi: ee.Geometry,
    *,
    scale: int,
    bands: list[str],
    num_segments: int,
) -> dict[str, ee.Array]:
  """Returns CCDC data.

  Args:
    ic: CCDC's ImageCollection.
    roi: ROI UTM bounds geometry with an HxW grid. Alternatively, it can be an
      ee.Point which will results in (H, W) == (1, 1).
    scale: Output pixel spacing in meters.
    bands: What bands to include in the output.
    num_segments: How many detected CCDC segments to return. If not enough, will
      be padded with zeros.
  Returns:
    A dict of band names to ee.Array().
  """
  crs = roi.projection().crs()
  ccdc = ic.filterBounds(roi)
  ccdc = ee.Algorithms.If(ccdc.size().eq(0), ic.first(), ccdc.mosaic())
  ccdc = ee.Image(ccdc)
  zeros_1d = ee.Array([0]).repeat(0, num_segments)
  zeros_2d = ee.Array([0]).repeat(0, num_segments).repeat(1, 8)
  bands_1d = [band for band in bands if not band.endswith("_coefs")]
  bands_2d = [band for band in bands if band.endswith("_coefs")]
  ccdc_1d = ccdc.select(bands_1d).unmask(zeros_1d, False).arrayCat(
      zeros_1d, 0).float().arraySlice(0, 0, num_segments)
  ccdc_2d = ccdc.select(bands_2d).unmask(zeros_2d, False).arrayCat(
      zeros_2d, 0).float().arraySlice(0, 0, num_segments)
  ccdc_1d = ee.Array(ccdc_1d.toArray().reproject(crs=crs, scale=scale)
                     .sampleRectangle(roi).get("array"))
  ccdc_2d = ee.Array(ccdc_2d.toArray().reproject(crs=crs, scale=scale)
                     .sampleRectangle(roi).get("array"))
  res = {}
  for i, band in enumerate(bands_1d):
    res[band] = ccdc_1d.slice(axis=2,
                              start=i*num_segments, end=(i+1)*num_segments)
  for i, band in enumerate(bands_2d):
    res[band] = ccdc_2d.slice(axis=2,
                              start=i*num_segments, end=(i+1)*num_segments)
  return res


def sample_roi(
    ic: ee.ImageCollection | ee.Image,
    roi: ee.Geometry,
    *,
    scale: int,
    crs: str | None = None,
    mask_value: int = 0,
) -> tuple[ee.Array, ee.Array]:
  """Samples one ROI in Image or ImageCollection and returns as Array.

  Args:
    ic: ImageCollection of N images (or a single Image) with C bands each.
    roi: ROI UTM bounds geometry with an HxW grid. Alternatively, it can be an
      ee.Point which will results in (H, W) == (1, 1).
    scale: Output pixel spacing in meters.
    crs: CRS of ROI (usually UTM). If not given, extracted from ROI.
    mask_value: Value to insert into invalid masked out pixels. Defaults to `0`,
      which is fine for most cases. For categorical data, one might want to set
      it eg. to `-1`.
  Returns:
    An ee.Array() with dimensions (N, H, W, C) or (H, W, C).
    An ee.Array() of the mask with the same dimensions.
  """
  crs = crs or roi.projection().crs()
  # Change masked out values with 0 (unmask(0, False)) and add the mask.
  if isinstance(ic, ee.ImageCollection):
    ic = ic.map(lambda x: x.unmask(mask_value, False).addBands(x.mask()))
  else:
    ic = ic.unmask(mask_value, False).addBands(ic.mask())
  arr = (
      ic.toArray()
      .reproject(crs=crs, scale=scale)
      .sampleRectangle(roi)
      .get("array")
  )
  arr = ee.Array(arr)
  if isinstance(ic, ee.ImageCollection):
    arr = arr.transpose(1, 2).transpose(0, 1)
  # Split in half on the last axis to separate the mask.
  axis = arr.length().length().subtract(1).get([0]).toInt()
  num_bands = arr.length().get([-1]).divide(2)
  return arr.slice(axis, 0, num_bands), arr.slice(axis, num_bands)


def ic_timestamps(
    ic: ee.ImageCollection, name: str = "system:time_start"
) -> ee.Array:
  """Returns a 1D array of timestamps (as millis)."""
  t = ee.FeatureCollection(ic).aggregate_array(name)
  t = ee.Array(t)
  return t


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
    scale: int,
    filter_bounds: bool = True,
    limit: int | None = None,
) -> tuple[ee.Array, ee.Array, ee.Array]:
  """Returns ROI samples for all filtered IC images."""
  if filter_bounds:  # Can lead to error if all images filtered out.
    # For NICFI (& other fixed-sized ICs), consider to set filter_bounds=False.
    ic = ic.filterBounds(roi)
    # To avoid errors if empty, provide default_array_value to sample_roi(), or
    # ic = ee.Algorithms.If(ic.size().neq(0), ic, ic_orig)
  if limit:
    ic = ic.limit(limit)
  samples, samples_mask = sample_roi(ic, roi, scale=scale)
  timestamps = ic_timestamps(ic)
  return samples, samples_mask, timestamps


def rgb_ic_sample(
    roi: ee.Geometry,
    ic: ee.ImageCollection,
    *,
    scale: int,
    filter_bounds: bool = True,
    roi_filter_fn: Callable[..., ee.ImageCollection] | None = None,
    limit: int | None = None,
) -> tuple[ee.Array, ee.Array, ee.Array]:
  """Returns ROI samples for all filtered RGB/Magrathean IC images."""
  if filter_bounds:  # For NICFI (& other fixed-sized ICs), set to False.
    ic = ic.filterBounds(roi)
  if roi_filter_fn:
    ic = roi_filter_fn(ic, roi=roi)
  if limit:
    ic = ic.limit(limit)
  # If IC is empty, return empty arrays.
  # TODO: Figure out if sample_roi is call twice on GEE side and
  # optimize the code if needed.
  samples = ee.Array(ee.Algorithms.If(ic.size().neq(0),
                                      sample_roi(ic, roi, scale=scale)[0],
                                      ee.Array([], ee.PixelType.uint8())))
  samples_mask = ee.Array(ee.Algorithms.If(ic.size().neq(0),
                                           sample_roi(ic, roi, scale=scale)[1],
                                           ee.Array([], ee.PixelType.uint8())))
  timestamps = ee.Array(ee.Algorithms.If(ic.size().neq(0), ic_timestamps(ic),
                                         ee.Array([], ee.PixelType.int64())))
  return samples, samples_mask, timestamps


def ic_sample_reduced(
    roi: ee.Geometry,
    ic: ee.ImageCollection,
    *,
    scale: int,
    dummy_im: ee.Image,
    reduce_fn: str = "mosaic",
    **reduce_fn_kwargs,
) -> tuple[ee.Array, ee.Array]:
  """Returns ROI samples reduced over the whole IC."""
  ic = ic.filterBounds(roi)
  reduce_fn = get_ic_reduce_fn(reduce_fn, scale=scale, roi=roi,
                               **reduce_fn_kwargs)
  reduced = ee.Algorithms.If(ic.size().neq(0),
                             reduce_fn(ic),
                             reduce_fn(ee.ImageCollection([dummy_im])))
  reduced = ee.Image(reduced)
  samples, samples_mask = sample_roi(reduced, roi, scale=scale)
  return samples, samples_mask


def ic_sample_date_ranges(
    roi: ee.Geometry,
    ic: ee.ImageCollection,
    *,
    date_ranges: tuple[str, int, int],
    scale: int,
    dummy_im: ee.Image,
    cloud_mask_fn: Callable[[ee.Image], ee.Image] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    reduce_fn: str = "mosaic",
    bands: list[str] | None = None,
    **reduce_fn_kwargs,
) -> tuple[ee.Array, ee.Array, ee.Array]:
  """Returns ROI samples reduced over given date ranges."""
  ic = ic.filterBounds(roi)
  if sort_by:
    ic = ic.sort(sort_by, ascending)

  reduce_fn = get_ic_reduce_fn(reduce_fn, scale=scale, roi=roi,
                               **reduce_fn_kwargs)
  update_cloud_mask = (
      lambda x: x.updateMask(cloud_mask_fn(x)) if cloud_mask_fn else x)
  dummy_im = reduce_fn(ee.ImageCollection([dummy_im]))
  orig_bandnames = dummy_im.bandNames()

  def _proc_date_range(date_range_feature):
    dr = ee.DateRange(date_range_feature.get("date_range"))
    dr_ic = ic.filterDate(dr)
    reduced = ee.Algorithms.If(
        dr_ic.size().neq(0),
        reduce_fn(dr_ic.map(update_cloud_mask))
        .rename(orig_bandnames),
        dummy_im)
    return ee.Image(reduced).set("timestamp", times.date_range_mean(dr))

  date_ranges = ee.FeatureCollection(
      [ee.Feature(None, {"date_range": ee.DateRange(
          dr[0], ee.Date(dr[0]).advance(dr[1], "month").advance(dr[2], "day"))})
       for dr in date_ranges]
  )
  reduced_fc = date_ranges.map(_proc_date_range)
  # If there are no images within a time range, there will be no image bands,
  # and the sampling will drop those.
  # To keep the timestamps synchronized, let's drop those bands early on.
  reduced_fc = reduced_fc.filterBounds(roi)

  reduced_ic = ee.ImageCollection(reduced_fc)
  if bands:
    reduced_ic = reduced_ic.select(bands)
  samples, samples_mask = sample_roi(reduced_ic, roi, scale=scale)
  timestamps = ic_timestamps(reduced_ic, "timestamp")
  return samples, samples_mask, timestamps


def add_roi_validity(im: ee.Image, roi: ee.Geometry, band: str = "R",
                     scale: int | None = None) -> ee.Image:
  """Returns im with `validity` property set to percentage of non-0 values."""
  im_orig = im
  if scale:
    im = im_orig.reproject(crs=roi.projection().crs(), scale=scale)
  sample = (im
            .select(band)
            .mask()
            .sampleRectangle(roi)
            .get(band))
  arr = ee.Array(sample).toFloat()
  mean = arr.reduce(ee.Reducer.mean(), [0, 1])
  mean = mean.get([0, 0])  # Returns scalar instead of (1, 1) array.
  return im_orig.set({"validity": mean})
