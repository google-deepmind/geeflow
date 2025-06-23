# Welcome to GeeFlow

<div style="text-align: left">
<img align="right" src="https://raw.githubusercontent.com/google-deepmind/geeflow/refs/heads/main/docs/images/geeflow_logo_1.png" width="100">
</div>

<div style="
  background-color: transparent;
  padding: 5px;
  text-align: center;
  color:rgb(66, 66, 66);
  font-weight: bold;
  font-size: 1.5em;
  font-style: italic;
  margin-left: auto;
  margin-right: auto;
  width: fit-content;
">
  Python library for generating large scale geospatial datasets using Google
  Earth Engine.
</div>

<br>

<div class="mdx-hero__content">
  <a href="https://github.com/google-deepmind/geeflow" class="md-button">
    GitHub repository
  </a>
</div>

---

## Overview

GeeFlow is a Python library from Google DeepMind for creating and processing
large-scale geospatial datasets using [Google Earth Engine](https://earthengine.google.com/).
It contains tools and pipeline launch scripts to generate geospatial datasets.

GeeFlow primarily focuses on supporting geospatial AI research, and it is not
intended as a production-ready utility. The datasets created with GeeFlow
conform to the TensorFlow Datasets (TFDS) format, which lets you use them
directly with TFDS `tf.data.Dataset` data pipelines, streamlining integration
into common machine learning workflows.

The GeeFlow project is open-sourced to facilitate and accelerate research in
global-scale geospatial data preparation for AI applications. The goal is to
simplify the data preparation phase for machine learning experiments and
abstract workflows associated with Google Earth Engine for large-scale data
extraction and processing.

To access GeeFlow, visit [the GitHub repository](https://github.com/google-deepmind/geeflow).

## Key features

GeeFlow includes a set of capabilities tailored for geospatial dataset
generation and processing:

-   **Datasets:** Create small- and large-scale datasets, suitable for
    both supervised and unsupervised learning tasks. These datasets are prepared
    for direct ingestion into geospatial AI model training pipelines and include
    precomputed standard and robust statistics.
-   **Inference maps:** Generate inference maps that can scale up to global
    coverage at any resolution, letting you apply trained models over extensive
    geographical areas.
-   **Data support:** Provide support for various types of geospatial satellite
    and remote sensing data, as well as labels sourced from Google Earth Engine.
-   **Resolution and sampling:** Enable arbitrary spatial and temporal
    resolution and flexible sampling strategies for data sources, catering to
    diverse research requirements.
-   **Tooling:** Provide integrated tools for both sampling geospatial data and
    generating inference maps.

## Design

GeeFlow is designed to complement model training frameworks like
[Jeo](https://github.com/google-deepmind/jeo). While GeeFlow specializes in
generating large-scale, TFDS-compatible geospatial datasets using Google Earth
Engine, Jeo provides a framework for training machine learning models on such
datasets, particularly using JAX and Flax. A common and intended workflow
involves using GeeFlow to prepare the data, which is then ingested by Jeo or
other compatible frameworks for model development, training, and evaluation.

The existence of these separate but complementary tools provides a modular
toolchain for geospatial AI research. Therefore, GeeFlow addresses the upstream
data challenges inherent in working with large-scale Earth observation data,
while other frameworks tackles the downstream modeling challenges. For more
information, see [Out of scope](#out-of-scope).

### GitHub directory structure and conventions

The following is the directory structure of core modules in the
[GitHub repository](https://github.com/google-deepmind/geeflow):

*  `main`: all core components.
*  `stats/`: the tooling to compute data statistics.
*  `configs/`: the configuration files to generate datasets.
*  `data/`: example data.

The following are some internal conventions:

-   Unless otherwise specified, `lat` and `lon` refer to latitude and longitude,
    respectively.
-   For GPS, the system uses the
    [WGS 84](https://en.wikipedia.org/wiki/World_Geodetic_System) coordinate
    system.
-   For projected coordinates, the system uses local
    [UTM](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system)
    coordinates.

### Dataset configuration

To provide the needed flexibility for datasets, the `geeflow/configs/` directory
of the GitHub repository contains
[ML Collections](https://github.com/google/ml_collections) configuration files
for dataset configurations.

Dataset configuration is split into *label configuration* and
*source configuration*:

#### Label configuration

Consider the following aspects when configuring labels:

-   You must provide a CSV or [parquet](https://parquet.apache.org/) file with
    the locations of the samples, such as the `lat` and `lon` columns, and image
    sizes in meters (for UTM projected samples) or degrees (for spherical
    CRS).
-   If you want to include other columns in the generated dataset samples, such
    as image-level labels or metadata, provide the list to the `meta_keys`
    field.
-   You can specify the default resolution per pixel (`default_scale`) for all
    sources, which the system overwrites if the source specifies its own scale,
    and the reference maximal pixel size in meters (`max_cell_size_m`) for
    proper gridding of multi-scale sources.
-   If you want to generate separate training splits, for example, `train`,
    `val`, and `test`, include a column with the split name per sample in
    `meta_keys`. Otherwise, the system randomly generates the splits by
    geographically separating cells based on the selected
    [S2 geometry](http://s2geometry.io/) scale level.

The following is a label configuration example:

```
labels = ml_collections.ConfigDict()
labels.path = "data/demo_labels.csv"
labels.img_width_m = 240  # Image width and height of 240 meters on each side.
labels.max_cell_size_m = 30  # Reference maximal pixel size in meters.
labels.meta_keys = ("lat", "lon", "split")
labels.num_max_samples = 10  # Only for debugging, limiting the number of generated examples.
```

#### Source configuration

Consider the following aspects when configuring sources. This part contains
named sources that Google Earth Engine provides:

-   For every location `x` specified in the label configuration and every
    specified source `s`, the system creates an image tensor with shape
    `(T_s,H_s,W_s,C_s)` (temporal size, height, width, number of channels).
    `T_s` or `C_s` might be absent, and any dimension can be one. The
    dimensions can be different from source to source in dependence of the
    specified spatial resolution or scale, temporal sampling, and the selected
    channels.
-   You define at least the source class from the `geeflow/ee_data.py` file in
    the GitHub repository, where you can provide additional options, such as the
    data mode, using the `kw` field. If a source class isn't defined in the
    `ee_data.py` file, you can use a `CustomImage`, `CustomIC`, or `CustomFC`
    configuration and set all values explicitly, such as `asset_name`.
-   Other fields include the following:
      -  `scale`: resolution per pixel in meters.
      -  `select`: the bands to include in the given order.
      -  `sampling_kw`: keyword arguments for how to aggregate multiple images
         within a time range.
-   Date ranges (`date_ranges`) are a list of time ranges that aggregate the
    data for each returned time sample. A tuple of the form `[start date, number
    of months, number of days]` to aggregate over specifies each date range.

The following is a source configuration example:

```sh
sources = ml_collections.ConfigDict()

sources.s2 = utils.get_source_config("Sentinel2", "ic")
sources.s2.kw.mode = "L2A"
sources.s2.scale = 10
sources.s2.select = ["B3", "B2", "B1"]
sources.s2.sampling_kw.reduce_fn = "median"
sources.s2.sampling_kw.cloud_mask_fn = ee_data.Sentinel2.im_cloud_score_plus_mask
sources.s2.date_ranges = [("2023-01-01", 12, 0), ("2024-01-01", 12, 0)]  # 2 annual samples

sources.s1 = utils.get_source_config("Sentinel1", "ic")
sources.s1.kw = {"mode": "IW", "pols": ("VV", "VH"), "orbit": "both"}
sources.s1.scale = 10
sources.s1.sampling_kw.reduce_fn = "mean"
sources.s1.date_ranges = [("2023-01-01", 3, 0), ("2023-04-01", 3, 0)]  # 2 seasonal samples

sources.elevation = utils.get_source_config("NasaDem", "im")
sources.elevation.scale = 30
sources.elevation.select = ("elevation", "slope", "aspect")
```

The generated data in this example has different spatial, temporal, and spectral
dimensions. The temporal samples from Sentinel-2 and Sentinel-1 cover different
time ranges. This example emphasizes the flexibility of the specifications.

### Out of scope

Specific capabilities are considered out of scope from the GeeFlow library but
complement the focus on geospatial AI research:

-   **Model training and inference**: Pick a framework, for example:
    -  For [Jax](https://github.com/jax-ml/jax)/[Flax](https://github.com/google/flax),
       we use [Jeo](https://github.com/google-deepmind/jeo).
    - For [PyTorch](https://pytorch.org/), check out
      [TorchGeo](https://github.com/microsoft/torchgeo).
-  **Google Earth Engine data interactive visualization and analysis**: Check
   out, for example, [geemap](https://geemap.org/) for Python-based
   analysis. You can also explore GEE's own Javascript-based
   [EE Code Editor](https://developers.google.com/earth-engine/guides/playground).
-   **Datasets repository**: Check out, for example:
    -  [Hugging Face](https://huggingface.co/datasets)
    -  [TFDS Catalog](https://www.tensorflow.org/datasets/catalog)
    -  [TorchGeo Datasets](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#geospatial-datasets)

## Cite GeeFlow

Cite the GeeFlow codebase as follows:

```sh
@software{geeflow2025:github,
  author = {Maxim Neumann and Anton Raichuk and Michelangelo Conserva and Keith Anderson},
  title = {{GeeFlow}: Large scale datasets generation and processing for geospatial {AI} research}.
  url = {https://github.com/google-deepmind/geeflow},
  year = {2025}
}
```
