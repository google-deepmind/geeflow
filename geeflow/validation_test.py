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

"""Tests for config structural validation.

These tests validate the structural integrity of GeeFlow source configs without
requiring Earth Engine authentication. They catch:
  - Missing required fields (module, algo)
  - Typos in field names (unknown keys)
  - Source filter referencing non-existent source names
  - Inconsistencies between presets and config consumers
"""

from absl.testing import absltest
from absl.testing import parameterized
from geeflow import ee_algo
from geeflow import ee_data
from geeflow import validation
from geeflow.data_sources import presets
from ml_collections import config_dict


class PresetsStructureTest(parameterized.TestCase):
  """Tests that presets.get_all() entries are structurally valid."""

  def setUp(self):
    super().setUp()
    self.sources = presets.get_all()

  def test_all_sources_have_module(self):
    """Every source must have a 'module' field pointing to an ee_data class."""
    for name, src in self.sources.items():
      with self.subTest(source=name):
        self.assertIn("module", src, f"Source '{name}' missing 'module' field.")
        module_name = src.module
        self.assertTrue(
            hasattr(ee_data, module_name),
            f"Source '{name}' has module='{module_name}' which doesn't exist "
            "in ee_data.",
        )

  def test_all_sources_have_valid_out(self):
    """Every source must have a valid 'out' field."""
    valid_outs = {
        "ic",
        "im",
        "fc",
        "filter_by_cloud_percentage",
        "filter_by_cadence",
        "filter_and_sort",
    }
    for name, src in self.sources.items():
      with self.subTest(source=name):
        self.assertIn("out", src, f"Source '{name}' missing 'out' field.")
        out = src.out
        if out not in valid_outs:
          # Check that the out method exists on the ee_data class.
          module_cls = getattr(ee_data, src.module)
          self.assertTrue(
              hasattr(module_cls, out)
              or hasattr(module_cls(**src.get("kw", {})), out),
              f"Source '{name}' has out='{out}' which doesn't exist on "
              f"{src.module}.",
          )

  def test_all_sources_only_have_known_keys(self):
    """No source should have keys outside the known set (catches typos)."""
    for name, src in self.sources.items():
      with self.subTest(source=name):
        unknown = set(src.keys()) - validation.SOURCE_CONFIG_KEYS
        self.assertEmpty(
            unknown,
            f"Source '{name}' has unknown keys: {unknown}. "
            "If these are intentional, add them to SOURCE_CONFIG_KEYS.",
        )

  def test_sources_with_algo_have_valid_algo(self):
    """Sources with non-None algo must reference a valid ee_algo function."""
    valid_algo_names = {
        # Canonical names.
        "ic_stack",
        "ic_reduced",
        "ic_temporal_composites",
        "im_from_item",
        "im_static",
        "fc_properties",
        "fc_rasterize",
        "ccdc_coefficients",
        # Deprecated aliases (kept for backwards compatibility).
        "sample_roi",
        "ic_sample",
        "ic_sample_date_ranges",
        "ic_sample_reduced",
        "get_ccdc",
        "fc_get",
        "fc_to_image",
        "get_image_from_item",
    }
    for name, src in self.sources.items():
      algo = src.get("algo")
      if algo is None:
        continue
      with self.subTest(source=name):
        if isinstance(algo, str):
          self.assertIn(
              algo,
              valid_algo_names,
              f"Source '{name}' has unknown algo='{algo}'.",
          )
          self.assertTrue(
              hasattr(ee_algo, algo),
              f"Source '{name}' algo='{algo}' not found in ee_algo module.",
          )

  def test_ic_date_range_sources_have_date_related_fields_or_none(self):
    """Sources using ic_sample_date_ranges should have date config or accept it dynamically."""
    # These sources need date_ranges to be set somewhere (either in presets
    # defaults or in the project config that deepcopy's them).
    # This test just verifies the field structure is correct when present.
    for name, src in self.sources.items():
      algo = src.get("algo")
      if algo != "ic_sample_date_ranges":
        continue
      with self.subTest(source=name):
        if "date_ranges" in src and src.date_ranges is not None:
          self.assertIsInstance(
              src.date_ranges,
              (list, tuple),
              f"Source '{name}' date_ranges should be a list/tuple.",
          )


class ConfigInstantiationTest(parameterized.TestCase):
  """Tests that config files can be instantiated and produce valid structure."""

  def test_demo_config_instantiation(self):
    """The demo config should instantiate without error."""
    from geeflow.configs.public import demo

    config = demo.get_config()
    self.assertIn("sources", config)
    self.assertIn("labels", config)
    # Every source in the demo config should have at minimum the core fields.
    for name, src in config.sources.items():
      with self.subTest(source=name):
        self.assertIn("module", src, f"Demo source '{name}' missing 'module'.")

  def test_demo_config_sources_have_algo(self):
    """All sources in the demo config should have a valid algo."""
    from geeflow.configs.public import demo

    config = demo.get_config()
    for name, src in config.sources.items():
      with self.subTest(source=name):
        algo = src.get("algo")
        self.assertIsNotNone(
            algo,
            f"Demo source '{name}' has algo=None. It must be set before use.",
        )


class ValidateSourcesConfigTest(parameterized.TestCase):
  """Tests for the validate_sources_config utility function."""

  def test_valid_minimal_config_passes(self):
    """A minimal valid source config should pass validation."""
    sources = config_dict.ConfigDict({
        "s1": config_dict.create(
            module="Sentinel1",
            kw={"mode": "IW", "pols": ("VV", "VH"), "orbit": "both"},
            out="ic",
            out_kw={},
            select=None,
            select_final=None,
            sampling_kw={},
            scale=10.0,
            algo="ic_sample_date_ranges",
        ),
    })
    config = config_dict.ConfigDict(
        {"sources": sources, "labels": config_dict.ConfigDict()}
    )
    # Should not raise.
    validation.validate_sources_config(config)

  def test_shape_key_passes(self):
    """A source with explicit 'shape' should pass validation."""
    sources = config_dict.ConfigDict({
        "s1": config_dict.create(
            module="Sentinel1",
            kw={"mode": "IW", "pols": ("VV", "VH"), "orbit": "both"},
            out="ic",
            out_kw={},
            select=None,
            select_final=None,
            sampling_kw={},
            scale=10.0,
            algo="ic_sample_date_ranges",
            shape=(150, 150, 2),
        ),
    })
    config = config_dict.ConfigDict(
        {"sources": sources, "labels": config_dict.ConfigDict()}
    )
    # Should not raise.
    validation.validate_sources_config(config)

  def test_missing_module_raises(self):
    """A source without 'module' should be caught."""
    sources = config_dict.ConfigDict({
        "bad_source": config_dict.create(
            kw={},
            out="ic",
            out_kw={},
            select=None,
            select_final=None,
            sampling_kw={},
            scale=10.0,
            algo="ic_sample_date_ranges",
        ),
    })
    config = config_dict.ConfigDict(
        {"sources": sources, "labels": config_dict.ConfigDict()}
    )
    with self.assertRaises(ValueError):
      validation.validate_sources_config(config)

  def test_missing_algo_raises(self):
    """A source without 'algo' should be caught."""
    sources = config_dict.ConfigDict({
        "bad_source": config_dict.create(
            module="Sentinel1",
            kw={},
            out="ic",
            out_kw={},
            select=None,
            select_final=None,
            sampling_kw={},
            scale=10.0,
        ),
    })
    config = config_dict.ConfigDict(
        {"sources": sources, "labels": config_dict.ConfigDict()}
    )
    with self.assertRaises(ValueError):
      validation.validate_sources_config(config)

  def test_unknown_keys_warns(self):
    """A source with unknown keys should log a warning."""
    sources = config_dict.ConfigDict({
        "s1": config_dict.create(
            module="Sentinel1",
            kw={},
            out="ic",
            out_kw={},
            select=None,
            select_final=None,
            sampling_kw={},
            scale=10.0,
            algo="ic_sample_date_ranges",
            typo_field="oops",
        ),
    })
    config = config_dict.ConfigDict(
        {"sources": sources, "labels": config_dict.ConfigDict(),
         "fail_if_unknown_keys": False}
    )
    with self.assertLogs(level="WARNING") as cm:
      validation.validate_sources_config(config)
    self.assertTrue(
        any("typo_field" in msg for msg in cm.output),
        f"Expected warning about 'typo_field', got: {cm.output}",
    )

  def test_ee_asset_module_passes(self):
    """A source with a direct EE asset as module should pass."""
    # We can't create a real EE asset here, so just test string modules.
    sources = config_dict.ConfigDict({
        "s1": config_dict.create(
            module="Sentinel1",
            kw={},
            out="ic",
            out_kw={},
            select=None,
            select_final=None,
            sampling_kw={},
            scale=10.0,
            algo="ic_sample_date_ranges",
        ),
    })
    config = config_dict.ConfigDict(
        {"sources": sources, "labels": config_dict.ConfigDict()}
    )
    validation.validate_sources_config(config)


class ValidateLabelsConfigTest(parameterized.TestCase):
  """Tests for the validate_labels_config utility function."""

  def _make_config(self, labels_dict):
    return config_dict.ConfigDict({
        "sources": config_dict.ConfigDict(),
        "labels": config_dict.ConfigDict(labels_dict),
    })

  def test_valid_labels_passes(self):
    """A valid labels config should pass validation."""
    config = self._make_config({
        "path": "/some/path.csv",
        "img_width_m": 240,
        "max_cell_size_m": 60,
        "default_scale": 30,
        "meta_keys": ("lat", "lon"),
    })
    # Should not raise.
    validation.validate_labels_config(config)

  def test_skip_failing_plots_passes(self):
    """A labels config with skip_failing_plots should pass validation."""
    config = self._make_config({
        "path": "/some/path.csv",
        "skip_failing_plots": True,
    })
    # Should not raise.
    validation.validate_labels_config(config)

  def test_empty_labels_passes(self):
    """An empty labels config should pass (no keys to check)."""
    config = self._make_config({})
    validation.validate_labels_config(config)

  def test_unknown_label_key_raises(self):
    """A labels config with an unknown key should raise."""
    config = self._make_config({
        "path": "/some/path.csv",
        "img_width_m": 240,
        "max_cell_size_m": 60,
        "typo_key": "oops",
    })
    with self.assertRaises(ValueError):
      validation.validate_labels_config(config)

  def test_unknown_label_key_warns_when_not_failing(self):
    """Unknown label keys should warn when fail_if_unknown_keys=False."""
    config = self._make_config({
        "path": "/some/path.csv",
        "typo_key": "oops",
    })
    config.fail_if_unknown_keys = False
    with self.assertLogs(level="WARNING") as cm:
      validation.validate_labels_config(config)
    self.assertTrue(
        any("typo_key" in msg for msg in cm.output),
        f"Expected warning about 'typo_key', got: {cm.output}",
    )


class ValidateTopLevelConfigTest(parameterized.TestCase):
  """Tests for the validate_top_level_config utility function."""

  def test_valid_top_level_passes(self):
    """A config with only known top-level keys should pass."""
    config = config_dict.ConfigDict({
        "sources": config_dict.ConfigDict(),
        "labels": config_dict.ConfigDict(),
    })
    validation.validate_top_level_config(config)

  def test_unknown_top_level_key_raises(self):
    """A config with an unknown top-level key should raise."""
    config = config_dict.ConfigDict({
        "sources": config_dict.ConfigDict(),
        "labels": config_dict.ConfigDict(),
        "typo_top_key": 42,
    })
    with self.assertRaises(ValueError):
      validation.validate_top_level_config(config)

  def test_known_optional_keys_pass(self):
    """Optional top-level keys like export, skip_keys should pass."""
    config = config_dict.ConfigDict({
        "sources": config_dict.ConfigDict(),
        "labels": config_dict.ConfigDict(),
        "export": config_dict.ConfigDict(),
        "skip_keys": ["split"],
        "post_process_map": None,
    })
    validation.validate_top_level_config(config)


if __name__ == "__main__":
  absltest.main()
