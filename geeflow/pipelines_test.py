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

from typing import Any
from unittest import mock

from absl.testing import absltest
from geeflow import pipelines
from ml_collections import config_dict


def _get_fc_to_image_props(cfg_dict: dict[str, Any]) -> list[str] | None:
  cfg = config_dict.ConfigDict(cfg_dict)
  mock_algo = mock.MagicMock(return_value=mock.MagicMock())
  pipelines._handle_fc_to_image(
      algo=mock_algo,
      asset=mock.sentinel.asset,
      roi=mock.sentinel.roi,
      name="test_source",
      cfg=cfg,
      mask_value=0,
      kw={},
  )
  return mock_algo.call_args.kwargs["props"]


class HandleFcToImageTest(absltest.TestCase):

  def test_select_final_none_falls_back_to_select(self):
    """Preset default select_final=None must not shadow an explicit select."""
    props = _get_fc_to_image_props({
        "select": ["ADM0_CODE", "ADM1_CODE"],
        "select_final": None,
    })
    self.assertEqual(props, ["ADM0_CODE", "ADM1_CODE"])

  def test_select_final_overrides_select(self):
    props = _get_fc_to_image_props({
        "select": ["ADM0_CODE", "ADM1_CODE"],
        "select_final": ["ADM0_CODE"],
    })
    self.assertEqual(props, ["ADM0_CODE"])

  @mock.patch.object(pipelines, "_add_mask_and_rename")
  @mock.patch.object(pipelines.ee, "Image")
  def test_handle_get_image_from_item_applies_cloud_mask_before_select(
      self, mock_ee_image, mock_add_mask
  ):
    raw_im = mock.MagicMock()
    masked_im = mock.MagicMock()
    selected_im = mock.MagicMock()
    mock_ee_image.return_value = raw_im
    raw_im.updateMask.return_value = masked_im
    masked_im.select.return_value = selected_im
    cloud_mask_fn = mock.MagicMock(return_value=mock.sentinel.cloud_mask)
    cfg = config_dict.ConfigDict(
        {"asset_id_key": "asset_id", "select": ["R", "G", "B"]}
    )
    pipelines._handle_get_image_from_item(
        name="planet",
        cfg=cfg,
        mask_value=0,
        item={"asset_id": "projects/p/assets/img1"},
        kw={"cloud_mask_fn": cloud_mask_fn},
    )
    cloud_mask_fn.assert_called_once_with(raw_im)
    raw_im.updateMask.assert_called_once_with(mock.sentinel.cloud_mask)
    masked_im.select.assert_called_once_with(["R", "G", "B"])
    mock_add_mask.assert_called_once_with(selected_im, "planet", "", 0)

  def test_resolve_date_range_supports_static_and_dynamic(self):
    self.assertEqual(
        pipelines.resolve_date_range(
            {"date_range": ["2020-01-01", "2021-01-01"]}, {}
        ),
        ("2020-01-01", "2021-01-01"),
    )
    self.assertEqual(
        pipelines.resolve_date_range(
            {
                "date_range_fn": lambda it: (
                    f"{it['year']}-01-01",
                    f"{it['year'] + 1}-01-01",
                )
            },
            {"year": 2022},
        ),
        ("2022-01-01", "2023-01-01"),
    )
    with self.assertRaises(ValueError):
      pipelines.resolve_date_range(
          {
              "date_range": ("2020-01-01", "2021-01-01"),
              "date_range_fn": lambda _: ("2020-01-01", "2021-01-01"),
          },
          {},
      )

  def test_resolve_date_ranges_supports_static_and_dynamic(self):
    self.assertEqual(
        pipelines.resolve_date_ranges(
            {"date_ranges": [["2020-01-01", 6, 0]]}, {}
        ),
        (("2020-01-01", 6, 0),),
    )
    self.assertEqual(
        pipelines.resolve_date_ranges(
            {"date_ranges_fn": lambda it: [(f"{it['year']}-01-01", 12, 0)]},
            {"year": 2023},
        ),
        (("2023-01-01", 12, 0),),
    )
    with self.assertRaises(ValueError):
      pipelines.resolve_date_ranges(
          {
              "date_ranges": [("2020-01-01", 6, 0)],
              "date_ranges_fn": lambda _: [("2020-01-01", 6, 0)],
          },
          {},
      )


if __name__ == "__main__":
  absltest.main()
