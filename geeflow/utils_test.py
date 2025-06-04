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

from absl.testing import absltest
from absl.testing import parameterized
from geeflow import utils


class CommonTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_parse_arg_works(self, lazy):
    spec = dict(
        res=224,
        lr=0.1,
        runlocal=False,
        schedule="short",
    )

    def check(result, runlocal, schedule, res, lr):
      self.assertEqual(result.runlocal, runlocal)
      self.assertEqual(result.schedule, schedule)
      self.assertEqual(result.res, res)
      self.assertEqual(result.lr, lr)
      self.assertIsInstance(result.runlocal, bool)
      self.assertIsInstance(result.schedule, str)
      self.assertIsInstance(result.res, int)
      self.assertIsInstance(result.lr, float)

    check(utils.parse_arg(None, lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("", lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("runlocal=True", lazy=lazy, **spec),
          True, "short", 224, 0.1)
    check(utils.parse_arg("runlocal=False", lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("runlocal=", lazy=lazy, **spec),
          False, "short", 224, 0.1)
    check(utils.parse_arg("runlocal", lazy=lazy, **spec),
          True, "short", 224, 0.1)
    check(utils.parse_arg("res=128", lazy=lazy, **spec),
          False, "short", 128, 0.1)
    check(utils.parse_arg("128", lazy=lazy, **spec),
          False, "short", 128, 0.1)
    check(utils.parse_arg("schedule=long", lazy=lazy, **spec),
          False, "long", 224, 0.1)
    check(utils.parse_arg("runlocal,schedule=long,res=128",
                          lazy=lazy, **spec),
          True, "long", 128, 0.1)

  @parameterized.parameters(
      (None, {}, {}),
      (None, {"res": 224}, {"res": 224}),
      ("640", {"res": 224}, {"res": 640}),
      ("runlocal", {}, {"runlocal": True}),
      ("res=640,lr=0.1,runlocal=false,schedule=long", {},
       {"res": 640, "lr": 0.1, "runlocal": False, "schedule": "long"}),
      )
  def test_lazy_parse_arg_works(self, arg, spec, expected):
    self.assertEqual(dict(utils.parse_arg(arg, lazy=True, **spec)),
                     expected)


if __name__ == "__main__":
  absltest.main()
