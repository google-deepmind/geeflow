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

"""Working with dates/times.

Ensure all dates and times are timezone aware, and always prefer to work in
UTC timezone instead of any local timezone.
"""
from collections.abc import Sequence
import datetime
import zoneinfo

from dateutil import relativedelta
import numpy as np

import ee

DateT = float | datetime.datetime | str

UTC = datetime.timezone.utc
ZRH = zoneinfo.ZoneInfo("Europe/Zurich")  # CEST(UTC+2) or CET(UTC+1).
PT = zoneinfo.ZoneInfo("America/Los_Angeles")  # PDT(UTC-7) or PST(UTC-8).
ET = zoneinfo.ZoneInfo("America/New_York")  # EDT(UTC-4) or EST(UTC-5).

TIME_FORMAT_ISO_8601 = "%Y-%m-%dT%H:%M:%S.%f%z"  # Time-zone aware.
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S-%Z"  # With time-zone string.
DATE_FORMAT = "%Y-%m-%d"


def millis_to_datetime(millis: float) -> datetime.datetime:
  return datetime.datetime.fromtimestamp(millis / 1e3, tz=UTC)


def to_timestr(d: float | datetime.datetime) -> str:
  if isinstance(d, (float, int, np.number)):
    d = millis_to_datetime(d)
  return datetime.datetime.strftime(d, TIME_FORMAT)


def to_datestr(d: float | datetime.datetime) -> str:
  if isinstance(d, (float, int, np.number)):
    d = millis_to_datetime(d)
  return datetime.datetime.strftime(d, DATE_FORMAT)


def now(tz=UTC) -> datetime.datetime:
  return datetime.datetime.now(tz)


def is_tza(d: datetime.datetime) -> bool:
  """Is given datetime timezone aware or not (is naive)?"""
  return d.tzinfo is not None and d.tzinfo.utcoffset(d) is not None


def make_tza(d: datetime.datetime) -> datetime.datetime:
  """Lazily converts datetime to timezone-aware datetime."""
  if not is_tza(d):
    d = d.replace(tzinfo=UTC)  # pylint: disable=g-tzinfo-replace
  return d


def to_datetime(d: str, dt_format: str = DATE_FORMAT) -> datetime.datetime:
  """Parses string into timezone-aware datetime object."""
  d = datetime.datetime.strptime(d, dt_format)
  return make_tza(d)


def incremental_date_list(
    start, n, *, years=0, months=0, days=0) -> Sequence[datetime.datetime]:
  """Returns list of n datetimes with given increments in years/months/days."""
  assert years or months or days, "At least one of the inc units should be set."
  if isinstance(start, (float, int, np.number)):
    start = millis_to_datetime(start)
  elif isinstance(start, str):
    start = to_datetime(start, DATE_FORMAT)
  else:
    start = make_tza(start)
  return [start + relativedelta.relativedelta(
      months=i*months, years=i*years, days=i*days) for i in range(n)]


def get_date_ranges(start: DateT, n: int, months: int = 0, months_skip: int = 0,
                    days: int = 0) -> list[tuple[str, int, int]]:
  """Returns a list with n dates and corresponding duration in months/days."""
  return [(to_datestr(x), months, days)
          for x in incremental_date_list(start, n,
                                         months=months + months_skip,
                                         days=days)]


def get_date_ranges_from_year(data, *, year_key: str = "", date_key: str = "",
                              **kwargs) -> list[tuple[str, int, int]]:
  """Returns a list with n dates starting at given year or timestamp/date."""
  assert bool(year_key) != bool(date_key), "Year or date key must be set."
  if year_key:
    return get_date_ranges(f"{data[year_key]}-01-01", **kwargs)
  else:
    return get_date_ranges(data[date_key], **kwargs)


def get_date_from_year(data, year_key: str, add_years: int = 0) -> str:
  """Returns a start of year date."""
  return f"{int(data[year_key]) + add_years}-01-01"


def adjust_for_hemisphere(data, north, south):
  if data["lat"] >= 0:
    return north
  else:
    return south


def outer_dates(date_ranges: list[tuple[str, int]], to_str=True):
  """Returns min/max dates from given list of start dates and durations."""
  dates = [to_datetime(x[0]) for x in date_ranges]
  dates += [to_datetime(x[0]) + relativedelta.relativedelta(months=x[1])
            for x in date_ranges]
  if to_str:
    return to_datestr(min(dates)), to_datestr(max(dates))
  return min(dates), max(dates)


def date_range_mean(dr: ee.DateRange) -> ee.Number:
  """Returns the mean date from a date range as milliseconds."""
  return dr.start().millis().add(dr.end().millis()).divide(2.)


def outer_date_range(date_ranges: ee.FeatureCollection) -> "ee.DateRange":
  """Returns the outer date range from a FeatureCollection."""
  fc = date_ranges.map(lambda x: ee.Feature(None, {  # pylint: disable=g-long-lambda
      "start": ee.DateRange(x.get("date_range")).start(),
      "end": ee.DateRange(x.get("date_range")).end()}))
  return ee.DateRange(fc.aggregate_min("start"), fc.aggregate_max("end"))
