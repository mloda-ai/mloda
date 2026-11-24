"""Test that aware datetime zone naming and Polars dict builds work when pyarrow is absent."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_PRELUDE: str = """
import sys

try:
    import polars
except ImportError:
    print("POLARS_MISSING")
    sys.exit(0)
"""

_NAMES_BODY: str = """
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from mloda_plugins.compute_framework.base_implementations.pyarrow.arrow_build import aware_zone

from datetime import tzinfo
from tests.helpers.time_zones import MyZone

def at(tz):
    return datetime(2023, 3, 11, 12, 0, tzinfo=tz)

class PytzLike(tzinfo):
    zone = "America/New_York"

    def utcoffset(self, dt):
        return ZoneInfo("America/New_York").utcoffset(dt)
    def dst(self, dt):
        return ZoneInfo("America/New_York").dst(dt)
    def tzname(self, dt):
        return ZoneInfo("America/New_York").tzname(dt)

cases = [
    (ZoneInfo("America/New_York"), "America/New_York"),
    (timezone.utc, "UTC"),
    (timezone(timedelta(hours=5, minutes=30)), "+05:30"),
    (timezone(timedelta(hours=-5)), "-05:00"),
    (timezone(timedelta(0), "GMT"), "+00:00"),
    (PytzLike(), "America/New_York"),
]
try:
    from dateutil import tz as dateutil_tz

    cases.append((dateutil_tz.UTC, "UTC"))
except ImportError:
    print("DATEUTIL_SKIP")
for tz, want in cases:
    try:
        got = aware_zone([at(tz), None], "zoned")
        print("NAME:" + want + ":" + str(got))
    except Exception as e:
        print("NAME_FAILED:" + want + ":" + type(e).__name__ + ":" + str(e))

import io
from pathlib import Path
import zoneinfo

keyless = None
for root in zoneinfo.TZPATH:
    path = Path(root) / "America" / "New_York"
    if path.is_file():
        keyless = ZoneInfo.from_file(io.BytesIO(path.read_bytes()))
        break
if keyless is None:
    print("KEYLESS_SKIP")

bad = [("SUBMIN", timezone(timedelta(seconds=30))), ("CUSTOM", MyZone())]
if keyless is not None:
    bad.append(("KEYLESS", keyless))
for label, tz in bad:
    try:
        aware_zone([at(tz)], "zoned")
        print(label + "_NO_RAISE")
    except ValueError as e:
        ok = "Cannot name" in str(e) and "zoned" in str(e)
        print(label + "_OK" if ok else label + "_WRONGMSG:" + str(e))
    except Exception as e:
        print(label + "_WRONG:" + type(e).__name__ + ":" + str(e))

try:
    aware_zone([at(ZoneInfo("America/New_York")), at(timezone(timedelta(hours=5, minutes=30)))], "zoned")
    print("MIXED_NO_RAISE")
except ValueError as e:
    ok = "zoned" in str(e) and "America/New_York" in str(e) and "+05:30" in str(e)
    print("MIXED_OK" if ok else "MIXED_WRONGMSG:" + str(e))
except Exception as e:
    print("MIXED_WRONG:" + type(e).__name__ + ":" + str(e))
"""

_BUILD_BODY: str = (
    _PRELUDE
    + """
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from mloda.user import ParallelizationMode
import polars as pl
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

framework = PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
lazy_framework = PolarsLazyDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
ny = datetime(2023, 3, 11, 12, 0, tzinfo=ZoneInfo("America/New_York"))
kolkata = datetime(2023, 3, 11, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

try:
    framework.transform({"zoned": [ny, kolkata]}, ["zoned"])
    print("MIXED_NO_RAISE")
except ValueError as e:
    ok = "zoned" in str(e) and "America/New_York" in str(e) and "+05:30" in str(e)
    print("MIXED_OK" if ok else "MIXED_WRONGMSG:" + str(e))
except Exception as e:
    print("MIXED_WRONG:" + type(e).__name__ + ":" + str(e))

def expect_mixed(label, call):
    try:
        call()
        print(label + "_NO_RAISE")
    except ValueError as e:
        ok = "zoned" in str(e) and "America/New_York" in str(e) and "+05:30" in str(e)
        print(label + "_OK" if ok else label + "_WRONGMSG:" + str(e))
    except Exception as e:
        print(label + "_WRONG:" + type(e).__name__ + ":" + str(e))

mixed = {"zoned": [ny, kolkata]}
expect_mixed("APPEND", lambda: framework._append_columns(pl.DataFrame({"id": [1, 2]}), mixed))
expect_mixed("LAZY_APPEND", lambda: lazy_framework._append_columns(pl.LazyFrame({"id": [1, 2]}), mixed))
expect_mixed("LAZY_BUILD", lambda: lazy_framework.transform(mixed, ["zoned"]))

try:
    plus5 = timezone(timedelta(hours=5))
    values = [ny.astimezone(plus5), None]
    frame = framework.transform({"zoned": values}, ["zoned"])
    got = frame["zoned"].to_list()
    same = got[0] == values[0] and got[0].utcoffset() == values[0].utcoffset() and got[1] is None
    print("WHOLE_HOUR_OK" if frame.schema["zoned"].time_zone == "Etc/GMT-5" and same else "WHOLE_HOUR_WRONG:" + str(got))
except Exception as e:
    print("WHOLE_HOUR_FAILED:" + type(e).__name__ + ":" + str(e))

try:
    framework.transform({"zoned": [kolkata]}, ["zoned"])
    print("SUB_HOUR_NO_RAISE")
except ValueError as e:
    print("SUB_HOUR_OK" if "zoned" in str(e) and "+05:30" in str(e) else "SUB_HOUR_WRONGMSG:" + str(e))
except Exception as e:
    print("SUB_HOUR_WRONG:" + type(e).__name__ + ":" + str(e))

for order, naive_mix in [("AWARE_NAIVE", [ny, ny.replace(tzinfo=None)]), ("NAIVE_AWARE", [ny.replace(tzinfo=None), ny])]:
    try:
        framework.transform({"zoned": naive_mix}, ["zoned"])
        print(order + "_NO_RAISE")
    except ValueError as e:
        ok = "zoned" in str(e) and "naive" in str(e) and "aware" in str(e)
        print(order + "_OK" if ok else order + "_WRONGMSG:" + str(e))
    except Exception as e:
        print(order + "_WRONG:" + type(e).__name__ + ":" + str(e))

try:
    frame = framework.transform({"zoned": [ny, None]}, ["zoned"])
    print("SINGLE_OK" if frame.schema["zoned"].time_zone == "America/New_York" else "SINGLE_WRONGZONE")
except Exception as e:
    print("SINGLE_FAILED:" + type(e).__name__ + ":" + str(e))
"""
)


def _run(body: str) -> str:
    result = run_blocked(body)
    if "POLARS_MISSING" in result.stdout:
        pytest.skip("polars not installed in this environment")
    assert result.returncode == 0, f"Body crashed.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


@pytest.mark.timeout(30)
def test_aware_zone_names_zones_without_pyarrow() -> None:
    out = _run(_NAMES_BODY)
    for want in ["America/New_York", "UTC", "+05:30", "-05:00", "+00:00"]:
        assert f"NAME:{want}:{want}" in out, f"stdout: {out!r}"
    assert out.count("NAME:America/New_York:America/New_York") == 2, f"stdout: {out!r}"
    assert out.count("NAME:UTC:UTC") == 2 or "DATEUTIL_SKIP" in out, f"stdout: {out!r}"
    assert "MIXED_OK" in out, f"stdout: {out!r}"
    assert "SUBMIN_OK" in out, f"stdout: {out!r}"
    assert "CUSTOM_OK" in out, f"stdout: {out!r}"
    assert "KEYLESS_OK" in out or "KEYLESS_SKIP" in out, f"stdout: {out!r}"


@pytest.mark.timeout(30)
def test_polars_dict_build_aware_zones_without_pyarrow() -> None:
    out = _run(_BUILD_BODY)
    assert "MIXED_OK" in out, f"stdout: {out!r}"
    for label in ["APPEND", "LAZY_APPEND", "LAZY_BUILD"]:
        assert f"{label}_OK" in out, f"stdout: {out!r}"
    assert "SINGLE_OK" in out, f"stdout: {out!r}"
    for label in ["WHOLE_HOUR", "SUB_HOUR", "AWARE_NAIVE", "NAIVE_AWARE"]:
        assert f"{label}_OK" in out, f"stdout: {out!r}"
