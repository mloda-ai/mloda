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

_NAMES_BODY: str = (
    _PRELUDE
    + """
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from mloda_plugins.compute_framework.base_implementations.pyarrow.arrow_build import aware_zone

def at(tz):
    return datetime(2023, 3, 11, 12, 0, tzinfo=tz)

try:
    cases = [
        (ZoneInfo("America/New_York"), "America/New_York"),
        (timezone.utc, "UTC"),
        (timezone(timedelta(hours=5, minutes=30)), "+05:30"),
    ]
    for tz, want in cases:
        got = aware_zone([at(tz), None], "zoned")
        print("NAME:" + want + ":" + str(got))
except Exception as e:
    print("NAME_FAILED:" + type(e).__name__ + ":" + str(e))

try:
    aware_zone([at(ZoneInfo("America/New_York")), at(timezone(timedelta(hours=5, minutes=30)))], "zoned")
    print("MIXED_NO_RAISE")
except ValueError as e:
    ok = "zoned" in str(e) and "America/New_York" in str(e) and "+05:30" in str(e)
    print("MIXED_OK" if ok else "MIXED_WRONGMSG:" + str(e))
except Exception as e:
    print("MIXED_WRONG:" + type(e).__name__ + ":" + str(e))
"""
)

_BUILD_BODY: str = (
    _PRELUDE
    + """
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

framework = PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
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
    for want in ["America/New_York", "UTC", "+05:30"]:
        assert f"NAME:{want}:{want}" in out, f"stdout: {out!r}"
    assert "MIXED_OK" in out, f"stdout: {out!r}"


@pytest.mark.timeout(30)
def test_polars_dict_build_aware_zones_without_pyarrow() -> None:
    out = _run(_BUILD_BODY)
    assert "MIXED_OK" in out, f"stdout: {out!r}"
    assert "SINGLE_OK" in out, f"stdout: {out!r}"
