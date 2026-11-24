"""Zone fixtures for tests of aware datetime builders."""

import io
import zoneinfo
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

_NY = ZoneInfo("America/New_York")
_PLUS_0530 = timezone(timedelta(hours=5, minutes=30))


class MyZone(tzinfo):
    """A fixed-offset tzinfo that pyarrow can name only as "MYZONE", which it cannot read back."""

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return timedelta(hours=3)

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, dt: datetime | None) -> str:
        return "MYZONE"


def keyless_ny() -> ZoneInfo:
    """A ZoneInfo without a key, read from the system America/New_York tzfile; skips when absent."""
    for root in zoneinfo.TZPATH:
        path = Path(root) / "America" / "New_York"
        if path.is_file():
            return ZoneInfo.from_file(io.BytesIO(path.read_bytes()))
    pytest.skip("no system America/New_York tzfile")


def bad_zone_values(kind: str) -> tuple[list[Any], list[str]]:
    """Two aware values for kind mixed, keyless or custom, and the strings the error must contain."""
    if kind == "mixed":
        return [
            datetime(2023, 3, 11, 12, 0, tzinfo=_NY),
            datetime(2023, 3, 11, 12, 0, tzinfo=_PLUS_0530),
        ], ["America/New_York", "+05:30"]
    if kind == "keyless":
        return [datetime(2023, 3, 11, 12, 0, tzinfo=keyless_ny())] * 2, ["ZoneInfo"]
    return [datetime(2023, 3, 11, 12, 0, tzinfo=MyZone())] * 2, ["MYZONE"]
