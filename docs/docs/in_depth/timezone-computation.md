# Time Zone Computation Rule

The [comparison contract](comparison-contract.md) says when timezone-aware and timezone-naive columns
may mix. This page says what a zone means once a computation runs on a timezone-aware column: which
clock applies, how a length is measured across a daylight saving change, and what a compute framework
does with the zone. Feature groups, operation references and compute frameworks follow it.

## The rule

A timezone-aware value is an instant plus a zone. The zone never changes which instant it is, only the
calendar it is read on.

1. **Instant operations ignore the zone.** Comparing, sorting, joining, as-of matching and subtracting
   two timestamps work on instants, so two aware columns in different zones compare correctly.
2. **Calendar operations use the column's own zone.** Field extraction (year, month, day, hour,
   weekday), bucket boundaries and calendar arithmetic read the wall clock of the column's zone, never
   UTC and never a session or system zone.
3. **The unit decides the clock.** `second`, `minute` and `hour` are elapsed time. `day`, `week`,
   `month` and `year` are calendar units on the wall clock.
4. **A length or a midpoint is elapsed time between calendar boundaries.** The calendar finds the
   boundaries, the clock measures the distance. A day is 23 to 25 hours and a month 28 to 31 days, and
   rounding compares against half of that real length.
5. **A wall time that does not exist or exists twice is resolved, never an error.** A time inside a
   gap moves forward by the length of the gap. A repeated time keeps the offset of the value it was
   computed from when that offset is one of the two, and is the earlier instant otherwise.
6. **A timestamp result keeps the zone and the unit of its input.**

| Unit | Clock | Adding one |
|------|-------|------------|
| `second`, `minute`, `hour` | elapsed time | exactly that long later |
| `day`, `week` | wall clock of the column's zone | the same wall time on the later date |
| `month`, `year` | wall clock of the column's zone | the same day and wall time in the later month or year, clamped to the month end |

On naive, UTC and fixed-offset columns both clocks agree. The rule only changes results in a zone with
daylight saving time, and `24_hour` stays available where `1_day` is not what is meant.

## Worked example: a 23 hour day

America/New_York skips from 02:00 to 03:00 on Sunday 2023-03-12, so that day has 23 hours.

| Computation | Result | Why |
|-------------|--------|-----|
| Saturday 12:00 plus 1 `day` | Sunday 12:00 | calendar unit: the same wall time, 23 hours later |
| Saturday 12:00 plus 24 `hour` | Sunday 13:00 | exact unit: 24 elapsed hours |
| Midpoint of Sunday when rounding to the day | 12:30 | half of 23 hours after local midnight, so 12:00 rounds down |
| A `1_day` window ending Sunday 12:00 | starts after Saturday 12:00 | 23 elapsed hours wide |
| A `24_hour` window ending Sunday 12:00 | starts after Saturday 11:00 | 24 elapsed hours wide |

`shift` is core's reference implementation of rules 3 and 5:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from mloda_plugins.feature_group.experimental.calendar_shift import shift

saturday_noon = datetime(2023, 3, 11, 12, tzinfo=ZoneInfo("America/New_York"))

assert shift(saturday_noon, 1, "day").isoformat() == "2023-03-12T12:00:00-04:00"
assert shift(saturday_noon, 24, "hour").isoformat() == "2023-03-12T13:00:00-04:00"
```

Python's own `aware + timedelta(hours=24)` adds on the wall clock, so it gives Sunday 12:00 here. Measure
an elapsed length through UTC.

## Gaps and repeated times

| Wall time in America/New_York | Reached from | Resolved to |
|-------------------------------|--------------|-------------|
| 2023-03-12 02:30 (does not exist) | any value | 03:30 at UTC-4, forward by the one hour gap |
| 2023-11-05 01:30 (exists twice) | 2023-11-04 01:30 at UTC-4 | 01:30 at UTC-4, the earlier instant |
| 2023-11-05 01:30 (exists twice) | 2023-11-06 01:30 at UTC-5 | 01:30 at UTC-5, the offset of the source value |

## Compute frameworks

A transfer into or out of a compute framework keeps the instant and the zone. A framework does one of:

- hold the zone in its native column type;
- store instants only, carry the zone beside the column, give its Implementations the wall clock of
  that zone and re-attach the zone on the way out;
- reject the column with a `ValueError` that names the column and the zone.

Handing the same instants back under another zone is not one of them. Every calendar operation then
gives a valid but different answer that nothing downstream can detect, so a
[divergence record](divergence-records.md) does not cover it.

## Core feature groups

| Group | Application |
|-------|-------------|
| `TimeWindowFeatureGroup` | A row at time `t` aggregates the rows in `(t - span, t]`. Membership is decided on instants (rule 1); `t - span` follows rule 3, so a `1_day` window starts at the same wall time one day earlier and a `1_month` window on the same day of the previous month. |
| `ForecastingFeatureGroup` | Horizon steps follow rule 3, and the time features of each step are read on the wall clock of the reference column's zone (rule 2). |

An Implementation that cannot follow the rule declares a [divergence record](divergence-records.md).
