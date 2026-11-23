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

For `day` and `week`, both clocks agree on naive, UTC and fixed-offset columns, so those only differ in a
zone with daylight saving time. `month` and `year` are calendar units on every column (a month is not 30 days).
`24_hour` stays available where `1_day` is not what is meant, and `calendar_units="fixed_length"` gives the
fixed lengths (day 24 hours, week 7 days, month 30 days, year 365 days).

## Worked example: a 23 hour day

America/New_York skips from 02:00 to 03:00 on Sunday 2023-03-12, so that day has 23 hours.

| Computation | Result | Why |
|-------------|--------|-----|
| Saturday 12:00 plus 1 `day` | Sunday 12:00 | calendar unit: the same wall time, 23 hours later |
| Saturday 12:00 plus 24 `hour` | Sunday 13:00 | exact unit: 24 elapsed hours |
| Midpoint of Sunday when rounding to the day | 12:30 | half of 23 hours after local midnight, so 12:00 rounds down |
| A `1_day` window ending Sunday 12:00 | starts after Saturday 12:00 | 23 elapsed hours wide |
| A `24_hour` window ending Sunday 12:00 | starts after Saturday 11:00 | 24 elapsed hours wide |

`DEFAULT_TIME_RULE.shift` is core's reference implementation of rules 3 and 5:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from mloda.provider import DEFAULT_TIME_RULE

saturday_noon = datetime(2023, 3, 11, 12, tzinfo=ZoneInfo("America/New_York"))

assert DEFAULT_TIME_RULE.shift(saturday_noon, 1, "day").isoformat() == "2023-03-12T12:00:00-04:00"
assert DEFAULT_TIME_RULE.shift(saturday_noon, 24, "hour").isoformat() == "2023-03-12T13:00:00-04:00"
```

Python's own `aware + timedelta(hours=24)` adds on the wall clock, so it gives Sunday 12:00 here. Measure
an elapsed length through UTC.

## Gaps and repeated times

| Wall time in America/New_York | Reached from | Resolved to |
|-------------------------------|--------------|-------------|
| 2023-03-12 02:30 (does not exist) | any value | 03:30 at UTC-4, forward by the one hour gap |
| 2023-11-05 01:30 (exists twice) | 2023-11-04 01:30 at UTC-4 | 01:30 at UTC-4, the earlier instant |
| 2023-11-05 01:30 (exists twice) | 2023-11-06 01:30 at UTC-5 | 01:30 at UTC-5, the offset of the source value |

## Choosing another rule for a feature

A feature can pick another rule through the `time_rule` group option, a mapping of these knobs:

| Knob | Values (default first) | Meaning |
|------|------------------------|---------|
| `calendar_units` | `wall_clock`, `fixed_length` | `fixed_length` makes day 24 hours, week 7 days, month 30 days and year 365 days; the other knobs then never apply |
| `gap` | `forward`, `raise` | a wall time that does not exist moves forward by the gap, or raises `ValueError` |
| `repeated` | `source`, `earlier`, `later`, `raise` | a wall time that exists twice keeps the source offset, takes the earlier or the later instant, or raises `ValueError` |
| `month_end` | `clamp`, `raise` | a day missing from the target month takes the month's last day, or raises `ValueError` |

```python
from datetime import datetime

from mloda.provider import TimeRule
from mloda.user import Feature, Options

rule = {"calendar_units": "fixed_length"}
feature = Feature("value__sum_1_day_window", Options(group={"time_rule": rule}))

assert TimeRule(**rule).shift(datetime(2023, 3, 11, 12), 1, "day") == datetime(2023, 3, 12, 12)
```

The option must be a group option (a `time_rule` in the context is refused at plan time). It applies to the feature
that carries it and is not forwarded to its input features. Spelling out a default makes a different
feature identity than leaving it out.

## Testing against the rule

`mloda.testing.time_rule` holds the pinned cases. `assert_time_rule` runs a backend's own shift through
them, and a `TimeWindowRuleSuite` host runs every window case and window function on a framework.

```py
from mloda.testing.framework_adapter import PandasAdapter
from mloda.testing.compute_framework.time_window_suite import TimeWindowRuleSuite
from mloda.testing.time_rule import assert_time_rule


def test_my_shift() -> None:
    assert_time_rule(shift=my_backend_shift)


class TestMyTimeWindowRule(TimeWindowRuleSuite):
    adapter_class = PandasAdapter
    implementation = MyTimeWindow
```

## Compute frameworks

A transfer into or out of a compute framework keeps the instant and the zone. A framework does one of:

- hold the zone in its native column type;
- store instants only, carry the zone beside the column, give its Implementations the wall clock of
  that zone and re-attach the zone on the way out;
- reject the column with a `ValueError` that names the column and the zone.

Handing the same instants back under another zone is not one of them. Every calendar operation then
gives a valid but different answer that nothing downstream can detect, so a
[divergence record](divergence-records.md) does not cover it.

`DataTypeValidatorSuite` checks this with a DST zone and a sub-hour offset.

Spark accepts timezone-aware Arrow columns only in `UTC`, with a `UTC` Spark session. It checks the
session again before export and raises a `ValueError` naming the column and its zone if either
restriction is violated. Native `TimestampNTZType` columns do not require a UTC session.

## Core feature groups

| Group | Application |
|-------|-------------|
| `TimeWindowFeatureGroup` | A row at time `t` aggregates the rows in `(t - span, t]`. Membership is decided on instants (rule 1); `t - span` follows rule 3, so a `1_day` window starts at the same wall time one day earlier and a `1_month` window on the same day of the previous month. |
| `ForecastingFeatureGroup` | Horizon steps follow rule 3, and the time features of each step are read on the wall clock of the reference column's zone (rule 2). |

Both groups take the `time_rule` option (see [Choosing another rule for a feature](#choosing-another-rule-for-a-feature)).

An Implementation that cannot follow the rule declares a [divergence record](divergence-records.md).
