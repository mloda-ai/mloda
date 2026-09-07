"""Pin the filter-criteria effective-options contract.

Feature intake rebinds a resolved feature's options to their materialized (post-default) view
before filter matching runs, so match_feature_group_criteria observes EFFECTIVE options when
called to match a filter feature, even though the very same classmethod observes DECLARED
(pre-default) options when called to resolve a feature group during ordinary feature resolution.
"""

from __future__ import annotations

from typing import Any


from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, PropertySpec
from mloda.user import FeatureName, FilterType, GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


# PROPERTY_MAPPING key/default for the throwaway probe; the pfc_ prefix keeps it unique to this module.
PFC_KEY = "pfc_criteria_key"
PFC_DEFAULT = "pfc_default_val"
PFC_OTHER_VAL = "pfc_other_val"  # explicit non-default value; guards test 1's discriminating power

PFC_MAIN = "pfc_main_feature"  # requested root feature; must always resolve on its declared (pre-default) options
PFC_TARGET = "pfc_target_feature"  # never requested directly; only reachable as a matched filter feature


def _make_probe_fg() -> type[FeatureGroup]:
    """A throwaway root FeatureGroup whose criteria observes PFC_TARGET's PFC_KEY, the option this test pins."""

    class PfcCriteriaProbeFeatureGroup(FeatureGroup):
        main_criteria_observed: list[Any] = []  # class-local recorder: no external ref, so no leak

        PROPERTY_MAPPING = {
            PFC_KEY: PropertySpec("Steers filter-criteria matching.", context=True, default=PFC_DEFAULT),
        }

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({PFC_MAIN, PFC_TARGET})

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            # PFC_MAIN must always resolve: feature resolution itself observes declared (pre-default)
            # options, so gating the root request on PFC_KEY would break it whenever the key is absent.
            # The recorder captures what this branch actually observed, so the declared-options half of
            # the module's asymmetry claim is assertable, not just assumed.
            name = str(feature_name)
            if name == PFC_MAIN:
                cls.main_criteria_observed.append(options.get(PFC_KEY))
                return True
            if name == PFC_TARGET:
                return bool(options.get(PFC_KEY) == PFC_DEFAULT)
            return False

        @classmethod
        def final_filters(cls) -> bool:
            # Read features.filters inline (below) instead: the payload is not filterable data, so the
            # framework's own post-calculation row elimination must not run against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            payload = {
                "names": sorted(str(f.name) for f in features.features),
                "filter_count": len(features.filters) if features.filters else 0,
                "main_criteria_observed": list(cls.main_criteria_observed),
            }
            return {str(feature.name): [payload] for feature in features.features}

    return PfcCriteriaProbeFeatureGroup


def _single_row(frame: Any, column: str) -> Any:
    """Extract the single payload row, tolerant of columnar dict or list-of-row-dicts results."""
    if isinstance(frame, dict):
        values = list(frame[column])
    else:
        values = [row[column] for row in frame]
    assert len(values) == 1, f"expected exactly one row for {column}, got {values!r}"
    return values[0]


def _run(options: Options) -> dict[str, Any]:
    """Run PFC_MAIN under a global EQUAL filter on PFC_TARGET; return PFC_MAIN's payload row.

    The probe class and collector stay locals of THIS frame, so a failing assert in the caller
    never pins the throwaway class in a traceback and the no-leak guard stays green. `fg` and
    `collector` are deleted from THIS frame too, right after `run_all` returns and before the
    asserts below: otherwise a failing assert here would pin them into this frame's own traceback,
    which would trip the no-leak fixture's assertion as well and mask the real failure.
    """
    fg = _make_probe_fg()
    collector = PluginCollector.enabled_feature_groups({fg})
    global_filter = GlobalFilter()
    global_filter.add_filter(PFC_TARGET, FilterType.EQUAL, {"value": 1})
    results = mloda.run_all(
        [Feature(PFC_MAIN, options)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )
    del fg, collector
    assert len(results) == 1, f"expected exactly one result frame, got: {results!r}"
    payload = _single_row(results[0], PFC_MAIN)
    assert isinstance(payload, dict)
    return payload


def test_filter_matches_via_materialized_default_when_request_leaves_key_absent() -> None:
    """An absent PFC_KEY is materialized to its declared default before filter matching runs.

    unify_options then carries that materialized value into the filter's own (also-absent)
    options, so PFC_TARGET's criteria observes the default and the filter attaches.
    """
    payload = _run(Options())
    assert PFC_TARGET in payload["names"], (
        f"the filter must attach PFC_TARGET via the materialized default: {payload!r}"
    )
    assert payload["filter_count"] == 1, f"exactly one filter must match: {payload!r}"
    assert payload["main_criteria_observed"] == [None], (
        f"resolve-time match must observe the declared (pre-default) view, not the materialized default: {payload!r}"
    )


def test_filter_does_not_match_when_explicit_option_differs_from_default() -> None:
    """An explicit non-default PFC_KEY survives materialization unchanged, so the filter must not attach.

    Guards test 1: a criteria override that ignored the option and always returned True would
    pass test 1 but fail here.
    """
    payload = _run(Options(context={PFC_KEY: PFC_OTHER_VAL}))
    assert PFC_TARGET not in payload["names"], f"the filter must not attach on a non-default value: {payload!r}"
    assert payload["filter_count"] == 0, f"no filter may match: {payload!r}"
