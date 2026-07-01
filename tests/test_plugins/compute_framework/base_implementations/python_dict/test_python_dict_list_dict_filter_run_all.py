"""Red-phase regression test for Defect 2 (P1): a PythonDict FeatureGroup that returns
row-wise ``list[dict]`` output AND has a final/global filter applied must work.

``list[dict]`` is still an accepted ``transform`` input (it pivots to columnar). But in
``ComputeFramework.run_calculation`` the final filter runs BEFORE ``transform``:

    data = run_calculate_feature(...)      # -> list[dict] (pre-transform, row-wise)
    data = run_final_filter(data, ...)      # <-- COLUMNAR filter engine hits row data here
    if not isinstance(data, expected_data_framework()): data = transform(data)

So ``run_final_filter`` -> ``_validate_filter_columns`` calls ``_extract_column_names`` on the
``list[dict]``, which returns an EMPTY set (it only understands columnar dicts). The filter
column therefore appears "missing" and the framework raises a spurious
``... output is missing filter column '<col>' ...`` error, even though the row data plainly
contains that column.

Desired behavior: the FG output is normalized to columnar BEFORE filtering, the filter is
applied, and the returned columnar result contains only the matching rows.

Two tests pin this:

1. ``test_list_dict_output_with_filter_run_all`` drives it end-to-end through the public API.
2. ``test_validate_and_apply_filters_on_list_dict_does_not_raise`` is a focused reproducer of
   the exact code path (``_validate_filter_columns`` + ``filter_engine.apply_filters``) on a
   ``list[dict]`` input, so the crash is pinned even independent of the API wiring.

Both are expected to FAIL against the current implementation with the "missing filter column"
error raised out of ``_validate_filter_columns``.
"""

from typing import Any, Optional

from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.provider import BaseInputData
from mloda.user import Feature, Features, GlobalFilter, ParallelizationMode, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_tooling import MlodaTestRunner


class ListDictRootFeatureGroup(FeatureGroup):
    """Root PythonDict FeatureGroup that returns ROW-WISE ``list[dict]`` output.

    ``calculate_feature`` returns ``[{"x": .., "y": ..}, ...]``, a still-accepted transform
    input. A filter on ``x`` must survive the pre-transform final-filter path.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"x", "y"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return [
            {"x": 1, "y": 10},
            {"x": 2, "y": 20},
            {"x": 3, "y": 30},
        ]


_ENABLED_LIST_DICT = PluginCollector.enabled_feature_groups({ListDictRootFeatureGroup})


def test_list_dict_output_with_filter_run_all(flight_server: Any) -> None:
    """End-to-end: list[dict] FG output + min filter x>=2 -> filtered columnar result.

    Expected failure reason (Red): ``run_final_filter`` runs before ``transform`` on the
    ``list[dict]`` output. ``_validate_filter_columns`` -> ``_extract_column_names(list)``
    returns ``set()``, so the filter column ``x`` is reported "missing" and the run raises
    ``... output is missing filter column 'x' ...`` instead of returning
    ``{"x": [2, 3], "y": [20, 30]}``.
    """
    features = Features([Feature(name="x"), Feature(name="y")])

    global_filter = GlobalFilter()
    global_filter.add_filter("x", "min", {"min": 2})

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks={PythonDictFramework},
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
        global_filter=global_filter,
        plugin_collector=_ENABLED_LIST_DICT,
    )

    assert len(result.results) == 1
    assert result.results[0] == {"x": [2, 3], "y": [20, 30]}


def test_validate_and_apply_filters_on_list_dict_does_not_raise() -> None:
    """Focused reproducer: the framework filter path on a ``list[dict]`` input must not crash.

    Reproduces exactly what ``run_final_filter`` does before ``transform``:
    ``_validate_filter_columns(data, features, fg)`` followed by
    ``filter_engine.apply_filters(data, features)``, with ``data`` a row-wise ``list[dict]``.

    Expected failure reason (Red): ``_validate_filter_columns`` calls
    ``_extract_column_names`` on the list, gets ``set()``, and raises
    ``... is missing filter column 'x' ...``. Desired: no raise, and the filter yields the
    columnar rows with x >= 2.
    """
    from mloda.core.filter.single_filter import SingleFilter

    fw = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    data: list[dict[str, Any]] = [
        {"x": 1, "y": 10},
        {"x": 2, "y": 20},
        {"x": 3, "y": 30},
    ]

    single_filter = SingleFilter("x", "min", {"min": 2})

    class _FeaturesStub:
        filter_engine = PythonDictFramework.filter_engine()
        filters = {single_filter}

        @staticmethod
        def get_all_names() -> set[str]:
            return {"x", "y"}

    features = _FeaturesStub()

    # This is the exact pre-transform sequence in run_final_filter. It must NOT raise.
    fw._validate_filter_columns(data, features, ListDictRootFeatureGroup)

    filtered = features.filter_engine().apply_filters(data, features)

    # After normalization + filtering, the columnar result keeps only x >= 2.
    assert filtered == {"x": [2, 3], "y": [20, 30]}
