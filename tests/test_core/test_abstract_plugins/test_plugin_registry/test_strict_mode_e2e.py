"""End-to-end strict mode run through the real ``mloda.run_all`` path.

A strict-mode run must succeed when everything it needs is registered: the
PluginLoader funnel registers the loaded compute framework plugins, and the
locally defined root FeatureGroup is registered explicitly. Mirrors the minimal
DataCreator recipe of tests/test_core/test_api/test_capability_run_all.py on
PythonDictFramework only, so the test never skips and stays fast.

Parallel-safety: assertions are membership-based and the feature name is unique
to this module; the autouse conftest fixture restores the default registry.
"""

from typing import Any, Optional

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, PluginLoader, mloda, register_plugin

FEAT = "strict_e2e_run_all_feature_unique_xyz"


class StrictE2ERunAllFeatureGroup(FeatureGroup):
    """Root feature group producing one columnar feature for the strict e2e run."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({FEAT})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {FEAT: [1, 2, 3]}


def _column_values(frame: Any) -> list[Any]:
    """Extract the FEAT column tolerant of columnar dict or list-of-row-dicts results."""
    if isinstance(frame, dict):
        return list(frame[FEAT])
    return [row[FEAT] for row in frame]


class TestStrictModeEndToEnd:
    def test_strict_run_all_succeeds_with_registered_plugins(self) -> None:
        # Non-regression guard for strict e2e.
        loader = PluginLoader()
        loader.load_matching("compute_framework", "*python_dict*")
        register_plugin(StrictE2ERunAllFeatureGroup)

        collector = PluginCollector().set_strict_mode("strict")
        result = mloda.run_all(
            [Feature(FEAT)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=collector,
        )

        assert isinstance(result, list)
        assert len(result) == 1, f"Expected exactly one result frame, got: {result}"
        assert _column_values(result[0]) == [1, 2, 3]
