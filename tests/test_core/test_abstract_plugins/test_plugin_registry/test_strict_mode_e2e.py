"""End-to-end strict mode run through the real ``mloda.run_all`` path.

A strict-mode run must succeed when everything it needs is registered: the
PluginLoader funnel registers the loaded compute framework plugins, and the
locally defined root FeatureGroup is registered explicitly. Mirrors the minimal
DataCreatorFeatureGroup recipe of tests/test_core/test_api/test_capability_run_all.py on
PythonDictFramework only, so the test never skips and stays fast.

Parallel-safety: assertions are membership-based and the feature name is unique
to this module; the autouse conftest fixture restores the default registry.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import PluginPolicy, PluginPolicyViolationError
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.provider import FeatureSet, DataCreatorFeatureGroup
from mloda.user import Feature, Options, PluginCollector, PluginLoader, mloda, register_plugin
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG

FEAT = "strict_e2e_run_all_feature_unique_xyz"


class StrictE2ERunAllFeatureGroup(DataCreatorFeatureGroup):
    """Root feature group producing one columnar feature for the strict e2e run."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {FEAT}

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


GOV_COLUMN = "gov_forged_csv_column"
FORGED_NAME = "ForgedCsvPythonDict"
ALLOWLIST = PluginPolicy(allowed_module_prefixes=("mloda", "mloda_plugins"))


class TestForgedImplementationIsGoverned:
    """An Implementation on (CsvFG, PythonDictFramework), a pair with no stock implementation, is governed."""

    @staticmethod
    def _forge(make: Callable[..., type[Implementation[Any]]], calls: list[int]) -> type[Implementation[Any]]:
        def forge(group: Any, match: Any, features: Any) -> Any:
            calls.append(1)
            return {GOV_COLUMN: ["FORGED"]}

        return make(CsvFG, PythonDictFramework, forge, name=FORGED_NAME)

    def _run(self, tmp_path: Path, collector: PluginCollector) -> list[Any]:
        path = tmp_path / "gov.csv"
        path.write_text(f"{GOV_COLUMN}\n1\n2\n")
        PluginLoader().load_matching("compute_framework", "*python_dict*")
        PluginLoader().load_matching("feature_group", "*csv_fg*")
        result: list[Any] = mloda.run_all(
            [Feature(GOV_COLUMN, Options({"CsvFG": str(path)}))],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=collector,
        )
        return list(next(iter(result[0].values())))

    def test_strict_mode_with_an_allowlist_policy_serves_the_real_values(
        self,
        tmp_path: Path,
        make_implementation: Callable[..., type[Implementation[Any]]],
        isolated_plugin_registry: PluginRegistry,
    ) -> None:
        isolated_plugin_registry.set_policy(ALLOWLIST)
        calls: list[int] = []
        forged = self._forge(make_implementation, calls)
        assert forged in loaded_implementations()
        with pytest.raises(PluginPolicyViolationError):
            register_plugin(forged)
        assert not isolated_plugin_registry.is_registered(forged)

        assert self._run(tmp_path, PluginCollector().set_strict_mode("strict")) == [1, 2]
        assert calls == []

    def test_a_collector_disabled_implementation_is_not_used(
        self, tmp_path: Path, make_implementation: Callable[..., type[Implementation[Any]]]
    ) -> None:
        calls: list[int] = []
        collector = PluginCollector()
        collector.add_disabled_implementation_classes({self._forge(make_implementation, calls)})

        assert self._run(tmp_path, collector) == [1, 2]
        assert calls == []

    def test_warn_mode_uses_it_and_warns_once_naming_the_class(
        self,
        tmp_path: Path,
        make_implementation: Callable[..., type[Implementation[Any]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        forged = self._forge(make_implementation, [])
        assert forged in loaded_implementations()

        assert self._run(tmp_path, PluginCollector().set_strict_mode("warn")) == ["FORGED"]
        assert len([r for r in caplog.records if FORGED_NAME in r.getMessage()]) == 1
