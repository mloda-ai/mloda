"""FormatFeatureGroup loaders, neutral fallback and the describe_columns/count_rows tools."""

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    ToyDeclaredFG,
    ToyFormatFG,
    column_values,
    neutral_csv_group,
    toy_dac,
)

SENTINEL = [{"toyfmt_ld": 111}, {"toyfmt_ld": 222}]


def _run(group: type, frameworks: Any = None, column: str = "toyfmt_ld") -> list[Any]:
    return mloda.run_all(
        [column],
        compute_frameworks=frameworks or [PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({group}),
        data_access_collection=toy_dac(h1={column: [1, 2]}),
    )


def _loader(match: SourceMatch, features: Any) -> Any:
    return SENTINEL


class _BlindDeclaredFG(ToyDeclaredFG):
    @classmethod
    def columns(cls, match: SourceMatch) -> None:
        return None


class TestRegisterLoader:
    def test_loader_result_is_the_data(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        group.register_loader(PythonDictFramework, _loader)

        assert column_values(_run(group), "toyfmt_ld") == [111, 222]

    def test_loader_only_applies_to_its_framework(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        group.register_loader(PyArrowTable, _loader)

        assert column_values(_run(group), "toyfmt_ld") == [1, 2]

    def test_second_registration_for_one_framework_on_one_class_raises(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        group.register_loader(PythonDictFramework, _loader)

        with pytest.raises(ValueError, match="PythonDictFramework"):
            group.register_loader(PythonDictFramework, _loader)

    def test_same_framework_on_a_subclass_is_allowed_and_wins(self, tmp_path: Path) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        parent.register_loader(PythonDictFramework, lambda m, f: [{"toyfmt_ld": 1}])

        class _Child(parent):  # type: ignore[valid-type, misc]
            pass

        _Child.register_loader(PythonDictFramework, _loader)

        assert column_values(_run(_Child), "toyfmt_ld") == [111, 222]

    def test_inherited_loader_is_found_through_the_mro(self, tmp_path: Path) -> None:
        parent = neutral_csv_group(tmp_path / "n.csv")
        parent.register_loader(PythonDictFramework, _loader)

        class _Child(parent):  # type: ignore[valid-type, misc]
            pass

        assert column_values(_run(_Child), "toyfmt_ld") == [111, 222]

    def test_loader_of_an_unavailable_framework_is_not_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        group.register_loader(PythonDictFramework, _loader)
        session = mloda.prepare(
            ["toyfmt_ld"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({group}),
            data_access_collection=toy_dac(h1={"toyfmt_ld": [1, 2]}),
        )
        monkeypatch.setattr(PythonDictFramework, "is_available", staticmethod(lambda: False))

        assert column_values(session.run(), "toyfmt_ld") == [1, 2]


class TestFrameworkSubclassLoader:
    def test_loader_registered_for_a_framework_serves_its_subclass(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        group.register_loader(PythonDictFramework, _loader)

        class _SubDictFramework(PythonDictFramework):
            pass

        assert group._loader_for(_SubDictFramework) is _loader


class TestNeutralFallback:
    def test_neutral_form_is_converted_by_the_existing_transformer(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")

        assert column_values(_run(group), "toyfmt_ld") == [1, 2]


class TestTools:
    def _step_group_and_match(self, group: type, feature: str) -> tuple[Any, Any]:
        requested = Feature(feature)
        resolve_or_raise(
            requested,
            {group: {PythonDictFramework}},
            None,
            toy_dac(h1={"toyfmt_a": [1], "toyfmt_b": [2]}),
        )
        assert requested.input_data_match is not None
        return group, requested.input_data_match[1]

    def test_describe_columns_defaults_to_columns_with_none_types(self) -> None:
        fg, match = self._step_group_and_match(ToyFormatFG, "toyfmt_a")

        assert fg.describe_columns(match) == {"toyfmt_a": None, "toyfmt_b": None}

    def test_describe_columns_raises_when_columns_cannot_enumerate(self) -> None:
        fg, match = self._step_group_and_match(_BlindDeclaredFG, "toyfmt_declared_col")

        with pytest.raises(NotImplementedError):
            fg.describe_columns(match)

    def test_count_rows_defaults_to_none(self) -> None:
        fg, match = self._step_group_and_match(ToyFormatFG, "toyfmt_a")

        assert fg.count_rows(match, PythonDictFramework) is None
