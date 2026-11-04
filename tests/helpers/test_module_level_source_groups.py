"""Unit tests for the module-level source group guard in tests/conftest.py (#2598)."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
from tests.helpers.module_level_source_groups import module_level_source_group_offenders

_FAKE_MODULE = "fake_collected_test_module"


def _fake_source_group(name: str, **overrides: Any) -> type:
    namespace: dict[str, Any] = {"__module__": _FAKE_MODULE, **overrides}
    return type(name, (ApiDataFG,), namespace)


def test_an_unnarrowed_module_level_source_group_is_an_offender() -> None:
    unnarrowed = _fake_source_group("UnnarrowedFakeFG")

    assert module_level_source_group_offenders({_FAKE_MODULE}) == [f"{_FAKE_MODULE}::{unnarrowed.__name__}"]


def test_a_narrowed_module_level_source_group_is_clean() -> None:
    def find_sources(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return []

    narrowed = _fake_source_group("NarrowedFakeFG", find_sources=classmethod(find_sources))

    assert f"{_FAKE_MODULE}::{narrowed.__name__}" not in module_level_source_group_offenders({_FAKE_MODULE})


def test_a_group_in_an_uncollected_module_is_ignored() -> None:
    _fake_source_group("ElsewhereFakeFG")

    assert module_level_source_group_offenders({"some_other_module"}) == []


def test_narrowing_inherited_from_a_test_module_parent_counts() -> None:
    def find_sources(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return []

    parent = _fake_source_group("NarrowingParentFakeFG", find_sources=classmethod(find_sources))
    child = type("InheritingFakeFG", (parent,), {"__module__": _FAKE_MODULE})

    offenders = module_level_source_group_offenders({_FAKE_MODULE})

    assert f"{_FAKE_MODULE}::{child.__name__}" not in offenders
    assert f"{_FAKE_MODULE}::{parent.__name__}" not in offenders
