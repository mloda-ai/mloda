"""Catalog.build loads plugins before indexing; describe resolves names through the configured sources."""

import json
from pathlib import Path
from typing import Any

import pytest

from mloda.core.api.plugin_docs import get_feature_group_docs
from mloda.core.api.plugin_info import FeatureGroupInfo
from mloda.interfaces.service.catalog import Catalog
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import is_error
from mloda.user import PluginCollector, PluginLoader
from tests.helpers.docs_scope import small_feature_group_walk
from tests.test_interfaces.format_groups import FILE_FORMAT_GROUPS

# Cold worker under xdist load: PluginLoader.all plus hashing the scoped shipped groups can exceed the default timeout.
pytestmark = pytest.mark.timeout(30)

LOAD = "PluginLoader.all"
ENUMERATE = "get_feature_group_docs"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record PluginLoader.all and the catalog's FeatureGroup enumeration, in call order."""
    recorded: list[str] = []
    original_all = PluginLoader.all
    original_docs = get_feature_group_docs

    def spy_all(cls: type[PluginLoader], force_reload: bool = False) -> PluginLoader:
        recorded.append(LOAD)
        return original_all(force_reload=force_reload)

    def spy_docs(*args: Any, **kwargs: Any) -> list[FeatureGroupInfo]:
        recorded.append(ENUMERATE)
        return original_docs(*args, **kwargs)

    monkeypatch.setattr(PluginLoader, "all", classmethod(spy_all))
    monkeypatch.setattr(f"{Catalog.__module__}.{ENUMERATE}", spy_docs)
    return recorded


def _settings(tmp_path: Path) -> Settings:
    path = tmp_path / "sales.csv"
    path.write_text("customer_id,income,segment\n1,10,a\n2,20,b\n")
    return Settings(sources={"sales": SourceConfig(alias="sales", files=(str(path),))})


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    return Catalog.build(_settings(tmp_path), PluginCollector.enabled_feature_groups(set(FILE_FORMAT_GROUPS)))


class TestBuildLoadsPlugins:
    def test_plugins_are_loaded_before_feature_groups_are_enumerated(
        self, calls: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        walk = small_feature_group_walk(*FILE_FORMAT_GROUPS)
        monkeypatch.setattr("mloda.core.api.plugin_docs.get_all_subclasses", walk)
        Catalog.build(Settings())

        assert LOAD in calls, f"Catalog.build never called PluginLoader.all: {calls}"
        assert ENUMERATE in calls, calls
        assert calls.index(LOAD) < calls.index(ENUMERATE), calls


class TestDescribeWithSources:
    def test_column_of_a_configured_source_is_described(self, catalog: Catalog, tmp_path: Path) -> None:
        result = catalog.describe("income")

        assert not is_error(result), result
        assert result["name"] == "income"
        assert result["feature_group"] == "CsvFG"
        dumped = json.dumps(result)
        assert str(tmp_path) not in dumped
        assert tmp_path.name not in dumped
        assert "sales.csv" not in dumped

    def test_misspelled_source_column_suggests_the_column(self, catalog: Catalog, tmp_path: Path) -> None:
        result = catalog.describe("incme")

        assert is_error(result), result
        assert result["error"]["code"] == "validation_failed"
        (detail,) = result["error"]["details"]
        assert detail["path"] == "/name"
        assert detail["problem"] == "unknown_feature"
        assert detail.get("suggestion") == "income", detail
        assert str(tmp_path) not in json.dumps(result)
