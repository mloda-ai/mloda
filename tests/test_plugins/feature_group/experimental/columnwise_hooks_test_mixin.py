"""
Shared test mixins for the three column-wise framework hooks.

The hooks (_get_available_columns, _check_source_features_exist, _add_result_to_data) are
declared on FeatureChainParserMixin as raising defaults, not as abstract methods, so nothing
in the type system forces a concrete plugin to implement them. These mixins are that guarantee.

Each family-specific test class inherits ColumnwiseHooksTestMixin and provides:
- plugin_class fixture: the concrete feature group class under test
- sample_data fixture: a two-column container of the family's compute framework
- strict fixture: True when the family raises if ANY source name is missing, False when it
  raises only if NONE of the names exist

Families that resolve column names against the data inherit ColumnDiscoveryHooksTestMixin
instead, which extends ColumnwiseHooksTestMixin with the discovery-hook tests.

A non-pandas family overrides column_names so the shared tests can read its container.
A family whose _add_result_to_data renames or expands the result column overrides
result_feature_name, make_result, and expected_result_columns.
"""

import inspect
from abc import abstractmethod
from typing import Any

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin

CHECK_HOOK = "_check_source_features_exist"
ADD_HOOK = "_add_result_to_data"
DISCOVERY_HOOK = "_get_available_columns"


def resolved_hook(owner: type[Any], hook_name: str) -> Any:
    """Return the plain function behind a hook attribute, or None when the owner has no such hook."""
    attribute = inspect.getattr_static(owner, hook_name, None)
    return getattr(attribute, "__func__", attribute)


def assert_hook_is_implemented(plugin_class: type[Any], hook_name: str) -> None:
    """Assert the plugin resolves the hook to its own function instead of the raising default."""
    plugin_hook = resolved_hook(plugin_class, hook_name)
    assert plugin_hook is not None, f"{plugin_class.__name__} provides no {hook_name}"
    default_hook = resolved_hook(FeatureChainParserMixin, hook_name)
    assert plugin_hook is not default_hook, (
        f"{plugin_class.__name__} inherits the raising {hook_name} default instead of implementing it"
    )


class ColumnwiseHooksTestMixin:
    """Shared tests for the check/add hook pair every column-wise feature group must implement."""

    # Overridable knobs for families whose _add_result_to_data renames or expands the result column.
    result_feature_name: str = "hook_result"

    @pytest.fixture
    @abstractmethod
    def plugin_class(self) -> Any:
        """Return the concrete feature group class under test.

        Override in the family-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def sample_data(self) -> Any:
        """Return a two-column container of the family's compute framework.

        Override in the family-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def strict(self) -> bool:
        """Return True when the family raises on ANY missing name, False when only on all missing.

        Override in the family-specific test class, or supply it from a shared strictness mapping.
        """
        raise NotImplementedError

    def column_names(self, data: Any) -> list[str]:
        """Return the column names of the data as strings (pandas-shaped by default)."""
        return [str(column) for column in data.columns]

    def make_result(self, sample_data: Any) -> Any:
        """Return a result value _add_result_to_data accepts (a row-aligned Series by default)."""
        return pd.Series(range(len(sample_data)))

    def expected_result_columns(self) -> set[str]:
        """Return the columns _add_result_to_data must add (the feature name by default)."""
        return {self.result_feature_name}

    def test_plugin_implements_check_source_features_exist(self, plugin_class: Any) -> None:
        """The family must own the source-feature check, not inherit the raising default."""
        assert_hook_is_implemented(plugin_class, CHECK_HOOK)

    def test_plugin_implements_add_result_to_data(self, plugin_class: Any) -> None:
        """The family must own the result writer, not inherit the raising default."""
        assert_hook_is_implemented(plugin_class, ADD_HOOK)

    def test_check_accepts_existing_feature_names(self, plugin_class: Any, sample_data: Any) -> None:
        """Names that all exist in the data pass the check in every family."""
        plugin_class._check_source_features_exist(sample_data, self.column_names(sample_data))

    def test_partial_presence_follows_family_strictness(
        self, plugin_class: Any, sample_data: Any, strict: bool
    ) -> None:
        """A strict family rejects a partially present name set, a tolerant family accepts it."""
        names = [self.column_names(sample_data)[0], "nonexistent"]
        if strict:
            with pytest.raises(ValueError):
                plugin_class._check_source_features_exist(sample_data, names)
            return
        plugin_class._check_source_features_exist(sample_data, names)

    def test_check_raises_when_no_feature_exists(self, plugin_class: Any, sample_data: Any) -> None:
        """Every family raises when no name exists, and names the available columns for debuggability."""
        with pytest.raises(ValueError) as exc_info:
            plugin_class._check_source_features_exist(sample_data, ["nonexistent"])
        message = str(exc_info.value)
        for column in self.column_names(sample_data):
            assert column in message, f"{plugin_class.__name__} error omits available column '{column}': {message}"

    def test_add_result_to_data_returns_data_with_new_column(self, plugin_class: Any, sample_data: Any) -> None:
        """The add hook returns data carrying the new column(s)."""
        updated = plugin_class._add_result_to_data(sample_data, self.result_feature_name, self.make_result(sample_data))
        missing = sorted(self.expected_result_columns() - set(self.column_names(updated)))
        assert missing == [], f"{plugin_class.__name__}._add_result_to_data did not add {missing}"


class ColumnDiscoveryHooksTestMixin(ColumnwiseHooksTestMixin):
    """Adds the discovery-hook tests for the families that resolve column names against the data."""

    def test_plugin_implements_get_available_columns(self, plugin_class: Any) -> None:
        """The family must own the discovery hook, not inherit the raising default."""
        assert_hook_is_implemented(plugin_class, DISCOVERY_HOOK)

    def test_get_available_columns_returns_column_names(self, plugin_class: Any, sample_data: Any) -> None:
        """The discovery hook reports exactly the column names of the data."""
        assert plugin_class._get_available_columns(sample_data) == set(self.column_names(sample_data))
