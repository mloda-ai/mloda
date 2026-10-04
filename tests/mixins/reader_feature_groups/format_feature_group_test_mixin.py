"""Shared contract tests for FormatFeatureGroup implementations.

A concrete test class sets the class attributes and implements the two collection builders. Not collected on its
own (no Test prefix); supersedes the cycle 1 foreign-source, no-mutation and class-name-shortcut matching tests.
"""

import copy
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import FormatFeatureGroup, NamePolicy
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class _LoadCapture(Extender):
    def __init__(self) -> None:
        self.priority = 100
        self.contexts: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None:
            self.contexts.append(context)
        return result


class FormatFeatureGroupTestMixin:
    """Contract of every format group; the concrete class supplies the hooks below."""

    feature_group_class: type[FormatFeatureGroup]
    present_column: str
    missing_column: str
    expected_source: str

    def own_dac(self) -> DataAccessCollection:
        """A fresh collection holding a source of the group's own kind with present_column."""
        raise NotImplementedError

    def foreign_dac(self) -> DataAccessCollection:
        """A fresh collection holding only sources of another kind."""
        raise NotImplementedError

    def _plugins(self) -> FeatureGroupEnvironmentMapping:
        return {self.feature_group_class: {PyArrowTable}}

    def _claims(self, feature: Feature, dac: DataAccessCollection | None) -> bool:
        result = IdentifyFeatureGroupClass.evaluate(feature, self._plugins(), None, dac)
        return self.feature_group_class in result.identified

    def test_claims_its_own_source(self) -> None:
        assert self._claims(Feature(self.present_column), self.own_dac())

    def test_declines_foreign_sources(self) -> None:
        assert not self._claims(Feature(self.present_column), self.foreign_dac())

    def test_leaves_collection_and_options_unchanged(self) -> None:
        dac = self.own_dac()
        credentials = copy.deepcopy(dac.credentials)
        files = copy.deepcopy(dac.files)
        feature = Feature(self.present_column)
        group = copy.deepcopy(feature.options.group)
        context = copy.deepcopy(feature.options.context)

        self._claims(feature, dac)

        assert dac.credentials == credentials
        assert dac.files == files
        assert feature.options.group == group
        assert feature.options.context == context
        assert RESERVED_READER_OPTION_KEY not in feature.options.group
        assert RESERVED_READER_OPTION_KEY not in feature.options.context

    def test_declines_a_missing_column_on_checked_routes(self) -> None:
        if not any(r.names is NamePolicy.CHECKED for r in self.feature_group_class.CLAIM_ROUTES):
            pytest.skip("no checked route")
        assert not self._claims(Feature(self.missing_column), self.own_dac())

    def test_no_route_is_open_and_searched(self) -> None:
        assert not [r for r in self.feature_group_class.CLAIM_ROUTES if r.names is NamePolicy.OPEN and r.searched]

    def test_declines_a_feature_named_after_the_group_without_a_source(self) -> None:
        name = self.feature_group_class.get_class_name()
        assert not self._claims(Feature(name), None)
        assert not self._claims(Feature(name), self.foreign_dac())

    def test_fires_input_data_load_naming_group_source_format_and_loader(self) -> None:
        capture = _LoadCapture()

        mloda.run_all(
            [self.present_column],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=self.own_dac(),
            function_extender={capture},
        )

        assert len(capture.contexts) == 1
        context = capture.contexts[0]
        group_name = self.feature_group_class.get_class_name()
        assert context.data_access_format == group_name
        assert context.reader_class is self.feature_group_class
        assert context.data_access_identity == self.expected_source
        assert context.data_access_loader in ("neutral", "PyArrowTable")
