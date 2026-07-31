"""The column-wise hook contract is machine-readable through get_feature_group_docs().

A downstream author reads the declaration off the catalog instead of the source: which hooks a
feature group needs, and which of them still resolve to the raising default on
FeatureChainParserMixin. Both fields are sorted lists and default to empty, so a parse-only feature
group reports nothing.

Both reads degrade per field like every other read in ``get_feature_group_docs``: a class whose
attribute lookup raises is still documented, with empty lists.

Test doubles are function-local and reaped in a ``finally`` block, because plugin docs walk the live
``__subclasses__()`` registry.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    COLUMN_DISCOVERY_HOOKS,
    COLUMNWISE_HOOKS,
)
from mloda.provider import ComputeFramework, FeatureChainParserMixin, FeatureGroup
from mloda.steward import FeatureGroupInfo, get_feature_group_docs
from mloda.user import PluginLoader
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

ADD_HOOK = "_add_result_to_data"

# A name no hook has, so the catalog must never carry it.
UNKNOWN_HOOK = "_not_a_columnwise_hook_docs"


class _RaisingDescriptor:
    """A REQUIRED_COLUMNWISE_HOOKS whose read raises, the broken-declaration double the degradation tests share."""

    def __get__(self, obj: Any, owner: type) -> Any:
        raise RuntimeError("attribute lookup fails")


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


@pytest.fixture(autouse=True)
def _reap_pending_dead_plugin_classes() -> None:
    """Collect dead test-local plugin classes before each test, so enumeration is stable."""
    gc.collect()


def _doc_for(name: str) -> FeatureGroupInfo:
    """Fetch the single FeatureGroupInfo whose name matches exactly."""
    exact = [doc for doc in get_feature_group_docs(name=name) if doc.name == name]
    assert len(exact) == 1, f"expected exactly one doc for {name}, got {[doc.name for doc in exact]}"
    return exact[0]


class TestFeatureGroupInfoColumnwiseFields:
    """FeatureGroupInfo gains two defaulted, independently owned list fields."""

    def test_positional_construction_defaults_the_new_fields(self) -> None:
        info = FeatureGroupInfo("n", "d", "v", "m", [], set(), "n_")
        assert info.required_columnwise_hooks == []
        assert info.missing_columnwise_hooks == []

    def test_keyword_construction_accepts_the_new_fields(self) -> None:
        info = FeatureGroupInfo(
            "n",
            "d",
            "v",
            "m",
            [],
            set(),
            "n_",
            required_columnwise_hooks=["_add_result_to_data"],
            missing_columnwise_hooks=["_add_result_to_data"],
        )
        assert info.required_columnwise_hooks == ["_add_result_to_data"]
        assert info.missing_columnwise_hooks == ["_add_result_to_data"]

    def test_mutable_defaults_are_not_shared(self) -> None:
        first = FeatureGroupInfo("n1", "d", "v", "m", [], set(), "n1_")
        second = FeatureGroupInfo("n2", "d", "v", "m", [], set(), "n2_")
        first.required_columnwise_hooks.append("x")
        first.missing_columnwise_hooks.append("x")
        assert second.required_columnwise_hooks == []
        assert second.missing_columnwise_hooks == []


class TestColumnwiseHookDocsReporting:
    """The catalog reports the declared requirement and what is still unimplemented."""

    def test_complete_family_reports_required_hooks_and_nothing_missing(self) -> None:
        """A shipped framework implementation carries its family's requirement and implements all of it."""
        doc = _doc_for(PandasAggregatedFeatureGroup.get_class_name())
        assert doc.required_columnwise_hooks == sorted(COLUMN_DISCOVERY_HOOKS)
        assert doc.missing_columnwise_hooks == []

    def test_feature_group_without_a_requirement_reports_empty_lists(self) -> None:
        """A parse-only feature group declares no hook, so both lists stay empty."""

        class _DocsNoColumnwiseFG(FeatureGroup):
            """Test double that needs no column-wise hook."""

        try:
            doc = _doc_for("_DocsNoColumnwiseFG")
            assert doc.required_columnwise_hooks == []
            assert doc.missing_columnwise_hooks == []
        finally:
            del _DocsNoColumnwiseFG
            gc.collect()

    def test_incomplete_class_reports_its_missing_hooks(self) -> None:
        """A class that declares the requirement and implements none of it reports every hook as missing."""

        class _DocsIncompleteColumnwiseFG(FeatureChainParserMixin, FeatureGroup):
            """Test double declaring the write-hook pair without implementing it."""

            REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

        try:
            doc = _doc_for("_DocsIncompleteColumnwiseFG")
            assert doc.required_columnwise_hooks == sorted(COLUMNWISE_HOOKS)
            assert doc.missing_columnwise_hooks == sorted(COLUMNWISE_HOOKS)
        finally:
            del _DocsIncompleteColumnwiseFG
            gc.collect()

    def test_partially_implemented_class_reports_only_the_unimplemented_hook(self) -> None:
        """An implemented hook drops out of the missing list while the requirement stays whole."""

        class _DocsPartialColumnwiseFG(FeatureChainParserMixin, FeatureGroup):
            """Test double implementing the writer but not the check."""

            REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

            @classmethod
            def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
                return data

        try:
            doc = _doc_for("_DocsPartialColumnwiseFG")
            assert doc.required_columnwise_hooks == sorted(COLUMNWISE_HOOKS)
            assert doc.missing_columnwise_hooks == ["_check_source_features_exist"]
        finally:
            del _DocsPartialColumnwiseFG
            gc.collect()

    def test_class_owning_no_hook_at_all_reports_them_all_missing(self) -> None:
        """A plain FeatureGroup declaring the requirement inherits no hook, so it owes all of them.

        Reporting nothing missing here would have the catalog claim the class owes nothing while it
        owns nothing: the attribute is simply ABSENT, which is not the same as implemented.
        """

        class _DocsHooklessColumnwiseFG(FeatureGroup):
            """Test double declaring the write-hook pair without inheriting the mixin that declares them."""

            REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

        try:
            doc = _doc_for("_DocsHooklessColumnwiseFG")
            assert doc.required_columnwise_hooks == sorted(COLUMNWISE_HOOKS)
            assert doc.missing_columnwise_hooks == sorted(COLUMNWISE_HOOKS)
        finally:
            del _DocsHooklessColumnwiseFG
            gc.collect()


class TestOnlyRealHookNamesReachTheCatalog:
    """A malformed declaration must not put names that are not hooks into a machine-readable field."""

    def test_string_declaration_reports_no_characters(self) -> None:
        """A string is iterable one character at a time, and a character is not a column-wise hook."""

        class _DocsStringColumnwiseFG(FeatureChainParserMixin, FeatureGroup):
            """Test double whose declaration lost its braces."""

            REQUIRED_COLUMNWISE_HOOKS = ADD_HOOK  # type: ignore[assignment]

        try:
            doc = _doc_for("_DocsStringColumnwiseFG")
            assert set(doc.required_columnwise_hooks) <= COLUMN_DISCOVERY_HOOKS, doc.required_columnwise_hooks
            assert set(doc.missing_columnwise_hooks) <= COLUMN_DISCOVERY_HOOKS, doc.missing_columnwise_hooks
        finally:
            del _DocsStringColumnwiseFG
            gc.collect()

    def test_unknown_name_is_dropped_from_both_fields(self) -> None:
        """The real hook of a half-wrong declaration survives; the unrecognized name never reaches the catalog."""

        class _DocsUnknownColumnwiseFG(FeatureChainParserMixin, FeatureGroup):
            """Test double declaring one real hook and one name that is not a hook."""

            REQUIRED_COLUMNWISE_HOOKS = frozenset({ADD_HOOK, UNKNOWN_HOOK})

        try:
            doc = _doc_for("_DocsUnknownColumnwiseFG")
            assert doc.required_columnwise_hooks == [ADD_HOOK]
            assert doc.missing_columnwise_hooks == [ADD_HOOK]
        finally:
            del _DocsUnknownColumnwiseFG
            gc.collect()


class TestColumnwiseHookDocsDegradation:
    """A broken declaration degrades to empty lists instead of sinking the catalog call."""

    def test_raising_attribute_read_degrades_to_empty_lists(self) -> None:
        """A REQUIRED_COLUMNWISE_HOOKS whose lookup raises leaves the class documented with empty lists.

        The double inherits the mixin AND pins compute_framework_rule, so ``__init_subclass__`` runs the
        class-definition guard over the same broken declaration the catalog reads. A plain FeatureGroup
        double would document the degradation while leaving the definition-time path, the one an author
        actually hits, untested.
        """

        class _DocsColumnwiseBoomFG(FeatureChainParserMixin, FeatureGroup):
            """Test double whose REQUIRED_COLUMNWISE_HOOKS lookup raises."""

            REQUIRED_COLUMNWISE_HOOKS = _RaisingDescriptor()

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        try:
            doc = _doc_for("_DocsColumnwiseBoomFG")
            assert doc.required_columnwise_hooks == []
            assert doc.missing_columnwise_hooks == []
        finally:
            del _DocsColumnwiseBoomFG
            gc.collect()

    def test_non_iterable_declaration_degrades_to_empty_lists(self) -> None:
        """REQUIRED_COLUMNWISE_HOOKS = 5 is author error, but the class is still defined and still documented."""

        class _DocsColumnwiseNonIterableFG(FeatureChainParserMixin, FeatureGroup):
            """Test double whose REQUIRED_COLUMNWISE_HOOKS is not a collection of names."""

            REQUIRED_COLUMNWISE_HOOKS = 5  # type: ignore[assignment]

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        try:
            doc = _doc_for("_DocsColumnwiseNonIterableFG")
            assert doc.required_columnwise_hooks == []
            assert doc.missing_columnwise_hooks == []
        finally:
            del _DocsColumnwiseNonIterableFG
            gc.collect()

    def test_broken_declaration_does_not_sink_the_catalog(self) -> None:
        """Every healthy feature group is still listed while the broken double is live."""
        baseline = {doc.name for doc in get_feature_group_docs()}
        assert len(baseline) > 0, "Need a populated baseline catalog"

        class _DocsColumnwiseSinkFG(FeatureGroup):
            """Test double whose REQUIRED_COLUMNWISE_HOOKS lookup raises."""

            REQUIRED_COLUMNWISE_HOOKS = _RaisingDescriptor()

        try:
            degraded = {doc.name for doc in get_feature_group_docs()}
            assert baseline.issubset(degraded), "A broken declaration must not drop healthy feature groups"
            assert "_DocsColumnwiseSinkFG" in degraded
        finally:
            del _DocsColumnwiseSinkFG
            gc.collect()
