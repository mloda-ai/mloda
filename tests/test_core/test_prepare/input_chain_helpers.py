"""Scoped candidates and source probes for input-chain resolution tests (#2320)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from copy import deepcopy
from typing import Any, ClassVar

import pyarrow as pa

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.prepare.resolution_types import EvaluationResult, ResolutionRecord
from mloda.provider import ClaimRoute, ComputeFramework, FeatureGroup, NamePolicy, SourceFeatureGroup, SourceMatch
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.helpers.plugin_stubs import StubFeatureGroup


CHOICE = "choice_chain_2320"
MIDDLE = "middle_chain_2320"
LEAF = "leaf_chain_2320"
MISSING = "missing_chain_2320"
OTHER_MISSING = "other_missing_chain_2320"
RAW_A = "raw_a_chain_2320"
RAW_B = "raw_b_chain_2320"
SOURCE_KEY = "columns_chain_2320"
MATCH_CALLS: Counter[tuple[type[FeatureGroup], str]] = Counter()
PROBE_CALLS: Counter[tuple[type[SourceFeatureGroup], str]] = Counter()
LOAD_CALLS: list[type[SourceFeatureGroup]] = []


class ChainStub2320(StubFeatureGroup):
    """Fresh declared inputs and a count of successful name checks."""

    INPUTS: ClassVar[tuple[Feature | str, ...] | None] = None
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]]] = {PyArrowTable}
    WRITES: ClassVar[dict[str, Any]] = {}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        matched = super().match_feature_group_criteria(feature_name, options, data_access_collection)
        if matched:
            MATCH_CALLS[cls, str(feature_name)] += 1
            for key, value in cls.WRITES.items():
                options.add_to_group(key, value)
        return matched

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if self.INPUTS is None:
            return None
        return {Feature(entry) if isinstance(entry, str) else deepcopy(entry) for entry in self.INPUTS}


class ChainA2320(ChainStub2320):
    MATCHED_NAMES = frozenset({CHOICE})
    INPUTS = (MIDDLE,)


class ChainB2320(ChainStub2320):
    MATCHED_NAMES = frozenset({CHOICE})
    INPUTS = (LEAF,)


class ChainBChild2320(ChainB2320):
    pass


class ChainC2320(ChainStub2320):
    MATCHED_NAMES = frozenset({CHOICE})
    INPUTS = (OTHER_MISSING,)


class ChainMiddle2320(ChainStub2320):
    MATCHED_NAMES = frozenset({MIDDLE})
    INPUTS = (MISSING,)


class ChainLeaf2320(ChainStub2320):
    MATCHED_NAMES = frozenset({LEAF})


class ChainOtherLeaf2320(ChainStub2320):
    MATCHED_NAMES = frozenset({LEAF})
    INPUTS = (MISSING,)


class ChainSourceBase2320(SourceFeatureGroup):
    """A marker-only source that records actual catalog probes, never a global toy reader."""

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        PROBE_CALLS[cls, feature_name] += 1
        pointed = options.get(cls.get_class_name())
        if isinstance(pointed, dict) and SOURCE_KEY in pointed:
            return [SourceMatch(source=f"{cls.__name__}:pointed", access=pointed[SOURCE_KEY])]
        if data_access_collection is None:
            return []
        return [
            SourceMatch(source=f"{handle}:chain2320", access=value[SOURCE_KEY])
            for handle, value in data_access_collection.credentials.items()
            if isinstance(value, dict) and SOURCE_KEY in value
        ]

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        return list(match.access)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        LOAD_CALLS.append(cls)
        return pa.table({name: match.access[name] for name in features.get_all_names()})


class ChainChecked2320(ChainSourceBase2320):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("chain2320", NamePolicy.CHECKED, True),)


class ChainOtherChecked2320(ChainSourceBase2320):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("chain2320", NamePolicy.CHECKED, True),)


class ChainOpen2320(ChainSourceBase2320):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("chain2320", NamePolicy.OPEN, False),)


class ChainGate2320(Extender):
    """Record every match hook and retain the exact refusals for selected reader-column pairs."""

    never_fall_back = True

    def __init__(self, refused: Collection[str] = (), reader: type[FeatureGroup] | None = None) -> None:
        self.priority = 100
        self.name = "chain_gate_2320"
        self.refused = frozenset(refused)
        self.reader = reader
        self.contexts: list[HookContext] = []
        self.errors: list[PermissionError] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        self.contexts.append(context)
        blocked = self.refused.intersection(context.feature_names)
        if blocked and (self.reader is None or self.reader is context.reader_class):
            error = PermissionError(f"refused input {min(blocked)} in chain2320")
            self.errors.append(error)
            raise error
        return func(*args, **kwargs)

    def calls_for(self, reader: type[FeatureGroup], name: str) -> list[HookContext]:
        return [
            context for context in self.contexts if context.reader_class is reader and context.feature_names == (name,)
        ]


def chain_access(**columns: list[int]) -> DataAccessCollection:
    return DataAccessCollection(credentials={"chain_handle_2320": {SOURCE_KEY: columns}})


def prepare_chain(
    groups: Collection[type[FeatureGroup]],
    *,
    feature: Feature | None = None,
    access: DataAccessCollection | None = None,
    extender: Extender | None = None,
    collector: PluginCollector | None = None,
) -> Any:
    return mloda.prepare(
        [feature if feature is not None else Feature(CHOICE)],
        compute_frameworks=[PyArrowTable],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=collector or PluginCollector.enabled_feature_groups(set(groups)),
        data_access_collection=access,
        function_extender={extender} if extender is not None else None,
    )


def result_for(records: list[ResolutionRecord], name: str = CHOICE) -> EvaluationResult:
    matching = [record.result for record in records if record.feature_name == name]
    assert len(matching) == 1
    return matching[0]
