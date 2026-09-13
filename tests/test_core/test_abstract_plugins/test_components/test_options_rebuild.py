from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec

GROUP_KEY = "grp_rebuild1408"
CONTEXT_KEY = "ctx_rebuild1408"
DEFAULT_KEY = "default_rebuild1408"
BINDING_KEY = "binding_rebuild1408"

SOURCE_GROUP: dict[str, Any] = {GROUP_KEY: "g"}
SOURCE_CONTEXT: dict[str, Any] = {CONTEXT_KEY: "c"}
REBUILT_GROUP: dict[str, Any] = {"rebuilt_grp_rebuild1408": "rg"}
REBUILT_CONTEXT: dict[str, Any] = {CONTEXT_KEY: "rc"}

BINDING_PATTERN = rf".*__(?P<{BINDING_KEY}>alpha|beta)_rebuild1408$"


def _source_options() -> Options:
    options = Options(
        group=dict(SOURCE_GROUP), context=dict(SOURCE_CONTEXT), propagate_context_keys=frozenset({CONTEXT_KEY})
    )
    for name, value in list(vars(options).items()):
        if isinstance(value, frozenset) and not value:
            setattr(options, name, frozenset({name}))
    return options


def _carried_state(options: Options) -> dict[str, Any]:
    return {name: value for name, value in vars(options).items() if name not in ("group", "context")}


def _rebuild(options: Options) -> Options:
    return options.rebuild(dict(REBUILT_GROUP), dict(REBUILT_CONTEXT))


def _options_with_defaults(options: Options) -> Options:
    class RebuildDefaultsProbeFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = {DEFAULT_KEY: PropertySpec("A context concrete default.", context=True, default="filled")}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return data

    return RebuildDefaultsProbeFeatureGroup.options_with_defaults(options)


def _build_effective_options(options: Options) -> Options:
    mapping = {BINDING_KEY: PropertySpec("A name-bound value.", context=True)}
    return FeatureChainParser.build_effective_options("src__alpha_rebuild1408", [BINDING_PATTERN], mapping, options)


@pytest.mark.parametrize(
    ("produce", "expected_group", "expected_context"),
    [
        pytest.param(_rebuild, REBUILT_GROUP, REBUILT_CONTEXT, id="rebuild"),
        pytest.param(copy.copy, SOURCE_GROUP, SOURCE_CONTEXT, id="copy"),
        pytest.param(copy.deepcopy, SOURCE_GROUP, SOURCE_CONTEXT, id="deepcopy"),
        pytest.param(
            _options_with_defaults, SOURCE_GROUP, {**SOURCE_CONTEXT, DEFAULT_KEY: "filled"}, id="options_with_defaults"
        ),
        pytest.param(
            _build_effective_options,
            SOURCE_GROUP,
            {**SOURCE_CONTEXT, BINDING_KEY: "alpha"},
            id="build_effective_options",
        ),
    ],
)
def test_new_options_carries_all_state_but_group_and_context(
    produce: Callable[[Options], Options], expected_group: dict[str, Any], expected_context: dict[str, Any]
) -> None:
    source = _source_options()
    assert all(_carried_state(source).values()), "a carried attribute is still empty; stamp it in _source_options"

    result = produce(source)

    assert result is not source
    assert result.group == expected_group
    assert result.context == expected_context
    assert _carried_state(result) == _carried_state(source)
