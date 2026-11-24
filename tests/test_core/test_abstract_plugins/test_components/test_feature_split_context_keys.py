"""Explicit context split keys survive ownership, copying, and option forwarding."""

import copy
import pickle  # nosec B403
from typing import Any

import pytest

from mloda.core.prepare.input_chain import resolution_key
from mloda.provider import FeatureGroup, FeatureSet, PropertySpec
from mloda.user import Feature, Options


def _feature(options: Options | None = None, keys: Any = (), **kwargs: Any) -> Feature:
    return Feature("split_context_feature", options=options, split_context_keys=keys, **kwargs)


class SplitContextDefaults(FeatureGroup):
    PROPERTY_MAPPING = {"padding": PropertySpec("A default applied after declaration.", default="filled")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data


class TaggedSplitOptions(Options):
    def __init__(self, context: dict[str, Any], tag: object) -> None:
        super().__init__(context=context)
        self.tag = tag


def test_omitting_split_keys_keeps_plain_context_as_metadata() -> None:
    feature = Feature("plain_context_feature", Options(context={"top_k": 2}))

    assert feature.options.split_context_keys == frozenset()


@pytest.mark.parametrize("keys", [set(), frozenset(), [], ()])
def test_empty_split_key_containers_are_valid(keys: Any) -> None:
    assert _feature(keys=keys).options.split_context_keys == frozenset()


@pytest.mark.parametrize("keys", [{"top_k"}, frozenset({"top_k"}), ["top_k", "top_k"], ("top_k",)])
def test_split_key_containers_normalize_without_forging_provenance(keys: Any) -> None:
    feature = _feature(Options(context={"top_k": 2}), keys)

    assert feature.options.split_context_keys == frozenset({"top_k"})
    assert isinstance(feature.options.split_context_keys, frozenset)
    assert feature.options.inherited_context_keys == frozenset()
    assert feature.options.twin_split_keys == frozenset()


@pytest.mark.parametrize("keys", [None, True, 7, "top_k", b"top_k", {"top_k": "renamed"}, [7], ("top_k", None)])
def test_invalid_split_key_containers_are_rejected(keys: Any) -> None:
    with pytest.raises(TypeError, match=r"split_context_keys (must be|elements must be)"):
        _feature(Options(context={"top_k": 2}), keys)


@pytest.mark.parametrize("options", [Options(), Options(group={"top_k": 2}), Options(context={"other": 2})])
def test_split_keys_must_already_exist_in_context(options: Options) -> None:
    with pytest.raises(ValueError, match=r"split_context_keys.*top_k.*context"):
        _feature(options, {"top_k"})


@pytest.mark.parametrize("value", [None, False, 0])
def test_split_key_presence_does_not_require_a_truthy_value(value: Any) -> None:
    feature = _feature(Options(context={"top_k": value}), {"top_k"})

    assert feature.options.context["top_k"] is value
    assert feature.options.split_context_keys == frozenset({"top_k"})


@pytest.mark.parametrize("frozen", [False, True])
def test_marking_owns_options_but_keeps_the_subclass_and_leaf_identity(frozen: bool) -> None:
    leaf: list[int] = [1, 2]
    tag = object()
    source = TaggedSplitOptions({"top_k": 2, "payload": leaf}, tag)
    if frozen:
        source.freeze()

    feature = _feature(source, {"top_k"})

    assert isinstance(feature.options, TaggedSplitOptions)
    assert feature.options is not source
    assert feature.options.context is not source.context
    assert feature.options.context["payload"] is leaf
    assert feature.options.tag is tag
    assert source.split_context_keys == frozenset()
    assert source.context == {"top_k": 2, "payload": [1, 2]}
    assert feature.options.split_context_keys == frozenset({"top_k"})


def test_marking_additional_keys_preserves_existing_explicit_and_inherited_keys() -> None:
    source = Options(context={"top_k": 2, "limit": 4})
    source.inherit_from(Options(context={"tenant": "one"}), inherit_context_keys=frozenset({"tenant"}))
    first = _feature(source, {"top_k"})

    second = _feature(first.options, {"limit"})

    assert second.options.split_context_keys == frozenset({"top_k", "limit", "tenant"})
    assert second.options.inherited_context_keys == frozenset({"tenant"})
    assert second.options.twin_split_keys == frozenset()
    assert first.options.split_context_keys == frozenset({"top_k", "tenant"})
    assert source.split_context_keys == frozenset({"tenant"})


@pytest.mark.parametrize(
    "operation",
    [
        "copy",
        "deepcopy",
        "rebuild",
        "fork",
        "adopt",
        "defaults",
        "pickle",
        "feature-copy",
        "feature-deepcopy",
        "plan-copy",
    ],
)
def test_split_markers_survive_options_and_feature_lifecycle(operation: str) -> None:
    feature = _feature(Options(context={"top_k": 2}), {"top_k"})
    source = feature.options
    if operation == "copy":
        result = copy.copy(source)
    elif operation == "deepcopy":
        result = copy.deepcopy(source)
    elif operation == "rebuild":
        result = source.rebuild(dict(source.group), dict(source.context))
    elif operation == "fork":
        result = source._fork()
    elif operation == "adopt":
        result = Options()
        result._adopt(source)
    elif operation == "defaults":
        result = SplitContextDefaults.options_with_defaults(source)
        assert result.context["padding"] == "filled"
    elif operation == "pickle":
        result = pickle.loads(pickle.dumps(feature)).options  # nosec B301
    elif operation == "feature-copy":
        result = copy.copy(feature).options
    elif operation == "feature-deepcopy":
        result = copy.deepcopy(feature).options
    else:
        feature.freeze()
        result = feature.copy_for_plan().options

    assert result.split_context_keys == frozenset({"top_k"})
    assert result.inherited_context_keys == frozenset()
    assert result.twin_split_keys == frozenset()
    assert result.context["top_k"] == 2
    assert source.context == {"top_k": 2}


def test_forwarding_preview_keeps_explicit_keys_separate_from_inherited_keys() -> None:
    source = Options(context={"top_k": 2})
    feature = _feature(source, {"top_k"}, inherit_context_keys={"tenant"})
    consumer = Options(group={"mode": "fast"}, context={"tenant": "one"})

    preview = feature.options_forwarded_from(consumer, SplitContextDefaults)

    assert preview.context == {"top_k": 2, "tenant": "one"}
    assert preview.group == {"mode": "fast"}
    assert preview.split_context_keys == frozenset({"top_k", "tenant"})
    assert preview.inherited_context_keys == frozenset({"tenant"})
    assert preview.twin_split_keys == frozenset()
    assert feature.options.context == source.context == {"top_k": 2}
    assert feature.options.split_context_keys == frozenset({"top_k"})
    assert source.split_context_keys == consumer.split_context_keys == frozenset()


def test_split_markers_do_not_change_feature_identity_but_do_change_resolution_inputs() -> None:
    options = Options(context={"top_k": 2})
    plain = Feature("split_context_feature", options)
    marked = _feature(options, {"top_k"})

    assert plain == marked
    assert hash(plain) == hash(marked)
    assert resolution_key(plain) != resolution_key(marked)
    assert resolution_key(plain.options) != resolution_key(marked.options)
